"""Slurm over SSH. The only fully implemented runner.

Verified against NEU Explorer: key auth with no 2FA, `sbatch` / `squeue` /
`sacct` / `scancel`, real GPU jobs recovered end to end.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import (Runner, RunnerError, DONE, FAILED, PENDING,
                   RUNNING, UNKNOWN, FETCH_DIR)

# Slurm states that mean the job is over, one way or another
_FINISHED_OK = ("COMPLETED",)
_FINISHED_BAD = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                 "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")

# Pending reasons that will never clear on their own - the request itself is
# impossible, so the job sits in the queue forever looking like a normal wait.
_NEVER_CLEARS = ("QOSMaxGRESPerJob", "QOSMaxGRESPerUser", "QOSMaxJobsPerUser",
                 "AssocMaxJobsLimit", "QOSMaxWallDurationPerJobLimit",
                 "PartitionTimeLimit", "PartitionConfig", "InvalidAccount",
                 "QOSMaxCpuPerUserLimit", "ReqNodeNotAvail")


class SlurmSSHRunner(Runner):
    name = "hpc"

    REQUIRED = ("host", "user", "key")

    # --------------------------------------------------------------- connect

    def connect(self) -> None:
        missing = [k for k in self.REQUIRED if not self.cfg.get(k)]
        if missing:
            raise RunnerError(
                f"connector_config.json → runners.hpc is missing: {', '.join(missing)}")

        key = Path(os.path.expanduser(self.cfg["key"]))
        if not key.exists():
            raise RunnerError(
                f"SSH key not found: {key}\n"
                f"  Generate one and install it on the cluster — RUNBOOK.md section 1.")
        if shutil.which("ssh") is None:
            raise RunnerError("`ssh` not found on PATH. Install OpenSSH.")

        self.key = str(key)
        self.target = f"{self.cfg['user']}@{self.cfg['host']}"
        self.project_dir = self.cfg.get("project_dir", "$HOME/hpcjobs")

        proc = self._run("echo ok", check=False)
        if proc.returncode != 0:
            raise RunnerError(
                f"cannot reach {self.target} with key auth.\n"
                f"  {proc.stderr.strip()}\n"
                f"  If it asks for a password, key auth is not working — RUNBOOK.md section 6.")

        self._run(f"mkdir -p {self.project_dir}/jobs {self.project_dir}/logs")

    def describe(self) -> str:
        return (f"{self.target}  account={self.cfg.get('account') or '(unset!)'}  "
                f"partition={self.cfg.get('partition')}  gres={self.cfg.get('gres')}")

    # -------------------------------------------------------------- plumbing

    def _ssh_args(self) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=30", "-i", self.key, self.target]

    def _exec(self, args: list[str], stdin: str | None, timeout: int):
        """Talk to the child in binary, always.

        With text=True, Python opens the stdin pipe in text mode and on Windows
        rewrites every \\n back to \\r\\n — silently reintroducing DOS line
        endings into scripts we just stripped them from, which sbatch rejects.
        """
        proc = subprocess.run(
            args,
            input=stdin.encode("utf-8") if stdin is not None else None,
            capture_output=True,
            timeout=timeout,
        )
        proc.stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        proc.stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        return proc

    def _run(self, command: str, *, stdin: str | None = None,
             check: bool = True, timeout: int = 300, action: str = "remote"):
        """Every command to the cluster goes through here, so every command is logged.

        There is deliberately no public escape hatch: a caller that wants to run
        something ad hoc still lands in the audit log.
        """
        exports = self._env_exports()
        proc = self._exec(self._ssh_args() + [exports + command], stdin, timeout)
        self.audit(action, command.replace("\n", " ")[:150], proc.returncode)
        if check and proc.returncode != 0:
            raise RunnerError(f"remote command failed ({proc.returncode}): "
                               f"{command[:80]}\n{proc.stderr.strip()}")
        return proc

    def _env_exports(self) -> str:
        """Config values the job script should see, exported ahead of the command.

        Kept literal (`$HOME`, `$USER`) so the remote shell expands them — doing
        it locally would bake in this machine's paths.
        """
        pairs = [("HPC_PROJECT_DIR", self.project_dir)]
        for key in ("hf_home", "pip_cache_dir", "torch_home"):
            if self.cfg.get(key):
                pairs.append((key.upper(), self.cfg[key]))
        return "; ".join(f"export {k}={v}" for k, v in pairs) + "; "

    def _put(self, local: Path, remote_path: str) -> None:
        text = local.read_text(encoding="utf-8").replace("\r\n", "\n")
        parent = remote_path.rsplit("/", 1)[0]
        self._run(f"mkdir -p {parent} && cat > {remote_path}",
                  stdin=text, action="push")
        self.audit("push", f"{local.name} -> {remote_path} ({len(text)} B)", 0)

    # ---------------------------------------------------------------- submit

    def submit(self, script: Path, opts: dict) -> str:
        if not script.exists():
            raise RunnerError(f"script not found: {script}")

        account = opts.get("account") or self.cfg.get("account")
        if not account:
            raise RunnerError("no Slurm account set (config `account` or --account); "
                               "sbatch will reject the job")

        remote = f"{self.project_dir}/jobs/{script.name}"
        self._put(script, remote)

        flags = [
            "--parsable",
            f"--partition={opts.get('partition') or self.cfg.get('partition', 'gpu-short')}",
            f"--gres={opts.get('gres') or self.cfg.get('gres', 'gpu:1')}",
            f"--account={account}",
            f"--time={opts.get('time') or self.cfg.get('time', '00:10:00')}",
            f"--output={self.project_dir}/logs/%x_%j.out",
            f"--error={self.project_dir}/logs/%x_%j.out",
            "--export=ALL",
        ]
        if opts.get("array"):
            flags.append(f"--array={opts['array']}")
        if opts.get("name"):
            flags.append(f"--job-name={opts['name']}")

        proc = self._run(f"sbatch {' '.join(flags)} {remote}",
                         check=False, action="submit")
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 or not out:
            raise RunnerError(f"sbatch rejected the job: "
                               f"{(proc.stderr or out).strip()}")

        run_id = out.split(";")[0]
        self.record_run({
            "job_id": run_id,
            "script": script.name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "partition": opts.get("partition") or self.cfg.get("partition"),
            "gres": opts.get("gres") or self.cfg.get("gres"),
            "array": opts.get("array") or "",
            "label": opts.get("label") or "",
        })

        # Accepting a job is not the same as being able to run it. An over-QOS
        # request is accepted, gets an id, and then pends forever. Check now
        # rather than discovering it an hour later - but the scheduler does not
        # fill in a reason the instant sbatch returns, so give it a few seconds.
        reason = self._await_blocking_reason(run_id)
        if reason:
            print(f"  ! accepted but unschedulable as submitted: {reason}")
            print(f"  ! it will pend indefinitely - cancel it or resubmit smaller")
            self.audit("blocked", f"{run_id} {reason}")

        return run_id

    # ---------------------------------------------------------------- status

    def status(self, run_id: str) -> str:
        q = self._run(f'squeue -h -j {run_id} -o "%T"',
                      check=False, action="status").stdout.strip()
        if q:
            first = q.splitlines()[0].strip()
            return PENDING if first in ("PENDING", "CONFIGURING") else RUNNING

        s = self._run(f"sacct -n -j {run_id} -o State -P",
                      check=False, action="status").stdout.strip()
        if not s:
            return UNKNOWN
        head = s.splitlines()[0].strip().split()[0]
        if head in _FINISHED_OK:
            return DONE
        if any(head.startswith(bad) for bad in _FINISHED_BAD):
            return FAILED
        return UNKNOWN

    def blocking_reason(self, run_id: str) -> str | None:
        """Distinguish a normal queue wait from a request that cannot ever run."""
        r = self._run(f'squeue -h -j {run_id} -o "%r"',
                      check=False, action="status").stdout.strip()
        if not r:
            return None
        reason = r.splitlines()[0].strip()
        return reason if any(reason.startswith(x) for x in _NEVER_CLEARS) else None

    def _await_blocking_reason(self, run_id: str, tries: int = 4,
                               gap: float = 4.0) -> str | None:
        """Same check, but tolerant of the scheduler's lag.

        Right after sbatch returns, `%r` is usually `None` or `Priority` even for
        a request that can never be satisfied; the real reason appears a few
        seconds later. Stop early if the job starts running or leaves the queue,
        since neither case is blocked.
        """
        for attempt in range(tries):
            if attempt:
                time.sleep(gap)
            reason = self.blocking_reason(run_id)
            if reason:
                return reason
            if self.status(run_id) != PENDING:
                return None
        return None

    # ------------------------------------------------------------ logs/fetch

    def _log_path(self, run_id: str) -> str | None:
        out = self._run(f"ls {self.project_dir}/logs/*_{run_id}.out 2>/dev/null | head -1",
                        check=False, action="logs").stdout.strip()
        return out or None

    def logs(self, run_id: str, tail: int = 0) -> str:
        path = self._log_path(run_id)
        if not path:
            return (f"no log yet for {run_id} under {self.project_dir}/logs/ "
                    f"(still pending, or it wrote elsewhere)")
        cmd = f"tail -n {tail} {path}" if tail else f"cat {path}"
        return self._run(cmd, check=False, timeout=120, action="logs").stdout

    def fetch(self, run_id: str, dest: Path | None = None) -> list[Path]:
        dest = Path(dest or FETCH_DIR)
        dest.mkdir(parents=True, exist_ok=True)
        got: list[Path] = []
        for pattern in (f"*_{run_id}.out", f"telemetry_{run_id}.csv",
                        f"heartbeat_{run_id}.log"):
            listing = self._run(f"ls {self.project_dir}/logs/{pattern} 2>/dev/null",
                                check=False, action="fetch").stdout.split()
            for remote in listing:
                body = self._run(f"cat {remote}", check=False,
                                 timeout=180, action="fetch").stdout
                local = dest / remote.rsplit("/", 1)[-1]
                local.write_text(body, encoding="utf-8")
                got.append(local)
        self.audit("fetch", f"{run_id} -> {len(got)} file(s)", 0)
        return got

    # ---------------------------------------------------------------- cancel

    def cancel(self, run_id: str) -> None:
        self._run(f"scancel {run_id}", check=False, action="cancel")

    # ------------------------------------------------------------------ cost

    def cost(self, run_ids: list[str]) -> list[dict]:
        """GPU-seconds per job, from sacct. Caller orders and accumulates."""
        if not run_ids:
            return []
        ids = ",".join(run_ids)
        out = self._run(f"sacct -j {ids} --format=JobID,State,Start,End,Elapsed,ReqTRES%60 -P",
                        check=False, timeout=180, action="cost").stdout

        rows = []
        for line in out.splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 6:
                continue
            job_id, state, start, end, elapsed, req = parts[:6]
            if "." in job_id:          # skip .batch / .extern steps
                continue
            secs = _parse_elapsed(elapsed)
            ngpu = _parse_gpu_count(req)
            rows.append({
                "run_id": job_id, "state": state, "end": end,
                "elapsed_s": secs, "ngpu": ngpu,
                "gpu_hours": secs * ngpu / 3600.0,
            })
        return rows


# ------------------------------------------------------------------ parsing

def _parse_elapsed(text: str) -> float:
    """Slurm elapsed -> seconds. Format is [DD-]HH:MM:SS."""
    text = text.strip()
    if not text:
        return 0.0
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _parse_gpu_count(req_tres: str) -> int:
    """Pull N from a ReqTRES like 'billing=102,cpu=2,gres/gpu=1,mem=8G'.

    Slurm writes both `gres/gpu=2` and `gres/gpu:a100=2` depending on how the
    request was phrased, so handle each.
    """
    for field in req_tres.split(","):
        if field.startswith(("gres/gpu=", "gres/gpu:")):
            try:
                return int(field.split("=", 1)[1])
            except (ValueError, IndexError):
                return 0
    return 0
