"""RunPod GPU cloud. Metered — an idle pod bills until it is terminated.

Unlike Slurm there is no scheduler and no job: create_pod gives you a machine,
not a queued run. So the run's state lives in a sentinel file the script writes
on exit, and `status()` reads it back over SSH. A pod that is up tells you
nothing about whether the work finished.

Two bookkeeping rules this runner has to respect, both learned the hard way:

  * `cost()` must return an `end` key. The ledger sorts on it, and a runner that
    omits it raises KeyError for every row it produced.
  * Terminations do not go in `jobs.jsonl`. The ledger counts rows there, so a
    second row per run double-counts every pod that was ever cancelled. They go
    in a sidecar only this runner reads.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import (Runner, RunnerError, DONE, FAILED, PENDING,
                   RUNNING, UNKNOWN, FETCH_DIR, RUNS_FILE)

try:
    import runpod as runpod_sdk
except ImportError:
    runpod_sdk = None

REMOTE_DIR = "/workspace/run"
SENTINEL = f"{REMOTE_DIR}/exit_code"
LOGFILE = f"{REMOTE_DIR}/run.log"

# Termination times live here, not in jobs.jsonl - see the module docstring.
TERM_FILE = RUNS_FILE.parent / "runpod_terminations.jsonl"


class RunPodRunner(Runner):
    name = "runpod"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._ssh: dict[str, tuple[str, int]] = {}   # run_id -> (ip, port)

    # --------------------------------------------------------------- connect

    def connect(self) -> None:
        if runpod_sdk is None:
            raise RunnerError("`runpod` package not installed. pip install runpod")

        key_env = self.cfg.get("api_key_env", "RUNPOD_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RunnerError(
                f"${key_env} is not set. Get a key from the RunPod console "
                f"(Settings -> API Keys) and export it. Never put it in the config file.")
        runpod_sdk.api_key = api_key

        self.pubkey = Path(os.path.expanduser(
            self.cfg.get("pubkey", "~/.ssh/id_ed25519.pub")))
        self.key = str(Path(os.path.expanduser(
            self.cfg.get("key", "~/.ssh/id_ed25519"))))
        if not self.pubkey.exists():
            raise RunnerError(
                f"public key not found: {self.pubkey}\n"
                f"  RunPod injects it into the pod so SSH works. ssh-keygen -t ed25519")
        if shutil.which("ssh") is None:
            raise RunnerError("`ssh` not found on PATH. Install OpenSSH.")

        try:
            runpod_sdk.get_pods()
        except Exception as exc:
            raise RunnerError(f"RunPod API key rejected or unreachable: {exc}") from exc

        self.audit("connect", f"api ok, gpu={self.cfg.get('gpu_type')}", 0)

    def describe(self) -> str:
        return (f"runpod  gpu={self.cfg.get('gpu_type')}  "
                f"image={self.cfg.get('image')}  cloud={self.cfg.get('cloud_type', 'SECURE')}")

    # -------------------------------------------------------------- plumbing

    def _ssh_for(self, run_id: str) -> tuple[str, int]:
        """IP and port of the pod's SSH service, once RunPod has assigned one."""
        if run_id in self._ssh:
            return self._ssh[run_id]
        pod = runpod_sdk.get_pod(run_id)
        if not pod:
            raise RunnerError(f"pod {run_id} not found (terminated?)")
        for p in (pod.get("runtime") or {}).get("ports") or []:
            if p.get("privatePort") == 22 and p.get("ip") and p.get("publicPort"):
                self._ssh[run_id] = (p["ip"], int(p["publicPort"]))
                return self._ssh[run_id]
        raise RunnerError(f"pod {run_id} has no SSH port yet")

    def _run(self, run_id: str, command: str, *, stdin: str | None = None,
             check: bool = True, timeout: int = 300, action: str = "remote"):
        """Every command to the pod goes through here, so every command is logged."""
        ip, port = self._ssh_for(run_id)
        args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
                "-i", self.key, "-p", str(port), f"root@{ip}", command]
        proc = subprocess.run(
            args,
            input=stdin.encode("utf-8") if stdin is not None else None,
            capture_output=True, timeout=timeout)
        proc.stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        proc.stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        self.audit(action, command.replace("\n", " ")[:150], proc.returncode)
        if check and proc.returncode != 0:
            raise RunnerError(f"remote command failed ({proc.returncode}): "
                               f"{command[:80]}\n{proc.stderr.strip()}")
        return proc

    def _await_ssh(self, run_id: str, timeout: int = 420) -> None:
        """A pod is billable before it is reachable. Wait for sshd, not for the API."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._run(run_id, "echo ok", check=False,
                             timeout=30, action="wait").returncode == 0:
                    return
            except RunnerError:
                pass          # no port assigned yet
            time.sleep(10)
        raise RunnerError(
            f"pod {run_id} never became reachable over SSH.\n"
            f"  It is still billing — terminate it: connector.py cancel {run_id}")

    # ---------------------------------------------------------------- submit

    def submit(self, script: Path, opts: dict) -> str:
        if not script.exists():
            raise RunnerError(f"script not found: {script}")

        pod = runpod_sdk.create_pod(
            name=opts.get("name") or f"capstone-{script.stem}",
            image_name=opts.get("image") or self.cfg.get(
                "image", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
            gpu_type_id=opts.get("gpu_type") or self.cfg.get("gpu_type"),
            gpu_count=int(opts.get("gpu_count") or self.cfg.get("gpu_count", 1)),
            cloud_type=self.cfg.get("cloud_type", "SECURE"),
            container_disk_in_gb=int(self.cfg.get("disk_gb", 30)),
            volume_in_gb=int(self.cfg.get("volume_gb", 0)),
            volume_mount_path="/workspace",
            ports="22/tcp",
            env={"PUBLIC_KEY": self.pubkey.read_text().strip()},
            support_public_ip=True,
        )
        run_id = pod["id"]
        self.audit("submit", f"pod {run_id} created", 0)
        print(f"  pod {run_id} created — billing has started")

        try:
            self._await_ssh(run_id)

            body = script.read_text(encoding="utf-8").replace("\r\n", "\n")
            self._run(run_id, f"mkdir -p {REMOTE_DIR} && cat > {REMOTE_DIR}/{script.name}",
                      stdin=body, action="push")
            self._run(run_id, f"chmod +x {REMOTE_DIR}/{script.name}", action="push")

            # nohup so the run survives the SSH session; the sentinel is written
            # last and is the only thing status() trusts.
            self._run(run_id, (
                f"cd {REMOTE_DIR} && rm -f {SENTINEL} && "
                f"nohup bash -c './{script.name} > {LOGFILE} 2>&1; "
                f"echo $? > {SENTINEL}' >/dev/null 2>&1 & echo started"),
                action="submit")
        except Exception:
            print(f"  ! submit failed — terminating {run_id} so it stops billing")
            self.cancel(run_id)
            raise

        self.record_run({
            "job_id": run_id,
            "script": script.name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_type": opts.get("gpu_type") or self.cfg.get("gpu_type"),
            "gpu_count": opts.get("gpu_count") or self.cfg.get("gpu_count", 1),
            "cost_per_hr": pod.get("costPerHr"),
            "label": opts.get("label") or "",
        })
        return run_id

    # ---------------------------------------------------------------- status

    def status(self, run_id: str) -> str:
        pod = runpod_sdk.get_pod(run_id)
        if not pod:
            return UNKNOWN                      # terminated, or never existed
        desired = (pod.get("desiredStatus") or "").upper()
        if desired in ("TERMINATED", "EXITED"):
            return UNKNOWN
        if not (pod.get("runtime") or {}).get("ports"):
            return PENDING                      # still provisioning

        proc = self._run(run_id, f"cat {SENTINEL} 2>/dev/null",
                         check=False, timeout=60, action="status")
        code = proc.stdout.strip()
        if proc.returncode != 0:
            return PENDING                      # unreachable; assume still coming up
        if not code:
            return RUNNING
        return DONE if code == "0" else FAILED

    # ------------------------------------------------------------ logs/fetch

    def logs(self, run_id: str, tail: int = 0) -> str:
        cmd = f"tail -n {tail} {LOGFILE}" if tail else f"cat {LOGFILE}"
        proc = self._run(run_id, f"{cmd} 2>/dev/null", check=False,
                         timeout=120, action="logs")
        return proc.stdout or f"no log yet for {run_id} at {LOGFILE}"

    def fetch(self, run_id: str, dest: Path | None = None) -> list[Path]:
        dest = Path(dest or FETCH_DIR)
        dest.mkdir(parents=True, exist_ok=True)
        got: list[Path] = []

        listing = self._run(run_id, f"ls {REMOTE_DIR}/outputs/* {LOGFILE} 2>/dev/null",
                            check=False, action="fetch").stdout.split()
        for remote in listing:
            body = self._run(run_id, f"cat {remote}", check=False,
                             timeout=300, action="fetch").stdout
            local = dest / remote.rsplit("/", 1)[-1]
            local.write_text(body, encoding="utf-8")
            got.append(local)

        self.audit("fetch", f"{run_id} -> {len(got)} file(s)", 0)
        if got:
            print(f"  fetched {len(got)} file(s) — the pod is still billing until cancelled")
        return got

    # ---------------------------------------------------------------- cancel

    def cancel(self, run_id: str) -> None:
        """Terminate the pod. Safe to call twice — an already-gone pod is success,
        not an error, because the only thing that matters is that it stopped billing."""
        self._ssh.pop(run_id, None)
        try:
            if runpod_sdk.get_pod(run_id) is None:
                self.audit("cancel", f"{run_id} already gone", 0)
                return
        except Exception:
            pass                       # fall through and try terminating anyway

        try:
            runpod_sdk.terminate_pod(run_id)
        except Exception as exc:
            if "not found" in str(exc).lower():
                self.audit("cancel", f"{run_id} already gone", 0)
                return
            self.audit("cancel", f"{run_id} FAILED: {exc}", 1)
            raise RunnerError(
                f"could not terminate {run_id}: {exc}\n"
                f"  It may still be billing. Check the RunPod console and kill it by hand."
            ) from exc

        self._record_termination(run_id)
        self.audit("cancel", f"{run_id} terminated", 0)

    # ---------------------------------------------------------- terminations

    def _record_termination(self, run_id: str) -> None:
        """When this pod stopped billing.

        Deliberately not `record_run()`: that appends to jobs.jsonl, which the
        ledger treats as one row per run. A termination row there makes every
        cancelled pod appear twice.
        """
        try:
            with TERM_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "job_id": run_id,
                    "terminated_utc": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except OSError:
            pass          # bookkeeping must never take down a termination

    @staticmethod
    def _terminations() -> dict[str, str]:
        """run_id -> termination timestamp. Last write wins."""
        if not TERM_FILE.exists():
            return {}
        out: dict[str, str] = {}
        for line in TERM_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("job_id") and rec.get("terminated_utc"):
                out[rec["job_id"]] = rec["terminated_utc"]
        return out

    # ------------------------------------------------------------------ cost

    def cost(self, run_ids: list[str]) -> list[dict]:
        """Billed wall-clock: creation to termination, including boot and idle.

        Not the duration of the work — a pod that booted for four minutes and sat
        idle for an hour after finishing billed for all of it. Start times come
        from jobs.jsonl and end times from the termination sidecar, because the
        API forgets a pod once it is terminated.

        Every row carries `end`, matching the Slurm runner, because the ledger
        sorts on that key.
        """
        kills = self._terminations()
        submits = {r["job_id"]: r for r in self.load_runs(self.name)
                   if r.get("submitted_utc")}

        rows = []
        for run_id in run_ids:
            rec = submits.get(run_id)
            if not rec:
                rows.append({"run_id": run_id, "state": "unrecorded", "end": "",
                             "elapsed_s": 0.0, "ngpu": 0, "gpu_hours": 0.0,
                             "usd": 0.0, "note": "not in jobs.jsonl"})
                continue

            start = datetime.fromisoformat(rec["submitted_utc"])
            end_s = kills.get(run_id)
            if end_s:
                end, state = datetime.fromisoformat(end_s), "terminated"
            else:
                live = None
                try:
                    live = runpod_sdk.get_pod(run_id)
                except Exception:
                    pass
                # No termination record and no live pod: the end time is unknown,
                # so this is a floor, not a figure. Say so rather than guessing.
                end = datetime.now(timezone.utc)
                state = "still-running" if live else "unknown-end"

            secs = max(0.0, (end - start).total_seconds())
            ngpu = int(rec.get("gpu_count") or 1)
            rate = float(rec.get("cost_per_hr") or 0)
            rows.append({
                "run_id": run_id,
                "state": state,
                "end": end.isoformat(timespec="seconds") if end_s else "",
                "elapsed_s": secs,
                "ngpu": ngpu,
                "gpu_hours": secs * ngpu / 3600.0,
                "usd": secs / 3600.0 * rate,
            })
        return rows
