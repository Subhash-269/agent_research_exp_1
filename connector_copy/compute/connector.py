#!/usr/bin/env python3
"""Drive a remote compute runner from your laptop.

    python connector.py check                      # config + connectivity
    python connector.py preflight                  # survey the cluster (hpc only)
    python connector.py submit probe_gpu.sbatch    # stage + start, record the run id
    python connector.py status                     # what is queued or ran today
    python connector.py watch <id>                 # poll until it finishes
    python connector.py logs <id>                  # stdout
    python connector.py fetch <id>                 # pull artifacts locally
    python connector.py cancel <id>                # stop it
    python connector.py ledger                     # GPU-hours, cumulative

Runner comes from connector_config.json ("default"), overridable with
--runner. Requires only `ssh` on PATH for the hpc runner; no pip installs.

Nothing is held in process memory: run ids are written to jobs.jsonl at submit
time, so a later invocation recovers state from the runner alone. Every action
is appended to audit.log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runners import REGISTRY, Runner, RunnerError

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "connector_config.json"
EXAMPLE = HERE / "connector_config.example.json"


# ------------------------------------------------------------------- config

def load_runner(name: str | None) -> Runner:
    if not CONFIG.exists():
        raise RunnerError(
            f"{CONFIG.name} not found.\n"
            f"  cp {EXAMPLE.name} {CONFIG.name}\n"
            f"  then fill in user and account.")
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{CONFIG.name} is not valid JSON: {exc}")

    chosen = name or cfg.get("default")
    if not chosen:
        raise RunnerError(f'{CONFIG.name} has no "default" and no --runner given')
    if chosen not in REGISTRY:
        raise RunnerError(f"unknown runner {chosen!r}; "
                           f"known: {', '.join(REGISTRY)}")

    settings = (cfg.get("runners") or {}).get(chosen)
    if settings is None:
        raise RunnerError(f'{CONFIG.name} has no settings for runner "{chosen}"')

    return REGISTRY[chosen](settings)


def resolve_script(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else (HERE / p)


# ------------------------------------------------------------------ commands

def cmd_check(be: Runner, args) -> int:
    print(f"config : {CONFIG}")
    print(f"runner : {be.name}")
    be.connect()
    print(f"target : {be.describe()}")
    print("\nOK")
    return 0


def cmd_preflight(be: Runner, args) -> int:
    script = HERE / "preflight_login.sh"
    if not script.exists():
        raise RunnerError(f"{script.name} not found")
    if be.name != "hpc":
        raise RunnerError("preflight is Slurm-specific; use --runner hpc")
    be.connect()
    text = script.read_text(encoding="utf-8").replace("\r\n", "\n")
    proc = be._exec(be._ssh_args() + ["bash -s"], text, 600)
    be.audit("preflight", script.name, proc.returncode)
    print(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    return proc.returncode


def cmd_submit(be: Runner, args) -> int:
    be.connect()
    script = resolve_script(args.script)
    opts = {k: getattr(args, k) for k in
            ("partition", "gres", "account", "time", "array", "name", "label")}
    run_id = be.submit(script, {k: v for k, v in opts.items() if v})
    print(f"submitted {run_id}  ({script.name})  runner={be.name}")
    print(f"  watch : python connector.py watch {run_id}")
    print(f"  logs  : python connector.py logs {run_id}")
    return 0


def cmd_status(be: Runner, args) -> int:
    be.connect()
    if be.name != "hpc":
        raise RunnerError("status overview is Slurm-specific so far")
    print("=== queue ===")
    out = be._run('squeue -u $(whoami) -o "%.12i %.14P %.18j %.9T %.11M %.6D %R"',
                  check=False, action="status").stdout.strip()
    print(out if out.count("\n") else out + "\n  (nothing queued or running)")
    print("\n=== finished recently ===")
    out = be._run(f'sacct -u $(whoami) -S {args.since} '
                  '--format=JobID,JobName%20,State,Elapsed,NodeList -P '
                  '| grep -v "\\.batch\\|\\.extern"',
                  check=False, action="status").stdout.strip()
    print(out or "  (none)")
    return 0


def cmd_watch(be: Runner, args) -> int:
    be.connect()
    final = be.watch(args.run_id, interval=args.interval, timeout=args.timeout)
    print(f"\n{args.run_id}: {final}")
    if be.name == "hpc":
        out = be._run(f"sacct -j {args.run_id} "
                      "--format=JobID,State,Elapsed,TotalCPU,MaxRSS,ReqTRES%40 -P | head -3",
                      check=False, action="status").stdout.strip()
        print(out)
    return 0


def cmd_logs(be: Runner, args) -> int:
    be.connect()
    print(be.logs(args.run_id, tail=args.tail))
    return 0


def cmd_fetch(be: Runner, args) -> int:
    be.connect()
    got = be.fetch(args.run_id, Path(args.dest) if args.dest else None)
    if not got:
        print(f"nothing found for {args.run_id}")
        return 1
    for p in got:
        print(f"  {p}")
    return 0


def cmd_cancel(be: Runner, args) -> int:
    be.connect()
    be.cancel(args.run_id)
    print(f"cancelled {args.run_id}")
    return 0


def cmd_ledger(be: Runner, args) -> int:
    """Compute spent, cumulative, in COMPLETION order.

    Completion order matters: charging in score order implies you knew which
    runs would succeed before paying for them, which makes any cost curve
    retrospectively optimistic.
    """
    runs = be.load_runs(be.name)
    if not runs:
        print(f"no runs recorded for runner {be.name}")
        return 1
    be.connect()
    labels = {r["job_id"]: (r.get("label") or r.get("script", "")) for r in runs}
    rows = be.cost([r["job_id"] for r in runs])
    if not rows:
        print("no cost data returned")
        return 1

    # unfinished runs sort last: they have consumed compute but have no end yet
    rows.sort(key=lambda r: (r["end"] in ("", "Unknown"), r["end"]))

    print(f"{'run':>10}  {'state':<11} {'gpus':>4} {'elapsed':>9} "
          f"{'gpu_hrs':>8} {'cumul':>8}  label")
    total = 0.0
    for r in rows:
        total += r["gpu_hours"]
        print(f"{r['run_id']:>10}  {r['state']:<11} {r['ngpu']:>4} "
              f"{r['elapsed_s']:>8.0f}s {r['gpu_hours']:>8.4f} {total:>8.4f}  "
              f"{labels.get(r['run_id'].split('_')[0], '')}")
    print(f"\ntotal: {total:.4f} GPU-hours across {len(rows)} runs")
    if any(r["ngpu"] == 0 for r in rows):
        print("note: runs showing 0 GPUs requested no gres — CPU time, not GPU time")
    return 0


# --------------------------------------------------------------------- entry

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="connector.py",
                                description="Drive a remote compute runner.")
    p.add_argument("--runner", choices=sorted(REGISTRY),
                   help="override connector_config.json default")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify config and connectivity")
    sub.add_parser("preflight", help="survey the cluster (hpc only)")

    s = sub.add_parser("submit", help="stage a job script and start it")
    s.add_argument("script")
    for flag in ("--partition", "--gres", "--account", "--time", "--name"):
        s.add_argument(flag)
    s.add_argument("--array", help="e.g. 0-18%%4 (throttled to 4 at a time)")
    s.add_argument("--label", default="", help="free-text tag recorded in the ledger")

    s = sub.add_parser("status", help="queue plus recent history")
    s.add_argument("--since", default="today")

    s = sub.add_parser("watch", help="poll until the run finishes")
    s.add_argument("run_id")
    s.add_argument("--interval", type=int, default=20)
    s.add_argument("--timeout", type=int, default=3600)

    s = sub.add_parser("logs", help="print a run's stdout")
    s.add_argument("run_id")
    s.add_argument("--tail", type=int, default=0, help="last N lines only")

    s = sub.add_parser("fetch", help="copy artifacts locally")
    s.add_argument("run_id")
    s.add_argument("--dest")

    s = sub.add_parser("cancel", help="stop a run")
    s.add_argument("run_id")

    sub.add_parser("ledger", help="GPU-hours, cumulative, completion order")
    return p


def main() -> int:
    args = build_parser().parse_args()
    handlers = {
        "check": cmd_check, "preflight": cmd_preflight, "submit": cmd_submit,
        "status": cmd_status, "watch": cmd_watch, "logs": cmd_logs,
        "fetch": cmd_fetch, "cancel": cmd_cancel, "ledger": cmd_ledger,
    }
    try:
        be = load_runner(args.runner)
        return handlers[args.command](be, args)
    except RunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted (the remote run is unaffected)\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
