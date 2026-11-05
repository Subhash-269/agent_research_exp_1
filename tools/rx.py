#!/usr/bin/env python3
"""Run one read-only command on the cluster through the connector's own runner.

Why this exists: connector.py exposes submit/status/logs/fetch but no generic
remote exec, and several things I must establish before spending GPU-hours
(quota evidence for S1, whether emit.py is staged, whether prior state sits
where I intend to work for S16) are one-line questions that do not justify a
queued job each.

It does NOT shell out to ssh itself. It imports the connector's hpc runner and
calls `_run`, which is the method that appends to compute/audit.log. Going
around that method would break the audit trail the operator reads; going
through it means every command issued here shows up there like any other.

Usage:
    python rx.py "<command>"                 # print stdout/stderr and exit code
    python rx.py --save <file> "<command>"   # also write stdout to <file>

Exit code is the remote command's exit code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPUTE = HERE.parent / "connector_copy" / "compute"
sys.path.insert(0, str(COMPUTE))

from runners import REGISTRY, RunnerError  # noqa: E402

# source: connector.py, which reads this same file to build its runner.
CONFIG = COMPUTE / "connector_config.json"

# source: chosen by me, no external source. Long enough for an `sacct` sweep or
# a `du` over a scratch tree, short enough that a hung ssh does not stall a run.
TIMEOUT_S = 600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--save", help="write stdout to this path as well")
    ap.add_argument("--action", default="probe",
                    help="label written into audit.log for this command")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    chosen = cfg.get("default")
    be = REGISTRY[chosen]((cfg.get("runners") or {}).get(chosen))
    be.connect()

    proc = be._run(args.command, check=False, timeout=args.timeout,
                   action=args.action)

    # Save BEFORE printing. This console is cp1252 and cluster tools emit
    # U+2713 and friends, so writing to stdout can raise UnicodeEncodeError -
    # which would otherwise lose the captured output as well as the display.
    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(proc.stdout, encoding="utf-8", newline="\n")

    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(proc.stdout.encode(enc, "replace").decode(enc, "replace"))
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr.encode(enc, "replace").decode(enc, "replace"))
    print(f"\n[rx] exit={proc.returncode}", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2)
