#!/usr/bin/env python3
"""Charge finished jobs to the run's ledger through emit.job_cost. Runs in a job.

AGENTS.md requires a job_cost record for EVERY job, including ones that failed
and ones cancelled while pending, because a curve that omits a failed
allocation is retrospectively optimistic. It also notes that a job's own sacct
figures are not final while it runs, so a job cannot charge itself - some later
process has to. This is that process, and it is meant to run on
--partition=short with --gres=none, where it costs no GPU-hours.

S11: the scheduler's stdout is saved to disk, not just its exit code. Every
job's sacct line is written to $RUN_ROOT/sacct/<id>.psv before anything is
derived from it, so every elapsed time in the ledger is auditable from a file.

Charging twice would double-count, so jobs that already have a job_ended record
in this run's events are skipped, and the skip is printed.

Usage:
    python charge_jobs.py 11637680 11638131 11638336 ...
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import emit

# The exact field list AGENTS.md names for job_cost, plus ReqTRES, which is
# where the GPU count lives.
SACCT_FIELDS = "JobID,State,ElapsedRaw,Start,End,Partition,NodeList,ReqTRES"


def already_charged() -> set[str]:
    charged: set[str] = set()
    events = Path(os.environ["RUN_DIR"]) / "events"
    if not events.is_dir():
        return charged
    import json
    for p in events.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "job_ended":
                charged.add(str(r.get("job_id")))
    return charged


def gpus_from_reqtres(req: str) -> int:
    """GPU count from a ReqTRES string such as 'billing=112,cpu=12,gres/gpu=1'.

    A job submitted with --gres=none has no gres/gpu term at all, and 0 is the
    correct answer for it: a CPU job holds no GPU and costs no GPU-hours.
    """
    m = re.search(r"gres/gpu(?::[a-z0-9_]+)?=(\d+)", req or "")
    return int(m.group(1)) if m else 0


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: charge_jobs.py <jobid> [<jobid> ...]")
        return 2

    run_root = os.environ.get("RUN_ROOT")
    if not run_root:
        raise SystemExit("RUN_ROOT is not set; the job script sets it")
    sacct_dir = Path(run_root) / "sacct"
    sacct_dir.mkdir(parents=True, exist_ok=True)

    charged = already_charged()
    print(f"[charge] already on record: {sorted(charged) or 'none'}")

    done = skipped = 0
    with emit.stage("admin", note="charging finished jobs from sacct"):
        for jid in argv:
            jid = jid.strip()
            if jid in charged:
                print(f"[charge] {jid}: already has a job_ended record, skipping")
                skipped += 1
                continue

            proc = subprocess.run(
                ["sacct", "-j", jid, "-X", "-P", "-n", "-o", SACCT_FIELDS],
                capture_output=True, text=True,
            )
            out = proc.stdout.strip()
            # S11: save the scheduler's output, not just its exit code.
            (sacct_dir / f"{jid}.psv").write_text(
                proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
                encoding="utf-8",
            )
            if proc.returncode != 0 or not out:
                # A missing sacct row is a hard stop, not a skipped record: an
                # uncharged job is a gap in the x axis of the curve (S12).
                raise SystemExit(
                    f"[charge] sacct for {jid} returned rc={proc.returncode} "
                    f"with {len(out)} bytes of stdout. Refusing to leave it uncharged.\n"
                    f"stderr: {proc.stderr.strip()}"
                )

            line = out.splitlines()[0]
            parts = line.split("|")
            if len(parts) < 8:
                raise SystemExit(f"[charge] {jid}: unparseable sacct line: {line!r}")
            job_id, state, elapsed, start, end, partition, node, reqtres = parts[:8]

            # A job cancelled before it ever started has Start=Unknown and
            # ElapsedRaw=0. That is a real, chargeable fact, not a reason to
            # drop the row.
            elapsed_s = int(elapsed or 0)
            gpus = gpus_from_reqtres(reqtres)
            rec = emit.job_cost(
                job_id.split(".")[0], state=state, elapsed_s=elapsed_s, gpus=gpus,
                start_utc=None if start in ("", "Unknown") else start,
                end_utc=None if end in ("", "Unknown") else end,
                partition=partition or None, node=node or None,
            )
            print(f"[charge] {job_id}: {state} elapsed={elapsed_s}s gpus={gpus} "
                  f"-> {rec['gpu_hours']} GPU-hours "
                  f"({elapsed_s} s x {gpus} GPU / 3600)")
            done += 1

    print(f"[charge] charged {done}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
