"""Runner interface shared by every compute target.

Subclasses hold their own connection state in `self`. Everything a runner does
goes through a method on this class, which is what keeps `audit.log` complete -
a runner that shells out directly, bypassing these methods, breaks that promise.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = ROOT / "jobs.jsonl"
AUDIT_FILE = ROOT / "audit.log"
FETCH_DIR = ROOT / "fetched"

# normalized status vocabulary - every runner maps its own states onto these
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
UNKNOWN = "unknown"
TERMINAL = (DONE, FAILED)


class RunnerError(RuntimeError):
    """Raised for configuration or remote-execution failures."""


class Runner(ABC):
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ------------------------------------------------------------------ audit

    def audit(self, action: str, detail: str = "", rc: int | None = None) -> None:
        """Append-only record of every action taken against this runner.

        The point is that a human can read what was actually done, in order,
        without trusting the agent's own summary of it. Never let a logging
        failure take down the run.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rc_s = "-" if rc is None else str(rc)
        line = f"{stamp}  {self.name:<7} {action:<9} rc={rc_s:<4} {detail}"
        try:
            with AUDIT_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line.rstrip() + "\n")
        except OSError:
            pass

    # -------------------------------------------------------------- run record

    def record_run(self, entry: dict) -> None:
        """Persist a run id at submit time.

        A run may outlive the session that started it, so nothing is held in
        process memory: a later invocation recovers state from the runner alone.
        """
        entry = {"runner": self.name, **entry}
        with RUNS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    @staticmethod
    def load_runs(runner: str | None = None) -> list[dict]:
        if not RUNS_FILE.exists():
            return []
        out = []
        for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if runner is None or rec.get("runner") == runner:
                out.append(rec)
        return out

    # ----------------------------------------------------------- generic poll

    def watch(self, run_id: str, interval: int = 20, timeout: int = 3600) -> str:
        """Poll until the run reaches a terminal state. Written once for all runners.

        Also surfaces a blocking reason while pending - a request the scheduler
        accepted but will never schedule looks identical to a normal queue wait
        unless you ask why it is waiting.
        """
        print(f"watching {run_id} on {self.name} "
              f"(poll {interval}s; Ctrl-C stops watching, not the run)")
        deadline = time.time() + timeout
        last = None
        warned = False
        while time.time() < deadline:
            state = self.status(run_id)
            if state != last:
                print(f"  {datetime.now():%H:%M:%S}  {state}")
                last = state
            if state in TERMINAL:
                return state
            if state == PENDING and not warned:
                reason = self.blocking_reason(run_id)
                if reason:
                    print(f"  ! will never schedule as submitted: {reason}")
                    self.audit("blocked", f"{run_id} {reason}")
                    warned = True
            time.sleep(interval)
        print(f"  still {last} after {timeout}s - giving up on watching, not on the run")
        return last or UNKNOWN

    def blocking_reason(self, run_id: str) -> str | None:
        """Why a pending run cannot be scheduled, if it never will be.

        Default: no opinion. Runners that can tell the difference override it.
        """
        return None

    # -------------------------------------------------------------- interface

    @abstractmethod
    def connect(self) -> None:
        """Establish and verify access. Must fail loudly, not lazily."""

    @abstractmethod
    def submit(self, script: Path, opts: dict) -> str:
        """Stage the script and start it. Returns a run id."""

    @abstractmethod
    def status(self, run_id: str) -> str:
        """One of PENDING / RUNNING / DONE / FAILED / UNKNOWN."""

    @abstractmethod
    def logs(self, run_id: str, tail: int = 0) -> str:
        """Standard output of the run so far."""

    @abstractmethod
    def fetch(self, run_id: str, dest: Path) -> list[Path]:
        """Copy the run's artifacts to the local machine."""

    @abstractmethod
    def cancel(self, run_id: str) -> None:
        """Stop the run. On metered runners this is what stops the billing."""

    def cost(self, run_ids: list[str]) -> list[dict]:
        """Per-run resource usage, for the ledger. Optional."""
        raise NotImplementedError(f"{self.name} runner reports no cost data")

    def describe(self) -> str:
        """One-line summary of where this runner points, for `check`."""
        return self.name
