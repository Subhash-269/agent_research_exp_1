"""AWS runner — NOT IMPLEMENTED.

Fill this in against the six methods on `Runner`. Same reasoning as
`runpod.py`: an honest stub beats an untested implementation.

Decisions left open, worth settling before writing code:

  * **EC2 or Batch.** EC2 means you manage the instance lifecycle yourself and
    there is no queue. Batch gives you a queue closer to Slurm's semantics, at
    the cost of more setup. The `Runner` interface fits either.
  * **Billing is wall-clock**, as with RunPod — from instance start to
    termination, including boot and idle. `cancel()` stops the meter.
  * **GPU instance quota defaults to zero** on new AWS accounts, and raising it
    is a support request that takes days. Check this before treating AWS as
    outage insurance; insurance that needs three days' notice is not insurance.
  * **Credentials never live in connector_config.json.** Use the standard AWS
    credential chain via `profile`; the config holds only the profile name,
    region, instance type and the path to a `.pem`.
"""
from __future__ import annotations

from pathlib import Path

from .base import Runner, RunnerError

_MSG = ("AWS runner is not implemented yet. "
        "Use --runner hpc, or implement runners/aws.py.")


class EC2Runner(Runner):
    name = "aws"

    def connect(self) -> None:
        raise RunnerError(_MSG)

    def submit(self, script: Path, opts: dict) -> str:
        raise RunnerError(_MSG)

    def status(self, run_id: str) -> str:
        raise RunnerError(_MSG)

    def logs(self, run_id: str, tail: int = 0) -> str:
        raise RunnerError(_MSG)

    def fetch(self, run_id: str, dest: Path) -> list[Path]:
        raise RunnerError(_MSG)

    def cancel(self, run_id: str) -> None:
        raise RunnerError(_MSG)
