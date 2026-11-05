"""Compute runners.

Each runner is a `Runner` subclass exposing the same six methods. Adding one
means adding a module and a REGISTRY entry — the CLI never branches on runner.
"""
from .aws import EC2Runner
from .base import (Runner, RunnerError, DONE, FAILED, PENDING, RUNNING,
                   TERMINAL, UNKNOWN)
from .hpc import SlurmSSHRunner
from .runpod import RunPodRunner

REGISTRY: dict[str, type[Runner]] = {
    "hpc": SlurmSSHRunner,
    "runpod": RunPodRunner,
    "aws": EC2Runner,
}

__all__ = ["Runner", "RunnerError", "REGISTRY", "PENDING", "RUNNING",
           "DONE", "FAILED", "UNKNOWN", "TERMINAL"]
