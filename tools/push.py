#!/usr/bin/env python3
"""Copy a file authored on this machine to the cluster, through the connector.

R8: "Author code on this machine and transfer it. Do not create or edit code
directly on the remote machine." `connector.py submit` transfers the one
.sbatch it is given and nothing else, so the Python helpers a job calls need
their own transfer step. This is it.

Like rx.py, it goes through the hpc runner's own `_put`, which strips CRLF
(S9) and writes a `push` line into compute/audit.log, so a file arriving on
the cluster is visible in the same trail as every other action.

Usage:
    python push.py <local path> <remote absolute path>
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

CONFIG = COMPUTE / "connector_config.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("remote", help="absolute path on the cluster")
    args = ap.parse_args()

    local = Path(args.local)
    if not local.is_file():
        sys.stderr.write(f"error: no such file: {local}\n")
        return 2
    if not args.remote.startswith("/"):
        sys.stderr.write("error: remote path must be absolute\n")
        return 2

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    chosen = cfg.get("default")
    be = REGISTRY[chosen]((cfg.get("runners") or {}).get(chosen))
    be.connect()
    be._put(local, args.remote)
    print(f"pushed {local} -> {args.remote}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2)
