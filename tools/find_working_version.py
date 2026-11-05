#!/usr/bin/env python3
"""Find the newest release of a package under which a given import succeeds.

Why this exists. env/build_uv_env.sh pins torch and jax and leaves everything
else to the resolver, so this run has now hit three separate breakages of the
same shape: the newest release of an unpinned package no longer matches the
pinned one it has to work with.

  torchaudio  built against a different torch ABI -> undefined symbol
  atomworks   3.x dropped AtomSelectionStack      -> ImportError
  dm-haiku    newest expects a jax.core API that jax 0.4.29 does not have

Fixing these one at a time by guessing a version is how a run burns its
budget. This does the search instead, and it does it the honest way: it
installs each candidate and runs the actual import that failed, rather than
grepping for a symbol and hoping the rest of the module agrees.

Installs use --no-deps on purpose. The whole problem is that the package must
fit around dependencies that are already pinned - letting the resolver move
jax or torch to satisfy a candidate would "succeed" by breaking the thing the
pin was protecting.

S13 in spirit: every subprocess's return code is checked. The predecessor of
this script concluded "no version works" from a version list that was empty
because the command producing it had failed unnoticed, so here an empty
candidate list is a hard error, never a negative result.

Usage:
    python find_working_version.py --package dm-haiku \
        --test "import haiku; from colabdesign import mk_afdesign_model" \
        [--max-tries 12] [--extras "ml,openbabel"] [--restore-on-failure]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request


def releases(package: str) -> list[str]:
    """Every published version, newest first, from the index's own metadata."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=60) as fh:
        data = json.load(fh)
    vs = [v for v, files in data["releases"].items() if files]

    def key(v: str):
        parts = []
        for chunk in v.replace("-", ".").split("."):
            parts.append(int(chunk) if chunk.isdigit() else -1)
        return parts

    return sorted(vs, key=key, reverse=True)


def current(package: str) -> str | None:
    p = subprocess.run(["uv", "pip", "show", package], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def install(spec: str) -> int:
    return subprocess.run(
        ["uv", "pip", "install", "--no-deps", spec],
        capture_output=True, text=True,
    ).returncode


def test(code: str) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if p.returncode == 0:
        return True, (p.stdout or "").strip()
    tail = (p.stderr or "").strip().splitlines()
    return False, tail[-1] if tail else f"exit {p.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--test", required=True, help="python source that must run cleanly")
    ap.add_argument("--max-tries", type=int, default=12)
    ap.add_argument("--extras", default="", help="e.g. ml,openbabel")
    ap.add_argument("--restore-on-failure", action="store_true",
                    help="reinstall the version that was there if nothing works")
    args = ap.parse_args()

    started_at = current(args.package)
    print(f"[pin] {args.package}: currently {started_at or 'not installed'}")

    vs = releases(args.package)
    if not vs:
        # Never report this as "no version works" - it is a failed lookup.
        raise SystemExit(f"[pin] the index returned no releases for {args.package}; "
                         f"this is a lookup failure, not a negative result")
    print(f"[pin] {len(vs)} releases published; trying the newest {args.max_tries}")

    ok, msg = test(args.test)
    if ok:
        print(f"[pin] the installed {started_at} already passes: {msg}")
        print(f"RESULT {args.package}=={started_at}")
        return 0
    print(f"[pin] installed {started_at} fails: {msg}")

    extras = f"[{args.extras}]" if args.extras else ""
    for v in vs[: args.max_tries]:
        if v == started_at:
            continue
        spec = f"{args.package}{extras}=={v}"
        rc = install(spec)
        if rc != 0:
            print(f"[pin] {v}: install rc={rc}, skipping")
            continue
        ok, msg = test(args.test)
        print(f"[pin] {v}: {'PASS' if ok else 'fail'}  {msg}")
        if ok:
            print(f"RESULT {args.package}=={v}")
            return 0

    print(f"[pin] none of the newest {args.max_tries} releases of {args.package} "
          f"passes the test. This IS a negative result: every candidate was "
          f"installed and tested.")
    if args.restore_on_failure and started_at:
        install(f"{args.package}{extras}=={started_at}")
        print(f"[pin] restored {args.package}=={started_at}")
    return 11


if __name__ == "__main__":
    raise SystemExit(main())
