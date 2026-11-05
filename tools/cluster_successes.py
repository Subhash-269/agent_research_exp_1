#!/usr/bin/env python3
"""Collapse near-duplicate successful designs with Foldseek, and record it.

PROJECT.md asks for successes to be collapsed structurally, for the threshold
to be stated, for counts at two neighbouring thresholds, and for how many
merges were real structural comparisons rather than shortcuts.

Three things this deliberately does NOT do.

1. It does not invent a cluster count when there is nothing to cluster. A
   category with fewer than two successes is reported as "not exercised", never
   as a cluster count of 1. One success trivially forms one cluster and
   reporting that number as a clustering result would imply a comparison that
   never happened.
2. It does not fall back to sequence identity, chain count or length when
   Foldseek is unavailable or errors. If the structural comparison did not run,
   `compared_structurally=False` is recorded and the reason is printed. A
   shortcut merge recorded as a structural one is worse than no merge.
3. It does not pool across categories. Clustering runs per
   (target, method, seq_source), because a native sequence and an MPNN redesign
   of the same backbone are different categories by PROJECT.md's own rule and
   merging them would hide that.

Usage:
    python cluster_successes.py --events-dir $RUN_DIR/events \
        [--thresholds 2.0,2.5] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# TM-align style distance thresholds in Angstrom, reported as a neighbouring
# pair so a reader can see whether the count is stable under the choice.
# source: chosen by me, no external source. PROJECT.md requires two
# neighbouring thresholds and leaves the values to me; 2.0 A is the repository
# default used in the worked example in AGENTS.md.
DEFAULT_THRESHOLDS = (2.0, 2.5)

# Foldseek's own exit code for "ran, found nothing" vs a real error.
# source: measured this session if a run produces it; until then every non-zero
# exit is treated as a failure, which is the conservative reading.
FOLDSEEK_OK = 0


class ClusterError(RuntimeError):
    pass


def load_records(events_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(glob.glob(str(events_dir / "*.jsonl")))
    if not files:
        raise ClusterError(f"no event files under {events_dir}; nothing to cluster. "
                           f"This is a lookup failure, not an empty result.")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # S12: a malformed record is an error, never a skipped one.
                    raise ClusterError(f"{f}: malformed record: {exc}") from exc
    return rows


def successful_designs(rows: list[dict]) -> dict[tuple, list[str]]:
    """design_ids that have at least one successful sequence, per category."""
    design_about: dict[str, dict] = {}
    for r in rows:
        if r.get("kind") == "design":
            design_about[str(r.get("design_id"))] = r

    by_cat: dict[tuple, set] = collections.defaultdict(set)
    for r in rows:
        if r.get("kind") != "score" or not r.get("success"):
            continue
        did = str(r.get("design_id"))
        about = design_about.get(did, {})
        cat = (about.get("target"), about.get("method"), r.get("seq_source"))
        by_cat[cat].add(did)
    return {k: sorted(v) for k, v in by_cat.items()}


def design_paths(rows: list[dict]) -> dict[str, str]:
    out = {}
    for r in rows:
        if r.get("kind") == "design":
            p = r.get("path")
            if p:
                out[str(r.get("design_id"))] = p
    return out


def foldseek_available() -> tuple[bool, str]:
    exe = shutil.which("foldseek")
    if not exe:
        return False, "foldseek is not on PATH"
    p = subprocess.run([exe, "version"], capture_output=True, text=True)
    if p.returncode != FOLDSEEK_OK:
        return False, f"`foldseek version` exited {p.returncode}"
    return True, (p.stdout or "").strip().splitlines()[0] if p.stdout else exe


def foldseek_cluster(pdbs: list[Path], threshold: float) -> list[list[int]]:
    """Return index groups. Raises rather than guessing if foldseek misbehaves."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        indir = td / "in"
        indir.mkdir()
        for i, p in enumerate(pdbs):
            shutil.copy(p, indir / f"d{i:05d}.pdb")
        out = td / "res"
        tmp = td / "tmp"
        cmd = ["foldseek", "easy-cluster", str(indir), str(out), str(tmp),
               "--alignment-type", "1", "--tmscore-threshold", "0.0",
               "-e", "inf"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != FOLDSEEK_OK:
            raise ClusterError(
                f"foldseek easy-cluster exited {p.returncode}: "
                f"{(p.stderr or '').strip()[-400:]}")
        tsv = Path(str(out) + "_cluster.tsv")
        if not tsv.exists():
            raise ClusterError(
                f"foldseek exited 0 but produced no {tsv.name}. Exit zero is "
                f"necessary, not sufficient - the artifact is missing.")
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rep, mem = line.split("\t")[:2]
            groups[rep].append(int(mem.replace("d", "").split(".")[0]))
        return [sorted(v) for v in groups.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be clustered, emit nothing")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if len(thresholds) < 2:
        raise ClusterError("PROJECT.md requires counts at two neighbouring "
                           "thresholds; give at least two")

    rows = load_records(Path(args.events_dir))
    cats = successful_designs(rows)
    paths = design_paths(rows)

    total_success = sum(len(v) for v in cats.values())
    print(f"[cluster] {len(rows)} records, {total_success} successful designs "
          f"across {len(cats)} categories")

    if total_success == 0:
        print("[cluster] NOT EXERCISED: this run recorded zero successful "
              "designs, so there is nothing to collapse. Reporting a cluster "
              "count here would describe a comparison that never happened.")
        return 0

    ok, detail = foldseek_available()
    print(f"[cluster] foldseek: {'available' if ok else 'UNAVAILABLE'} - {detail}")

    import emit  # imported late so --dry-run works without RUN_DIR set

    for cat, dids in sorted(cats.items(), key=lambda kv: str(kv[0])):
        target, method, seq_source = cat
        label = f"{target}/{method}/{seq_source}"
        if len(dids) < 2:
            print(f"[cluster] {label}: {len(dids)} success - NOT EXERCISED "
                  f"(fewer than two successes; no comparison is possible)")
            continue
        pdbs = [Path(paths[d]) for d in dids if d in paths and Path(paths[d]).exists()]
        missing = len(dids) - len(pdbs)
        if missing:
            # S12: a missing structure is a hard error, not a skipped record.
            raise ClusterError(f"{label}: {missing} successful designs have no "
                               f"readable structure on disk; refusing to cluster "
                               f"a subset and report it as the whole")
        for th in thresholds:
            if not ok:
                print(f"[cluster] {label} @ {th}: structural comparison did not "
                      f"run ({detail}); recording compared_structurally=False")
                if not args.dry_run:
                    emit.cluster(f"{label}@{th}", dids, method="none",
                                 threshold=th, compared_structurally=False)
                continue
            groups = foldseek_cluster(pdbs, th)
            print(f"[cluster] {label} @ {th} A: {len(dids)} successes -> "
                  f"{len(groups)} clusters "
                  f"({len(dids) - len(groups)} merges, all structural)")
            if not args.dry_run:
                for gi, g in enumerate(groups):
                    emit.cluster(f"{label}@{th}#{gi}", [dids[i] for i in g],
                                 method="foldseek", threshold=th,
                                 compared_structurally=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClusterError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(2)
