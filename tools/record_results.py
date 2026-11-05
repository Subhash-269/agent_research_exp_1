#!/usr/bin/env python3
"""Turn one Complexa evaluation directory into emit records. Runs in the job.

AGENTS.md: "Record every design, every score, every clustering decision and
every job's cost through `emit`, as the work happens ... Do not write your own
ledger, results CSV or curve plotter in its place." So this file writes no CSV
of its own. It reads what the pipeline produced and calls emit.design() and
emit.score(); emit computes `success` and assembles the curve.

                        --- where the numbers come from ---

Not from `binder_results_*.csv`. Those carry `{seq}_complex_i_pAE`,
`{seq}_complex_pLDDT` and `{seq}_binder_scRMSD`, and each is the BEST value
over that sample's sequences, chosen independently per metric
(src/proteinfoundation/evaluation/binder_eval.py:255-290). PROJECT.md requires
all three thresholds to hold for a SINGLE sequence, so taking three
independently-chosen bests and testing them together would count designs that
no one sequence ever satisfied.

The per-sequence values live in `sequence_type_stats.json`, written once per
sample (binder_eval.py:244). `complex_stats[i]` and `rmsd_stats[i]` are the
same sequence i, which is what makes a per-sequence join valid.

                               --- ipAE scale ---

The repository stores interface pAE divided by 31. Verified in this
repository, not taken from the brief:
  src/proteinfoundation/result_analysis/binder_analysis_utils.py:79
      "scale": 31.0,  # ipae * 31 <= 7
  src/proteinfoundation/utils/refolded_structure_utils.py:131
      if (ipae * 31 <= 7) and (plddt >= 0.9) and (rmsd < 1.5):
Both the stored value and the Angstrom value are recorded, so a later reader
can see which scale a threshold was applied on.

                             --- survival rules ---

S12  No truncation cap anywhere here, and a missing metric is a hard error
     rather than a skipped record. A silently dropped sample is a result that
     disappears without leaving a hole.
S14  Harvesting is scoped to the directory this run's job wrote, passed in as
     --eval-dir. Nothing is selected by modification time, so another run's
     output in a sibling directory cannot be swept in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import emit

# The repository's own divisor, read from the source cited above, not assumed.
IPAE_SCALE_DIVISOR = 31.0      # source: binder_analysis_utils.py:79, refolded_structure_utils.py:131

# PROJECT.md step 1 asks for the model's own native sequence and the ProteinMPNN
# redesigns reported as separate categories. The repo calls them "self" and
# "mpnn"; emit's vocabulary is "native" and "mpnn".
SEQ_SOURCE_OF = {"self": "native", "mpnn": "mpnn", "mpnn_fixed": "mpnn"}

# Key aliases, tried in order. If none is present the script raises and prints
# every key it did see, so a schema change is loud rather than silent.
IPAE_KEYS = ("i_pAE", "ipAE", "i_pae")
PLDDT_KEYS = ("pLDDT", "plddt")
RMSD_KEYS = ("binder_scRMSD", "binder_scRMSD_bb3", "binder_scRMSD_ca")


class SchemaError(RuntimeError):
    pass


def pick(d: dict, keys: tuple[str, ...], what: str, where: str):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise SchemaError(
        f"{where}: no {what} key among {keys}. Keys present: {sorted(d)}"
    )


def find_stats_files(eval_dir: Path) -> list[Path]:
    """Every sequence_type_stats.json under this run's own evaluation dir.

    Sorted for determinism. No cap, no mtime filter (S12, S14).
    """
    return sorted(eval_dir.rglob("sequence_type_stats.json"))


def design_id_for(stats_path: Path) -> str:
    # binder_eval writes each sample under tmp_<sample_name>/
    name = stats_path.parent.name
    return name[4:] if name.startswith("tmp_") else name


def generated_pdb_for(stats_path: Path, design_id: str) -> Path | None:
    """The generated backbone this sample was evaluated from, if it is here."""
    for cand in (stats_path.parent / f"{design_id}.pdb",
                 stats_path.parent / f"tmp_{design_id}.pdb"):
        if cand.is_file():
            return cand
    hits = sorted(stats_path.parent.glob("*.pdb"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="this job's own evaluation output directory (S14)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seq-types", default="self,mpnn")
    ap.add_argument("--dump-schema", action="store_true",
                    help="print the first stats file's structure and stop")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        raise SystemExit(f"--eval-dir does not exist: {eval_dir}")

    files = find_stats_files(eval_dir)
    print(f"[record] {len(files)} sequence_type_stats.json under {eval_dir}")
    if not files:
        # Not an error in itself: a job may legitimately produce nothing. It is
        # reported as a fact rather than papered over.
        print("[record] nothing to record from this directory")
        return 0

    if args.dump_schema:
        blob = json.loads(files[0].read_text())
        print(f"[schema] file: {files[0]}")
        print(f"[schema] seq types: {sorted(blob)}")
        for st, body in blob.items():
            print(f"[schema]   {st}: sections {sorted(body)}")
            for section, rows in body.items():
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    print(f"[schema]     {section}[0] keys: {sorted(rows[0])}")
                    print(f"[schema]     {section} length: {len(rows)}")
        return 0

    wanted = [s.strip() for s in args.seq_types.split(",") if s.strip()]
    n_designs = n_scores = 0

    with emit.stage("evaluation", note=f"{args.target}/{args.method}"):
        for stats_path in files:
            design_id = design_id_for(stats_path)
            blob = json.loads(stats_path.read_text())

            pdb = generated_pdb_for(stats_path, design_id)
            if pdb is None:
                raise SchemaError(
                    f"{stats_path.parent}: no .pdb beside the stats file, so this "
                    f"design cannot be hashed. Refusing to record it as scored "
                    f"without the structure it scores (S12)."
                )
            emit.design(pdb, design_id=design_id, target=args.target,
                        method=args.method, seed=args.seed)
            n_designs += 1

            for seq_type in wanted:
                body = blob.get(seq_type)
                if not body:
                    print(f"[record] {design_id}: no '{seq_type}' block")
                    continue
                complex_stats = body.get("complex_stats") or []
                rmsd_stats = body.get("rmsd_stats") or []
                if len(complex_stats) != len(rmsd_stats):
                    raise SchemaError(
                        f"{stats_path}: {seq_type} has {len(complex_stats)} complex_stats "
                        f"but {len(rmsd_stats)} rmsd_stats. They are joined by index, so "
                        f"a length mismatch would silently pair the wrong sequence."
                    )
                for i, (cs, rs) in enumerate(zip(complex_stats, rmsd_stats)):
                    where = f"{stats_path}:{seq_type}[{i}]"
                    stored = float(pick(cs, IPAE_KEYS, "ipAE", where))
                    plddt = float(pick(cs, PLDDT_KEYS, "pLDDT", where))
                    rmsd = float(pick(rs, RMSD_KEYS, "binder scRMSD", where))

                    # pLDDT must already be on 0-1 here; the repo's own rule
                    # tests `plddt >= 0.9`. If it is not, say so rather than
                    # quietly rescaling and reporting a number nobody can trace.
                    if not (0.0 <= plddt <= 1.0):
                        raise SchemaError(
                            f"{where}: pLDDT={plddt} is outside 0-1. emit expects "
                            f"plddt_0to1; rescaling it here would hide a units change."
                        )

                    emit.score(
                        design_id, seq_idx=i,
                        seq_source=SEQ_SOURCE_OF[seq_type],
                        ipae_stored=stored,
                        ipae_angstrom=stored * IPAE_SCALE_DIVISOR,
                        plddt_0to1=plddt,
                        scrmsd_angstrom=rmsd,
                        note=f"{args.method}/{args.target}/seq_type={seq_type}",
                    )
                    n_scores += 1

    print(f"[record] designs={n_designs} scores={n_scores}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchemaError as exc:
        # AGENTS.md: a bad record must raise inside the job. Do not catch and
        # continue - a missing record reads later as absence, which is worse.
        sys.stderr.write(f"SCHEMA ERROR: {exc}\n")
        raise SystemExit(9)
