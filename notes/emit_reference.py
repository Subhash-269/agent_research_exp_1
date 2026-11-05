#!/usr/bin/env python3
"""Record what a run produced, and draw its curve. One file, staged to the job.

    import emit

    with emit.stage("inference"):
        emit.design("designs/trka_040.pdb", target="TrkA", method="independent")

    with emit.stage("evaluation"):
        emit.score("trka_040", seq_idx=3, ipae_angstrom=5.8,
                   plddt_0to1=0.934, scrmsd_angstrom=1.12)
        emit.cluster("c1", ["trka_040", "trka_041"], method="tmalign",
                     threshold=2.0, compared_structurally=True)

    with emit.stage("admin"):
        emit.job_cost("11451240", state="COMPLETED", elapsed_s=415, gpus=1)
        emit.curve()                       # -> curve.csv (+ curve.png)

Seven verbs. Everything else goes through `emit.write(kind, **fields)`, which
validates against the same schema, so the surface does not grow for records
that are only a set of fields.

                        --- why these seven are typed ---

A verb earns its own name only when it does work the caller would otherwise get
wrong:

  design()    hashes the file. Without it an artifact cannot be verified later.
  score()     COMPUTES success from the declared rule. The caller never passes
              it, so a score and the rule it cites cannot disagree.
  cluster()   carries compared_structurally, which is what separates a real
              clustering result from a shortcut on length or chain count.
  job_cost()  normalises per-job GPU-hours, so the x axis has one shape across
              runs instead of each run inventing a ledger.
  stage()     a context manager, so stage_ended cannot be forgotten and is
              still written when the body raises.
  curve()     assembles and writes the curve from this run's own records, using
              the same function the operator's record.py uses. The agent's
              figure and the operator's chart cannot disagree, because there is
              one implementation.
  write()     the escape hatch. Validated, so it is not a back door.

                          --- where the ids come from ---

SLURM_JOB_ID is set by Slurm, and that is the one that matters: no caller ever
passes a job id, so design-to-job cannot be omitted. Off the cluster a job id is
generated and job_id_source records that it did not come from Slurm, so the two
cases never get confused.

RUN_ID, RUN_DIR and RUN_STAGE are set by the *job script*, not by the connector.
`connector.py submit` exports only HPC_PROJECT_DIR and the cache paths
(`runners/hpc.py:107`), so every job script has to declare its own:

    export RUN_ID="${RUN_ID:-<one id for the whole run>}"
    export RUN_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}/runs/$RUN_ID"

One RUN_ID for the whole run, not per job: it is what groups several jobs into
one run, and what the operator passes to fetch_events afterwards.

                             --- one file per writer ---

Records go to RUN_DIR/events/<job>-<pid>.jsonl, not to one shared file. Several
jobs write at once, and an append from two processes to one file on a cluster
filesystem is not atomic. Readers merge the directory by (utc, seq). Each file
is independently valid, so a walltime kill costs only the lines never flushed.

Standard library only, so this imports on a login node and inside a job.
matplotlib is imported lazily by curve() alone, and its absence costs the PNG,
never the CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ===========================================================================
# SCHEMA  (was runspec.py; merged so the agent stages one file, not three)
# ===========================================================================

SCHEMA = "run-events/1.0"

# The stages the architecture diagram already names, plus one for work that
# belongs to no stage (writing the manifest, declaring the rule, ending a run).
STAGES = ("plan", "env", "data", "search", "inference", "evaluation", "admin")

# How a run finished. Every value but the first two is a run that produced no
# measurement — recorded as a fact, so those runs stop reading as empty rows.
OUTCOMES = (
    "completed",                # the brief's work was attempted end to end
    "partial",                  # categories or targets were cut; reason says which
    "refused",                  # the model declined the task
    "quota_exhausted",          # out of allocation or API quota
    "environment_unavailable",  # no usable GPU, or the stack would not build
    "synthetic_substitute",     # real systems unreachable; a labelled stand-in ran
    "crashed",                  # died with no clean ending
)

SEQ_SOURCES = ("mpnn", "native")
OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}

# Metric names carry their units, because that is exactly where the two existing
# runs diverged: one wrote pLDDT_0to1 and scRMSD_A, the other plddt and
# scrmsd_angstrom, for the same quantities.
METRICS = ("ipae_stored", "ipae_angstrom", "plddt_0to1", "scrmsd_angstrom")

ENVELOPE = ("schema", "seq", "utc", "kind", "run", "job_id", "stage")

# kind -> (required fields, optional fields). Envelope fields are implicit.
KINDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "run_started": (
        ("model", "arm", "harness", "brief_sha256", "workspace_root"),
        ("effort", "gpu_hour_ceiling", "brief_path", "job_id_source"),
    ),
    "rule_declared": (
        ("rule_id", "criteria", "all_on_one_sequence"),
        ("ipae_scale_divisor", "plddt_input_scale", "source"),
    ),
    "stage_started": ((), ("note",)),
    "stage_ended": ((), ("note", "ok")),
    "job_submitted": (
        ("script",),
        ("label", "gres", "partition", "submitted_utc"),
    ),
    # What job_cost() writes. start_utc/end_utc are what put a job in completion
    # order; without them the curve can only fall back on the order rows appear,
    # which is submission order and ranks a long early job as if it finished first.
    # The envelope job_id is the job that was charged; recorded_by_job is the
    # process that wrote the line, when that is a different job.
    "job_ended": (
        ("state",),
        ("elapsed_s", "gpu_hours", "gpus", "log",
         "start_utc", "end_utc", "partition", "node", "recorded_by_job"),
    ),
    "design": (
        ("design_id", "path", "sha256"),
        ("target", "method", "seed", "binder_len", "parent_design_id"),
    ),
    "score": (
        ("design_id", "seq_idx", "seq_source", "rule_id", "success"),
        METRICS + ("note",),
    ),
    "cluster": (
        ("cluster_id", "members", "method"),
        ("threshold", "compared_structurally"),
    ),
    "run_ended": (
        ("outcome",),
        ("reason", "gpu_hours_total"),
    ),
}

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class SpecError(ValueError):
    """A record that does not conform. Raised at write time, on purpose."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise SpecError(f"unknown kind {kind!r}; expected one of {', '.join(sorted(KINDS))}")


def check_stage(stage: str | None) -> None:
    if stage is not None and stage not in STAGES:
        raise SpecError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")


def check(kind: str, fields: dict) -> None:
    """Reject a record before it is written. Unknown field is as bad as missing.

    A typo'd field name is the failure mode this exists for: it would otherwise
    be written happily, read back as absent, and look like work that never
    happened.
    """
    check_kind(kind)
    required, optional = KINDS[kind]
    allowed = set(required) | set(optional)
    present = set(fields)

    missing = [f for f in required if fields.get(f) is None]
    if missing:
        raise SpecError(f"{kind}: missing required field(s): {', '.join(sorted(missing))}")
    unknown = sorted(present - allowed)
    if unknown:
        raise SpecError(
            f"{kind}: unknown field(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}. "
            "Add it to emit.KINDS if it belongs in the schema."
        )

    if kind == "run_ended" and fields["outcome"] not in OUTCOMES:
        raise SpecError(f"outcome {fields['outcome']!r} not in {', '.join(OUTCOMES)}")
    if kind == "score" and fields["seq_source"] not in SEQ_SOURCES:
        raise SpecError(f"seq_source must be one of {', '.join(SEQ_SOURCES)}")
    if kind == "design" and not ID_RE.match(str(fields["design_id"])):
        raise SpecError(f"design_id {fields['design_id']!r} is not a usable identifier")
    if kind == "rule_declared":
        check_rule(fields)


def check_rule(rule: dict) -> None:
    criteria = rule.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise SpecError("rule_declared: criteria must be a non-empty list")
    for c in criteria:
        if set(c) != {"metric", "op", "value"}:
            raise SpecError(f"criterion {c!r} needs exactly metric, op, value")
        if c["metric"] not in METRICS:
            raise SpecError(f"criterion metric {c['metric']!r} not in {', '.join(METRICS)}")
        if c["op"] not in OPS:
            raise SpecError(f"criterion op {c['op']!r} not in {', '.join(OPS)}")


def evaluate(rule: dict, metrics: dict) -> bool:
    """Apply a declared rule to one sequence's metrics.

    The caller never passes `success`. It is derived here, so a score row and
    the rule it cites cannot contradict each other.

    The brief is explicit that all criteria must hold for a single sequence, not
    the best value of each taken from different sequences (PROJECT.md:168).
    This function only ever sees one sequence, which is how that is enforced.
    """
    if not rule.get("all_on_one_sequence", True):
        raise SpecError("only single-sequence rules are supported")
    for c in rule["criteria"]:
        value = metrics.get(c["metric"])
        if value is None:
            raise SpecError(
                f"rule {rule['rule_id']!r} needs {c['metric']}, which this score does not carry"
            )
        if not OPS[c["op"]](value, c["value"]):
            return False
    return True


def rule_fingerprint(rule: dict) -> str:
    """A short, stable description of a rule, for comparing runs.

    Two runs are comparable only when their brief hash AND this match. A run
    scored under different thresholds is a different measurement wearing the
    same column name.
    """
    parts = [f"{c['metric']}{c['op']}{c['value']}" for c in rule["criteria"]]
    return " & ".join(sorted(parts))


#: The rule PROJECT.md:168-180 states, transcribed. Quoted there as: ipAE < 7,
#: complex pLDDT > 0.9, binder scRMSD < 1.5 Å, all three on the same sequence.
#: The ipAE divisor and the pLDDT input scale are recorded because the brief
#: warns that both are places a run can silently diverge.
STRICT_V1 = {
    "rule_id": "strict-v1",
    "all_on_one_sequence": True,
    "criteria": [
        {"metric": "ipae_angstrom", "op": "<", "value": 7},
        {"metric": "plddt_0to1", "op": ">", "value": 0.9},
        {"metric": "scrmsd_angstrom", "op": "<", "value": 1.5},
    ],
    "ipae_scale_divisor": 31,
    "plddt_input_scale": "0-100",
    "source": "PROJECT.md:168-180",
}

#: The repository's own convention, which uses >= for pLDDT. Kept as a separate
#: rule rather than a flag, so a run scored each way produces two labelled sets
#: instead of one ambiguous column. The brief asks for both to be reported.
REPO_V1 = {
    **STRICT_V1,
    "rule_id": "repo-v1",
    "criteria": [
        {"metric": "ipae_angstrom", "op": "<", "value": 7},
        {"metric": "plddt_0to1", "op": ">=", "value": 0.9},
        {"metric": "scrmsd_angstrom", "op": "<", "value": 1.5},
    ],
    "source": "PROJECT.md:177, repository convention",
}

RULES = {r["rule_id"]: r for r in (STRICT_V1, REPO_V1)}


# ===========================================================================
# WRITING
# ===========================================================================

_seq = 0
_stream = None


def _env_job() -> tuple[str, str]:
    """(job_id, source). Slurm's if we are in a job, otherwise a generated one."""
    for var in ("SLURM_JOB_ID", "SLURM_JOBID"):
        jid = os.environ.get(var)
        if jid:
            return jid, "slurm"
    jid = os.environ.get("RUN_JOB_ID")
    if jid:
        return jid, "env"
    # Not under Slurm. Generate something stable for this process and say so,
    # rather than writing null and letting a reader assume the job is unknown.
    return f"local-{os.getpid()}", "generated"


def run_dir() -> Path:
    d = os.environ.get("RUN_DIR")
    if not d:
        raise RuntimeError(
            "RUN_DIR is not set. The job script sets it, not the connector:\n"
            '  export RUN_ID="${RUN_ID:-<id for this run>}"\n'
            '  export RUN_DIR="${HPC_PROJECT_DIR:-$HOME/hpcjobs}/runs/$RUN_ID"\n'
            "connector.py submit exports only HPC_PROJECT_DIR."
        )
    return Path(d)


def _writer():
    """Open this process's own events file, once."""
    global _stream
    if _stream is None:
        job_id, _ = _env_job()
        out = run_dir() / "events"
        out.mkdir(parents=True, exist_ok=True)
        _stream = (out / f"{job_id}-{os.getpid()}.jsonl").open("a", encoding="utf-8")
    return _stream


def write(kind: str, *, for_job: str | None = None, **fields) -> dict:
    """Validate, stamp and append one record. Raises rather than guessing.

    `for_job` sets the envelope job_id to a job other than the one writing. Only
    job_cost() uses it: a job's final sacct figures exist only after it ends, so
    its cost is necessarily written by some other process.
    """
    global _seq
    fields = {k: v for k, v in fields.items() if v is not None}
    check(kind, fields)

    # A score may not disagree with the rule it cites, whichever path wrote it.
    # score() computes `success`; this makes the same thing true of a direct
    # write(), so there is no back door that produces an unverifiable boolean.
    if kind == "score":
        spec = RULES.get(fields["rule_id"])
        if spec is not None:
            metrics = {m: fields[m] for m in METRICS if m in fields}
            expected = evaluate(spec, metrics)
            if fields["success"] != expected:
                raise SpecError(
                    f"success={fields['success']} contradicts rule "
                    f"{fields['rule_id']!r}, which gives {expected} for these metrics"
                )

    stage = os.environ.get("RUN_STAGE")
    check_stage(stage)
    job_id, source = _env_job()
    if for_job is not None and for_job != job_id:
        job_id, source = for_job, "argument"
    _seq += 1

    record = {
        "schema": SCHEMA,
        "seq": _seq,
        "utc": utc_now(),
        "kind": kind,
        "run": os.environ.get("RUN_ID") or run_dir().name,
        "job_id": job_id,
        "stage": stage,
        **fields,
    }
    if source != "slurm":
        record["job_id_source"] = source

    fh = _writer()
    fh.write(json.dumps(record, sort_keys=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())          # a killed job keeps every line it reported
    return record


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ===========================================================================
# THE SEVEN VERBS
# ===========================================================================

class stage:
    """Mark a stage, as a context manager so its end cannot be forgotten.

        with emit.stage("inference", note="reward OFF"):
            ...

    stage_ended is written on the way out even when the body raises, with
    ok=False and the exception on the note. A stage that ends only on the happy
    path is the one case where the log claims less than the run did.

    RUN_STAGE is set for the duration, so every record written inside carries
    the stage without the caller passing it.
    """

    def __init__(self, name: str, note: str | None = None):
        check_stage(name)
        self.name = name
        self.note = note
        self._prev = None

    def __enter__(self):
        self._prev = os.environ.get("RUN_STAGE")
        os.environ["RUN_STAGE"] = self.name
        write("stage_started", note=self.note)
        return self

    def __exit__(self, exc_type, exc, tb):
        note = self.note
        if exc is not None:
            note = f"{note + '; ' if note else ''}{exc_type.__name__}: {exc}"[:500]
        write("stage_ended", ok=exc is None, note=note)
        if self._prev is None:
            os.environ.pop("RUN_STAGE", None)
        else:
            os.environ["RUN_STAGE"] = self._prev
        return False              # never swallow the exception


_declared: set[str] = set()


def _declare_rule(spec: dict | str = "strict-v1") -> dict | None:
    """Declare the scoring rule in force, as data rather than as prose.

    Not one of the seven: score() calls it on first use, so a rule is always on
    record before any number that cites it. Declaring twice is a no-op.
    """
    spec = RULES[spec] if isinstance(spec, str) else spec
    if spec["rule_id"] in _declared:
        return None
    _declared.add(spec["rule_id"])
    return write("rule_declared", **spec)


def design(path: str | Path, design_id: str | None = None, **about) -> dict:
    """Register a structure: hashed, and attributed to the job that made it."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no design at {p}")
    rel = p.resolve()
    try:
        rel = rel.relative_to(run_dir().resolve())
    except ValueError:
        pass                        # outside the run folder; record what we have
    return write("design", design_id=design_id or p.stem,
                 path=rel.as_posix(), sha256=sha256_file(p), **about)


def score(design_id: str, seq_idx: int, seq_source: str = "mpnn",
          rule_id: str = "strict-v1", note: str | None = None, **metrics) -> dict:
    """Record one scored sequence. `success` is computed, never passed in."""
    spec = RULES.get(rule_id)
    if spec is None:
        raise SpecError(
            f"unknown rule_id {rule_id!r}; declare it first or use one of "
            f"{', '.join(RULES)}"
        )
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise SpecError(f"unknown metric(s): {', '.join(unknown)}")
    _declare_rule(spec)
    return write("score", design_id=design_id, seq_idx=seq_idx,
                 seq_source=seq_source, rule_id=rule_id,
                 success=evaluate(spec, metrics), note=note, **metrics)


def cluster(cluster_id: str, members: list[str], method: str,
            threshold=None, compared_structurally: bool | None = None) -> dict:
    """One clustering decision.

    `compared_structurally` is not decoration. The brief asks how many merges
    were real structural comparisons rather than shortcuts on length or chain
    count, because a cluster count where nothing was ever compared is not a
    clustering result.
    """
    return write("cluster", cluster_id=cluster_id, members=members, method=method,
                 threshold=threshold, compared_structurally=compared_structurally)


def job_cost(job_id: str, state: str, elapsed_s: int, gpus: int = 1,
             start_utc: str | None = None, end_utc: str | None = None,
             partition: str | None = None, node: str | None = None,
             log: str | None = None) -> dict:
    """What one job cost. The x axis of every curve.

    `gpu_hours` is computed here — elapsed_s x gpus / 3600 — so the arithmetic
    is in one place and cannot be done differently by two runs. A 60x unit error
    ("21 min = 21 GPU-hours") reached this project's planning numbers once
    already, which is why no caller passes gpu_hours.

    Pass this for EVERY job, including the ones that failed and the ones
    cancelled while pending. A failed allocation is charged in full, and a curve
    that quietly omits it is retrospectively optimistic. A job cancelled before
    it started has elapsed_s=0 and still belongs on record.

    Source the fields from the scheduler, not from your own timing:

        sacct -j <id> -X -P -n -o JobID,State,ElapsedRaw,Start,End,Partition,NodeList

    The record is charged to `job_id`, not to the job that writes it, so one
    admin job can record the cost of every job before it. When the two differ,
    the writer is kept as `recorded_by_job`.

    Written as kind `job_ended`; the verb is named for what it is for.
    """
    job_id = str(job_id).strip()
    if not job_id:
        raise SpecError("job_cost: job_id is empty")
    if elapsed_s < 0 or gpus < 0:
        raise SpecError(f"elapsed_s={elapsed_s} gpus={gpus}: neither may be negative")
    writer, _ = _env_job()
    return write("job_ended", for_job=job_id,
                 state=state, elapsed_s=int(elapsed_s), gpus=int(gpus),
                 gpu_hours=round(int(elapsed_s) * int(gpus) / 3600, 6),
                 start_utc=start_utc, end_utc=end_utc,
                 partition=partition, node=node, log=log,
                 recorded_by_job=writer if writer != job_id else None)


def run_ended(outcome: str, reason: str | None = None,
              gpu_hours_total: float | None = None) -> dict:
    """How the run finished. Written for every run, including the ones that
    produced nothing — that is the point of the enum."""
    return write("run_ended", outcome=outcome, reason=reason,
                 gpu_hours_total=gpu_hours_total)


# ===========================================================================
# THE CURVE  (shared: record.py imports assemble() from here)
# ===========================================================================

CURVE_COLUMNS = ("job_id", "ended_utc", "gpu_hours", "cum_gpu_hours",
                 "cum_gpu_seconds", "raw_successes", "unique_successes",
                 "cum_raw_successes", "cum_unique_successes", "note")


def read_records(root: Path | None = None) -> list[dict]:
    """Every event record under <root>/events, merged in (utc, seq) order.

    A malformed line is skipped and counted, never silently dropped: the count
    comes back on the `note` of the assembled curve so a reader sees it.
    """
    root = Path(root) if root is not None else run_dir()
    out: list[dict] = []
    bad = 0
    for p in sorted((root / "events").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    out.sort(key=lambda r: (r.get("utc") or "", r.get("seq") or 0))
    if bad:
        out.append({"kind": "_malformed", "count": bad})
    return out


def assemble(records: list[dict]) -> dict:
    """Successes as a running function of cumulative GPU-hours.

    In *completion order*, not score order: charging in score order implies you
    knew which runs would succeed before paying for them, which makes any cost
    curve retrospectively optimistic.

    Raw and unique are carried separately the whole way. They are not
    interchangeable — the Oct 7 run reported 216 raw against 5 unique on TrkA,
    and a curve that quietly plotted the first would rank the methods backwards.

    A success is charged to the job that RECORDED IT, not to the job that made
    the structure. It is not a success until something scored it, and by then
    the generation cost is already on the x axis — charging it back to the
    generating job would make the curve rise before the evidence existed.

    A cluster is charged to the earliest-scored of its successful members, so a
    design found cheaply is not re-credited to a later, more expensive job. A
    successful design that no cluster record covers counts as its own cluster
    and `clustering_present` goes false, because an uncollapsed count is a raw
    count wearing the word unique.

    Returns the points plus what was missing, so an empty curve says why.
    """
    jobs: dict[str, dict] = {}
    design_job: dict[str, str] = {}
    success_job: dict[str, str] = {}       # design -> job that first scored it OK
    success_utc: dict[str, str] = {}
    success: set[str] = set()
    clusters: list[dict] = []
    malformed = 0

    for r in records:
        kind = r.get("kind")
        if kind == "_malformed":
            malformed = r.get("count", 0)
        elif kind == "job_ended":
            jobs[r.get("job_id")] = {
                "job_id": r.get("job_id"),
                "gpu_hours": float(r.get("gpu_hours") or 0.0),
                "ended_utc": r.get("end_utc") or r.get("utc"),
                "state": r.get("state"),
            }
        elif kind == "design":
            design_job.setdefault(r.get("design_id"), r.get("job_id"))
        elif kind == "score" and r.get("success"):
            did = r.get("design_id")
            success.add(did)
            success_job.setdefault(did, r.get("job_id"))
            success_utc.setdefault(did, r.get("utc") or "")
        elif kind == "cluster":
            clusters.append(r)

    missing = []
    if not jobs:
        missing.append("no job_ended records: call emit.job_cost() per job")
    if not success:
        missing.append("no successful score records: call emit.score()")

    # Which successes are distinct. Members outside this run's designs are kept
    # anyway: a cluster naming an unknown design is a defect to surface, not to
    # drop quietly.
    covered: set[str] = set()
    uniq_rep: dict[str, str] = {}          # representative design -> charged job
    for c in clusters:
        members = [m for m in (c.get("members") or []) if m in success]
        covered.update(m for m in (c.get("members") or []))
        if not members:
            continue
        rep = min(members, key=lambda m: success_utc.get(m, ""))
        uniq_rep[rep] = success_job.get(rep)
    singletons = sorted(success - covered)
    for d in singletons:
        uniq_rep[d] = success_job.get(d)

    raw_by: dict[str, int] = {}
    for d in success:
        j = success_job.get(d)
        raw_by[j] = raw_by.get(j, 0) + 1
    uniq_by: dict[str, int] = {}
    for d, j in uniq_rep.items():
        uniq_by[j] = uniq_by.get(j, 0) + 1

    orphan = sorted(d for d in success if d not in design_job)
    if orphan:
        missing.append(f"{len(orphan)} score(s) name a design with no design record")

    # Jobs with no end time sort last rather than being dropped, so their spend
    # still lands on the curve even though its position is unknown.
    ordered = sorted(jobs.values(),
                     key=lambda j: (j["ended_utc"] is None, j["ended_utc"] or ""))
    cum_gpu = cum_raw = cum_uniq = 0.0
    points = []
    for job in ordered:
        jid = job["job_id"]
        cum_gpu += job["gpu_hours"]
        cum_raw += raw_by.get(jid, 0)
        cum_uniq += uniq_by.get(jid, 0)
        points.append({
            "job_id": jid,
            "ended_utc": job["ended_utc"],
            "gpu_hours": round(job["gpu_hours"], 4),
            "cum_gpu_hours": round(cum_gpu, 4),
            # the step axis MLflow charts against; integer by necessity
            "cum_gpu_seconds": int(round(cum_gpu * 3600)),
            "raw_successes": raw_by.get(jid, 0),
            "unique_successes": uniq_by.get(jid, 0),
            "cum_raw_successes": int(cum_raw),
            "cum_unique_successes": int(cum_uniq),
            "note": job["state"] or "",
        })

    # Successes attributed to a job that never reported its cost would vanish
    # from the curve entirely. Say so rather than losing them.
    unplaced = sorted(j for j in raw_by if j not in jobs)
    if unplaced:
        missing.append(f"{len(unplaced)} job(s) have successes but no job_cost record")

    return {
        "points": points,
        "jobs": len(jobs),
        "designs": len(design_job),
        "raw_successes": len(success),
        "unique_successes": len(uniq_rep),
        "clustering_present": bool(clusters),
        "clusters": len(clusters),
        "orphan_scores": orphan or None,
        "unplaced_jobs": unplaced or None,
        "malformed_lines": malformed or None,
        "missing": missing or None,
    }


def curve(out_dir: Path | None = None, title: str | None = None) -> dict:
    """Assemble this run's curve and write it. CSV always, PNG when it can.

    The CSV is the contract and is written with the standard library alone. The
    PNG is convenience: matplotlib is imported here and nowhere else, and if it
    is absent the CSV is still correct and the result says so. A job must not
    fail for want of a plotting library.
    """
    root = Path(out_dir) if out_dir is not None else run_dir()
    result = assemble(read_records(root))
    root.mkdir(parents=True, exist_ok=True)

    csv_path = root / "curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(CURVE_COLUMNS) + "\n")
        for p in result["points"]:
            fh.write(",".join(_csv_cell(p.get(c)) for c in CURVE_COLUMNS) + "\n")
    result["csv"] = str(csv_path)

    result["png"] = None
    if result["points"]:
        try:
            result["png"] = _plot(result, root / "curve.png",
                                  title or os.environ.get("RUN_ID") or root.name)
        except ImportError as exc:
            result["png_skipped"] = f"no plotting library: {exc}"
    else:
        result["png_skipped"] = "no points to plot"
    return result


def _csv_cell(v) -> str:
    if v is None:
        return ""
    s = str(v)
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


def _plot(result: dict, path: Path, title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")                  # no display on a compute node
    import matplotlib.pyplot as plt

    pts = result["points"]
    x = [0.0] + [p["cum_gpu_hours"] for p in pts]
    raw = [0] + [p["cum_raw_successes"] for p in pts]
    uniq = [0] + [p["cum_unique_successes"] for p in pts]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    # Steps, not lines: a success is known at the moment its job reports, not
    # smoothly in between. drawstyle post says exactly that and stops a reader
    # interpolating a rate the run never measured.
    ax.plot(x, raw, drawstyle="steps-post", marker="o", ms=4,
            label=f"raw ({result['raw_successes']})")
    if result["clustering_present"]:
        ax.plot(x, uniq, drawstyle="steps-post", marker="s", ms=4,
                label=f"unique ({result['unique_successes']})")
    else:
        ax.plot([], [], " ", label="unique: clustering not exercised")
    ax.set_xlabel("cumulative GPU-hours (completion order, failures charged in full)")
    ax.set_ylabel("successes")
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.02, y=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("design", help="register a structure")
    s.add_argument("--path", required=True)
    s.add_argument("--design-id")
    s.add_argument("--target")
    s.add_argument("--method")
    s.add_argument("--seed", type=int)
    s.add_argument("--binder-len", type=int)

    s = sub.add_parser("score", help="record one scored sequence")
    s.add_argument("--design-id", required=True)
    s.add_argument("--seq-idx", type=int, required=True)
    s.add_argument("--seq-source", default="mpnn", choices=SEQ_SOURCES)
    s.add_argument("--rule-id", default="strict-v1")
    for m in METRICS:
        s.add_argument(f"--{m.replace('_', '-')}", type=float, dest=m)

    s = sub.add_parser("cluster", help="record one clustering decision")
    s.add_argument("--cluster-id", required=True)
    s.add_argument("--members", required=True, help="comma-separated design ids")
    s.add_argument("--method", required=True)
    s.add_argument("--threshold", type=float)
    s.add_argument("--compared-structurally", action="store_true", default=None)

    s = sub.add_parser("job-cost", help="what one job cost; the curve's x axis")
    s.add_argument("--job-id", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--elapsed-s", type=int, required=True)
    s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--start-utc")
    s.add_argument("--end-utc")
    s.add_argument("--partition")
    s.add_argument("--node")

    s = sub.add_parser("stage", help="mark a stage boundary")
    s.add_argument("edge", choices=("start", "end"))
    s.add_argument("--name")
    s.add_argument("--note")
    s.add_argument("--failed", action="store_true")

    s = sub.add_parser("rule", help="declare the scoring rule")
    s.add_argument("--rule-id", default="strict-v1")

    s = sub.add_parser("curve", help="write curve.csv (+ curve.png)")
    s.add_argument("--dir", type=Path, default=None)
    s.add_argument("--title")

    s = sub.add_parser("end", help="record how the run finished")
    s.add_argument("--outcome", required=True, choices=OUTCOMES)
    s.add_argument("--reason")
    s.add_argument("--gpu-hours-total", type=float)

    a = ap.parse_args()
    try:
        if a.cmd == "design":
            rec = design(a.path, a.design_id, target=a.target, method=a.method,
                         seed=a.seed, binder_len=a.binder_len)
        elif a.cmd == "score":
            metrics = {m: getattr(a, m) for m in METRICS if getattr(a, m) is not None}
            rec = score(a.design_id, a.seq_idx, a.seq_source, a.rule_id, **metrics)
        elif a.cmd == "cluster":
            rec = cluster(a.cluster_id, [m.strip() for m in a.members.split(",") if m.strip()],
                          a.method, a.threshold, a.compared_structurally)
        elif a.cmd == "job-cost":
            rec = job_cost(a.job_id, a.state, a.elapsed_s, a.gpus,
                           a.start_utc, a.end_utc, a.partition, a.node)
        elif a.cmd == "stage":
            if a.edge == "start":
                if a.name:
                    check_stage(a.name)
                    os.environ["RUN_STAGE"] = a.name
                rec = write("stage_started", note=a.note)
            else:
                rec = write("stage_ended", ok=not a.failed, note=a.note)
        elif a.cmd == "rule":
            rec = _declare_rule(a.rule_id)
        elif a.cmd == "curve":
            rec = curve(a.dir, a.title)
            rec.pop("points", None)        # the CSV is the payload, not stdout
        else:
            rec = run_ended(a.outcome, a.reason, a.gpu_hours_total)
    except (SpecError, RuntimeError, FileNotFoundError) as exc:
        # loud, in the job log, while someone is watching
        print(f"emit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rec, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
