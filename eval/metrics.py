"""Scoring. Three metrics, always reported together.

Reporting a single accuracy number invites picking the flattering one. These are
computed on every run and printed side by side:

  exact       — byte-identical to gold. The old harness's only metric. Kept so
                old and new numbers can be compared honestly.
  normalized  — same command path, same flag set (order-insensitive), same
                positional sequence. Credits flag reordering.
  functional  — normalized, plus a small set of documented equivalences where two
                different commands do the same thing.

`functional` is the headline number, but it is the loosest, so `exact` travels
with it everywhere. Where they diverge is itself informative: a big exact/normalized
gap means the model learned the right flags but not the training set's arbitrary
flag ordering, which is a scoring artifact rather than a model failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from eval.normalize import ParsedCommand, parse_command

# ---------------------------------------------------------------------------
# Documented functional equivalences
# ---------------------------------------------------------------------------

# Flags that do not change the effect of the command for our purposes. `-d` is NOT
# in here — detached vs foreground is a real behavioural difference.
IGNORABLE_FLAGS: Set[str] = set()

# Interactive shell invocation: `-i -t` together vs `-t` alone is a real difference,
# but our gold data always uses both, and a model emitting only one is wrong. Left
# strict deliberately.

# Pairs of command paths that are interchangeable in effect.
EQUIVALENT_PATHS: List[Set[Tuple[str, ...]]] = [
    {("docker", "ps"), ("docker", "container", "ls")},
    {("docker", "images"), ("docker", "image", "ls")},
    {("docker", "stop"), ("docker", "container", "stop")},
    {("docker", "rm"), ("docker", "container", "rm")},
    {("docker-compose",), ("docker", "compose")},
]

# Default values that may be stated or omitted with identical effect. Keyed by
# (command path prefix, flag) -> the value that is the documented default.
OMISSIBLE_DEFAULTS: Dict[Tuple[str, str], str] = {
    (("docker", "run"), "--network"): "bridge",
}


@dataclass
class Judgement:
    """The outcome of scoring one prediction against one gold command."""

    exact: bool
    normalized: bool
    functional: bool
    predicted: str
    gold: str
    note: str = ""

    @property
    def any_match(self) -> bool:
        return self.functional


def _paths_equivalent(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
    if a == b:
        return True
    for group in EQUIVALENT_PATHS:
        if a in group and b in group:
            return True
    return False


def _strip_default_flags(p: ParsedCommand) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Drop flags whose value equals the documented default for that command."""
    kept = []
    for flag, value in p.flags:
        default = OMISSIBLE_DEFAULTS.get((p.path, flag))
        if default is not None and value == default:
            continue
        if flag in IGNORABLE_FLAGS:
            continue
        kept.append((flag, value))
    return tuple(kept)


def score_one(predicted: str, gold: str) -> Judgement:
    """Score a single prediction. Never raises."""
    pred_s = (predicted or "").strip()
    gold_s = (gold or "").strip()

    exact = pred_s == gold_s

    p = parse_command(pred_s)
    g = parse_command(gold_s)

    normalized = (
        p.parse_ok
        and g.parse_ok
        and p.path == g.path
        and p.flags == g.flags
        and p.positionals == g.positionals
    )

    note = ""
    if not p.parse_ok and pred_s:
        note = "prediction failed to tokenize"

    if normalized:
        functional = True
    else:
        functional = (
            p.parse_ok
            and g.parse_ok
            and _paths_equivalent(p.path, g.path)
            and _strip_default_flags(p) == _strip_default_flags(g)
            and p.positionals == g.positionals
        )
        if functional and not normalized:
            note = note or "matched via documented equivalence"

    # An exact match must imply the looser ones. Guard against a normalization bug
    # silently downgrading a correct answer.
    if exact:
        normalized = True
        functional = True

    return Judgement(
        exact=exact,
        normalized=normalized,
        functional=functional,
        predicted=pred_s,
        gold=gold_s,
        note=note,
    )


@dataclass
class ScoreReport:
    """Aggregate scores over a run, overall and per category."""

    n: int
    exact: int
    normalized: int
    functional: int
    per_category: Dict[str, Dict[str, int]]
    judgements: List[Judgement]

    def rate(self, metric: str) -> float:
        return getattr(self, metric) / self.n if self.n else 0.0

    def misses(self) -> List[Judgement]:
        return [j for j in self.judgements if not j.functional]

    def format_table(self) -> str:
        lines = []
        lines.append("=" * 66)
        lines.append(f"{'METRIC':<16}{'CORRECT':>10}{'TOTAL':>8}{'RATE':>10}")
        lines.append("-" * 66)
        for metric in ("exact", "normalized", "functional"):
            v = getattr(self, metric)
            lines.append(f"{metric:<16}{v:>10}{self.n:>8}{self.rate(metric):>9.1%}")
        lines.append("=" * 66)
        lines.append("")
        lines.append(f"{'CATEGORY':<14}{'N':>5}{'EXACT':>10}{'NORM':>10}{'FUNC':>10}")
        lines.append("-" * 66)
        for cat in sorted(self.per_category):
            s = self.per_category[cat]
            n = s["n"]
            lines.append(
                f"{cat:<14}{n:>5}"
                f"{s['exact'] / n:>9.1%}"
                f"{s['normalized'] / n:>10.1%}"
                f"{s['functional'] / n:>10.1%}"
            )
        lines.append("=" * 66)
        return "\n".join(lines)


def score_all(
    predictions: Sequence[str],
    golds: Sequence[str],
    categories: Optional[Sequence[str]] = None,
) -> ScoreReport:
    """Score a full run."""
    if len(predictions) != len(golds):
        raise ValueError(f"length mismatch: {len(predictions)} predictions, {len(golds)} golds")

    cats = list(categories) if categories is not None else ["all"] * len(golds)
    judgements = [score_one(p, g) for p, g in zip(predictions, golds)]

    per_category: Dict[str, Dict[str, int]] = {}
    totals = {"exact": 0, "normalized": 0, "functional": 0}

    for j, cat in zip(judgements, cats):
        bucket = per_category.setdefault(
            cat, {"n": 0, "exact": 0, "normalized": 0, "functional": 0}
        )
        bucket["n"] += 1
        for metric in totals:
            if getattr(j, metric):
                bucket[metric] += 1
                totals[metric] += 1

    return ScoreReport(
        n=len(judgements),
        exact=totals["exact"],
        normalized=totals["normalized"],
        functional=totals["functional"],
        per_category=per_category,
        judgements=judgements,
    )


def categorize(command: str) -> str:
    """Bucket a gold command for per-category reporting."""
    p = parse_command(command)
    if not p.path:
        return "unknown"
    if p.path[0] == "docker-compose":
        return "compose"
    if len(p.path) < 2:
        return "system"
    sub = p.path[1]
    if sub in {"run", "create"}:
        return "run"
    if sub == "build":
        return "build"
    if sub == "exec":
        return "exec"
    if sub == "network":
        return "network"
    if sub == "volume":
        return "volume"
    if sub in {"ps", "images", "logs", "inspect", "stats", "top"}:
        return "ps_images"
    return "system"
