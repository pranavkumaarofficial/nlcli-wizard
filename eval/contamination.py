"""Contamination auditing for any (train, test) pair.

This module exists because the project published accuracy figures for nine months
that were measured on training data. The defence against a repeat is not care; it
is a tool that runs on every evaluation and refuses to stay quiet.

Four leak channels, in descending order of severity:

  1. PROMPT_VERBATIM  the exact instruction string appears in train.
                      The model may have memorized the answer. Fatal.
  2. PROMPT_NEAR_DUP  the instruction differs from a train instruction only by
                      token order / punctuation / casing. Nearly as bad.
  3. TARGET_OVERLAP   the gold command appears as a training target under some
                      other phrasing. Not automatically fatal — generalizing
                      across phrasings is the actual task — but it must be
                      measured and disclosed, because a test set made entirely of
                      seen targets measures paraphrase robustness, not translation.
  4. TEMPLATE_SHARED  the instruction matches a training instruction's generator
                      template (same string with entities substituted). This is
                      the leak that survives a naive random split.

Usage:
    python -m eval.contamination --train data/docker_training.jsonl \\
                                 --test  data/docker_test_handwritten.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from eval.normalize import normalized_string

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

INSTRUCTION_PREFIX = re.compile(r"^Translate to \S+ command:\s*", re.IGNORECASE)


@dataclass
class Example:
    instruction: str
    command: str
    source: str = ""

    @property
    def prompt(self) -> str:
        """The instruction with the task prefix stripped."""
        return INSTRUCTION_PREFIX.sub("", self.instruction).strip()


def load_jsonl(path: Path) -> List[Example]:
    """Load either the training format (instruction/output) or the test format."""
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            instruction = row.get("instruction") or row.get("prompt") or ""

            command = row.get("command")
            if command is None:
                output = row.get("output", "")
                if "COMMAND:" in output:
                    command = output.split("COMMAND:", 1)[1].split("\n")[0].strip()
                else:
                    command = output.strip()

            if not instruction or not command:
                raise ValueError(f"{path}:{lineno} missing instruction or command")

            examples.append(Example(instruction, command, source=f"{path.name}:{lineno}"))
    return examples


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Entity classes the dataset generator substitutes into templates. Replacing them
# with placeholders reveals whether two prompts share a generator template.
_ENTITY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d+\b"), "<NUM>"),
    (
        re.compile(
            r"\b(nginx|redis|postgres|mysql|mongo|node|python|ubuntu|alpine|busybox)\b",
            re.IGNORECASE,
        ),
        "<IMAGE>",
    ),
    (
        re.compile(
            r"\b(web|api|db|cache|worker|frontend|backend|app|service|proxy|server)\b",
            re.IGNORECASE,
        ),
        "<NAME>",
    ),
]


def prompt_fingerprint(prompt: str) -> str:
    """Casing/punctuation/word-order-insensitive fingerprint."""
    s = _PUNCT.sub(" ", prompt.lower())
    tokens = sorted(_WS.sub(" ", s).strip().split())
    return " ".join(tokens)


def template_fingerprint(prompt: str) -> str:
    """Fingerprint with generator entities masked out.

    `run nginx on port 8080` and `run redis on port 6379` collapse to the same
    fingerprint, exposing that they came from one template.
    """
    s = prompt.lower()
    for pattern, placeholder in _ENTITY_PATTERNS:
        s = pattern.sub(placeholder, s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ContaminationReport:
    n_train: int
    n_test: int
    prompt_verbatim: List[Example] = field(default_factory=list)
    prompt_near_dup: List[Tuple[Example, str]] = field(default_factory=list)
    target_overlap: List[Example] = field(default_factory=list)
    template_shared: List[Tuple[Example, str]] = field(default_factory=list)

    train_unique_prompts: int = 0
    train_unique_targets: int = 0
    test_unique_targets: int = 0

    @property
    def is_clean(self) -> bool:
        """Clean means no prompt-level leakage. Target overlap is disclosed, not fatal."""
        return not self.prompt_verbatim and not self.prompt_near_dup

    @property
    def is_strict_clean(self) -> bool:
        """Strict means no leakage of any kind, including shared targets."""
        return self.is_clean and not self.target_overlap and not self.template_shared

    def rate(self, channel: str) -> float:
        return len(getattr(self, channel)) / self.n_test if self.n_test else 0.0

    def format(self, verbose: bool = False) -> str:
        L: List[str] = []
        L.append("=" * 72)
        L.append("CONTAMINATION REPORT")
        L.append("=" * 72)
        L.append(f"train examples          {self.n_train:>6}   "
                 f"unique prompts {self.train_unique_prompts:>5}   "
                 f"unique targets {self.train_unique_targets:>5}")
        L.append(f"test  examples          {self.n_test:>6}   "
                 f"{'':>19}   unique targets {self.test_unique_targets:>5}")
        L.append("-" * 72)
        L.append(f"{'CHANNEL':<22}{'COUNT':>8}{'RATE':>10}   SEVERITY")
        L.append("-" * 72)
        rows = [
            ("PROMPT_VERBATIM", self.prompt_verbatim, "FATAL"),
            ("PROMPT_NEAR_DUP", self.prompt_near_dup, "FATAL"),
            ("TEMPLATE_SHARED", self.template_shared, "high"),
            ("TARGET_OVERLAP", self.target_overlap, "disclose"),
        ]
        for name, items, severity in rows:
            rate = len(items) / self.n_test if self.n_test else 0.0
            L.append(f"{name:<22}{len(items):>8}{rate:>9.1%}   {severity}")
        L.append("=" * 72)

        if self.is_strict_clean:
            L.append("VERDICT: STRICTLY CLEAN - no leakage on any channel.")
        elif self.is_clean:
            L.append("VERDICT: CLEAN - no prompt leakage.")
            if self.target_overlap:
                L.append(
                    f"         {len(self.target_overlap)} test items share a target command "
                    f"with train ({self.rate('target_overlap'):.0%})."
                )
                L.append(
                    "         This is expected and acceptable: generalizing to new phrasings "
                    "of known"
                )
                L.append(
                    "         commands is the task. It is disclosed so the number is read "
                    "correctly."
                )
        else:
            L.append("VERDICT: CONTAMINATED - do not publish accuracy from this pair.")

        if verbose:
            if self.prompt_verbatim:
                L.append("")
                L.append("-- verbatim prompt leaks (first 10) --")
                for ex in self.prompt_verbatim[:10]:
                    L.append(f"   {ex.source}  {ex.prompt!r}")
            if self.prompt_near_dup:
                L.append("")
                L.append("-- near-duplicate prompts (first 10) --")
                for ex, match in self.prompt_near_dup[:10]:
                    L.append(f"   {ex.source}  {ex.prompt!r}")
                    L.append(f"       ~ train: {match!r}")
            if self.template_shared:
                L.append("")
                L.append("-- shared generator templates (first 10) --")
                for ex, tmpl in self.template_shared[:10]:
                    L.append(f"   {ex.source}  {ex.prompt!r}")
                    L.append(f"       template: {tmpl!r}")
        return "\n".join(L)

    def to_dict(self) -> Dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_unique_prompts": self.train_unique_prompts,
            "train_unique_targets": self.train_unique_targets,
            "test_unique_targets": self.test_unique_targets,
            "prompt_verbatim": len(self.prompt_verbatim),
            "prompt_near_dup": len(self.prompt_near_dup),
            "template_shared": len(self.template_shared),
            "target_overlap": len(self.target_overlap),
            "is_clean": self.is_clean,
            "is_strict_clean": self.is_strict_clean,
        }


def audit(train: Sequence[Example], test: Sequence[Example]) -> ContaminationReport:
    """Audit a train/test pair across all four leak channels."""
    train_prompts_exact = {ex.prompt for ex in train}
    train_prompts_fp: Dict[str, str] = {}
    for ex in train:
        train_prompts_fp.setdefault(prompt_fingerprint(ex.prompt), ex.prompt)

    train_templates: Dict[str, str] = {}
    for ex in train:
        train_templates.setdefault(template_fingerprint(ex.prompt), ex.prompt)

    train_targets = {normalized_string(ex.command) for ex in train}

    report = ContaminationReport(
        n_train=len(train),
        n_test=len(test),
        train_unique_prompts=len(train_prompts_exact),
        train_unique_targets=len(train_targets),
        test_unique_targets=len({normalized_string(ex.command) for ex in test}),
    )

    for ex in test:
        prompt = ex.prompt

        if prompt in train_prompts_exact:
            report.prompt_verbatim.append(ex)
            continue

        fp = prompt_fingerprint(prompt)
        if fp in train_prompts_fp:
            report.prompt_near_dup.append((ex, train_prompts_fp[fp]))
            continue

        tfp = template_fingerprint(prompt)
        if tfp in train_templates:
            report.template_shared.append((ex, train_templates[tfp]))

        if normalized_string(ex.command) in train_targets:
            report.target_overlap.append(ex)

    return report


def self_audit(examples: Sequence[Example]) -> Dict[str, object]:
    """Describe a single dataset's internal redundancy.

    Run this on a training file before trusting any split of it. A file with 1500
    rows and 230 unique prompts cannot be randomly split into non-overlapping parts.
    """
    prompts = [ex.prompt for ex in examples]
    targets = [normalized_string(ex.command) for ex in examples]
    prompt_counts = Counter(prompts)
    target_counts = Counter(targets)
    return {
        "rows": len(examples),
        "unique_prompts": len(prompt_counts),
        "unique_targets": len(target_counts),
        "prompt_duplication_factor": round(len(examples) / max(len(prompt_counts), 1), 2),
        "target_duplication_factor": round(len(examples) / max(len(target_counts), 1), 2),
        "most_repeated_prompt": prompt_counts.most_common(1)[0] if prompt_counts else None,
        "most_repeated_target": target_counts.most_common(1)[0] if target_counts else None,
        "random_split_is_safe": len(prompt_counts) == len(examples),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit a train/test pair for contamination.")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", type=Path, help="omit to run a self-audit of --train only")
    ap.add_argument("--verbose", action="store_true", help="list offending examples")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    ap.add_argument(
        "--fail-on-contamination",
        action="store_true",
        help="exit non-zero if prompt leakage is found (for CI)",
    )
    args = ap.parse_args(argv)

    train = load_jsonl(args.train)

    if args.test is None:
        stats = self_audit(train)
        print("=" * 72)
        print(f"SELF-AUDIT: {args.train}")
        print("=" * 72)
        for k, v in stats.items():
            print(f"  {k:<28} {v}")
        print("=" * 72)
        if not stats["random_split_is_safe"]:
            print("WARNING: prompts repeat within this file. A random row-level split")
            print("         will place identical prompts on both sides. Split by target")
            print("         command instead (see eval/splits.py).")
        return 0

    test = load_jsonl(args.test)
    report = audit(train, test)
    print(report.format(verbose=args.verbose))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")

    if args.fail_on_contamination and not report.is_clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
