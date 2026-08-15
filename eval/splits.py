"""Train/test splitting that does not leak.

The failure this replaces: `dataset.train_test_split(test_size=0.1, seed=42)` on a
file where the generator emits 2-3 paraphrases per command. A random row-level split
puts "run nginx in background" in train and "run nginx detached" in test, both
targeting `docker run -d nginx`. The model does not have to generalize to score well.

The fix is to partition by *target command*: every row whose gold command lands in
the test partition goes to test, and no command appears on both sides. This makes
the held-out set measure what it claims to measure — translation of an unseen
instruction to a command the model was never trained to emit.

Note the honest limitation, which is disclosed in the report rather than hidden:
a command-level split makes the task strictly harder than production use. Real users
mostly ask for commands the model was trained on, phrased differently. So we report
BOTH partitions:

    unseen-command   held-out commands       — the hard, honest generalization test
    unseen-phrasing  seen commands, new NL   — the realistic deployment test

Both numbers matter. Publishing only the flattering one is how this project got into
trouble in the first place.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from eval.contamination import Example, load_jsonl
from eval.metrics import categorize
from eval.normalize import normalized_string


@dataclass
class Split:
    train: List[Example]
    test: List[Example]
    held_out_commands: Set[str]

    def summary(self) -> str:
        train_cmds = {normalized_string(e.command) for e in self.train}
        test_cmds = {normalized_string(e.command) for e in self.test}
        overlap = train_cmds & test_cmds
        L = [
            "=" * 66,
            "COMMAND-LEVEL SPLIT",
            "=" * 66,
            f"  train rows              {len(self.train):>6}",
            f"  test  rows              {len(self.test):>6}",
            f"  train unique commands   {len(train_cmds):>6}",
            f"  test  unique commands   {len(test_cmds):>6}",
            f"  COMMAND OVERLAP         {len(overlap):>6}   <- must be 0",
            "=" * 66,
        ]
        if overlap:
            L.append("SPLIT IS BROKEN - commands present on both sides:")
            for c in sorted(overlap)[:10]:
                L.append(f"    {c}")
        return "\n".join(L)


def split_by_command(
    examples: Sequence[Example],
    test_fraction: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Split:
    """Partition so that no target command appears in both train and test.

    Stratifies by category so the test set is not accidentally all `docker volume`.
    """
    by_command: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        by_command[normalized_string(ex.command)].append(ex)

    commands = sorted(by_command)
    rng = random.Random(seed)

    held_out: Set[str] = set()

    if stratify:
        by_category: Dict[str, List[str]] = defaultdict(list)
        for cmd in commands:
            by_category[categorize(by_command[cmd][0].command)].append(cmd)
        for cat, cmds in sorted(by_category.items()):
            cmds = sorted(cmds)
            rng.shuffle(cmds)
            n = max(1, round(len(cmds) * test_fraction))
            held_out.update(cmds[:n])
    else:
        shuffled = list(commands)
        rng.shuffle(shuffled)
        held_out.update(shuffled[: max(1, round(len(shuffled) * test_fraction))])

    train = [ex for ex in examples if normalized_string(ex.command) not in held_out]
    test = [ex for ex in examples if normalized_string(ex.command) in held_out]

    return Split(train=train, test=test, held_out_commands=held_out)


def partition_test_by_novelty(
    test: Sequence[Example], train: Sequence[Example]
) -> Tuple[List[Example], List[Example]]:
    """Split a test set into (unseen_command, unseen_phrasing_only) partitions."""
    train_cmds = {normalized_string(e.command) for e in train}
    unseen_command = [e for e in test if normalized_string(e.command) not in train_cmds]
    unseen_phrasing = [e for e in test if normalized_string(e.command) in train_cmds]
    return unseen_command, unseen_phrasing


def write_jsonl(examples: Sequence[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps({"instruction": ex.instruction, "command": ex.command}) + "\n"
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create a leak-free train/test split.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--train-out", type=Path)
    ap.add_argument("--test-out", type=Path)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    examples = load_jsonl(args.input)
    split = split_by_command(examples, args.test_fraction, args.seed)
    print(split.summary())

    if args.train_out:
        write_jsonl(split.train, args.train_out)
        print(f"train written to {args.train_out}")
    if args.test_out:
        write_jsonl(split.test, args.test_out)
        print(f"test  written to {args.test_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
