"""Evaluation entry point. Replaces scripts/legacy/evaluate_docker_LEGACY.py.

Guarantees the legacy script did not provide:

  * The contamination report runs first and prints above the accuracy. If prompt
    leakage is found, the run is marked CONTAMINATED and (with --strict) refuses
    to emit a headline number at all.
  * Results are broken out by novelty: unseen-command vs unseen-phrasing. A single
    blended number hides which of the two a model is actually good at.
  * Every generation is recorded to a JSONL so a published result can be re-scored
    later under improved metrics without re-running inference.
  * Three metrics side by side, never one.

Examples:
    # Score a GGUF with a prebuilt llama.cpp binary
    python -m eval.run_eval --model models/docker_gemma3_4b_q4km.gguf \\
        --template gemma3 --llama-bin vendor/llama.cpp/llama-cli.exe

    # Re-score a previous run under new metrics, no inference
    python -m eval.run_eval --replay results/run_gemma3_4b.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from eval.backends import (
    Backend,
    LlamaCppBinaryBackend,
    LlamaCppPythonBackend,
    LlamaServerBackend,
    ReplayBackend,
    TransformersBackend,
    build_prompt,
)
from eval.contamination import Example, audit, load_jsonl
from eval.metrics import categorize, score_all
from eval.normalize import normalized_string

DEFAULT_TRAIN = Path("data/docker_training.jsonl")
DEFAULT_TEST = Path("data/docker_test_handwritten.jsonl")


def extract_command(raw_output: str) -> str:
    """Pull the command out of the model's structured response.

    The model is trained to emit:
        COMMAND: docker ps -a
        CONFIDENCE: 0.94
        EXPLANATION: ...

    Falls back to the first non-empty line so a model that ignores the format still
    gets scored rather than counted as a blank.
    """
    text = (raw_output or "").strip()
    if not text:
        return ""

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("COMMAND:"):
            return line.split(":", 1)[1].strip()

    # Strip common markdown fencing before falling back.
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line or line.lower() in {"bash", "sh", "shell", "console"}:
            continue
        return line
    return ""


def build_backend(args: argparse.Namespace) -> Backend:
    if args.replay:
        return ReplayBackend(args.replay)
    if args.hf_model:
        return TransformersBackend(
            args.hf_model,
            template=args.template if args.template != "auto" else None,
            device=args.device,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
    if not args.model:
        raise SystemExit("one of --model, --hf-model, or --replay is required")

    template = args.template if args.template != "auto" else "gemma3"

    if args.backend == "llama-cli":
        return LlamaCppBinaryBackend(
            args.model,
            template=template,
            binary=args.llama_bin,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            temperature=args.temperature,
        )

    if args.backend == "llama-cpp-python":
        return LlamaCppPythonBackend(
            args.model,
            template=template,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            temperature=args.temperature,
        )

    # Default: llama-server. Loads the model once and returns JSON, so it does not
    # depend on scraping a TUI that changes between llama.cpp releases.
    return LlamaServerBackend(
        args.model,
        template=template,
        binary=args.llama_bin,
        port=args.port,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        temperature=args.temperature,
        server_url=args.server_url,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run a contamination-checked evaluation.")

    src = ap.add_argument_group("model source")
    src.add_argument("--model", type=Path, help="path to a GGUF file")
    src.add_argument("--hf-model", help="HuggingFace model id (GPU path)")
    src.add_argument("--replay", type=Path, help="re-score a recorded run, no inference")
    src.add_argument("--llama-bin", type=Path,
                     help="path to a prebuilt llama-server (or llama-cli with "
                          "--backend llama-cli)")
    src.add_argument("--backend", default="llama-server",
                     choices=["llama-server", "llama-cli", "llama-cpp-python"],
                     help="GGUF execution backend (default: llama-server)")
    src.add_argument("--server-url", help="use an already-running llama-server")
    src.add_argument("--port", type=int, default=8080)

    data = ap.add_argument_group("data")
    data.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    data.add_argument("--test", type=Path, default=DEFAULT_TEST)
    data.add_argument("--limit", type=int, help="evaluate only the first N examples")

    gen = ap.add_argument_group("generation")
    gen.add_argument("--template", default="auto",
                     help="gemma3 | gemma4 | qwen | llama3 | raw | auto")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--max-tokens", type=int, default=128)
    gen.add_argument("--n-ctx", type=int, default=512)
    gen.add_argument("--threads", type=int, default=4)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--load-in-4bit", action="store_true")

    out = ap.add_argument_group("output")
    out.add_argument("--label", default="run", help="name for this run's output files")
    out.add_argument("--results-dir", type=Path, default=Path("results"))
    out.add_argument("--strict", action="store_true",
                     help="refuse to report accuracy if the test set is contaminated")
    out.add_argument("--show-misses", type=int, default=15)

    args = ap.parse_args(argv)

    # ---------------------------------------------------------------- data ---
    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    if args.limit:
        test = test[: args.limit]

    # -------------------------------------------------- contamination gate ---
    report = audit(train, test)
    print(report.format(verbose=False))
    print()

    if not report.is_clean:
        print("!" * 72)
        print("The test set leaks prompts from training. Any accuracy below is invalid.")
        print("!" * 72)
        if args.strict:
            return 2

    # ------------------------------------------------------------- backend ---
    backend = build_backend(args)
    print(f"backend: {json.dumps(backend.describe())}")
    print(f"scoring {len(test)} examples...\n")

    train_cmds = {normalized_string(e.command) for e in train}

    records = []
    predictions: List[str] = []
    golds: List[str] = []
    categories: List[str] = []
    novelties: List[str] = []

    t_start = time.time()
    for i, ex in enumerate(test, 1):
        if isinstance(backend, TransformersBackend):
            prompt = backend.build_prompt_native(ex.instruction)
        else:
            template = args.template if args.template != "auto" else "gemma3"
            prompt = build_prompt(ex.instruction, template)

        gen_result = backend.generate(prompt, max_tokens=args.max_tokens)
        predicted = extract_command(gen_result.text)

        novelty = (
            "unseen_phrasing"
            if normalized_string(ex.command) in train_cmds
            else "unseen_command"
        )

        predictions.append(predicted)
        golds.append(ex.command)
        categories.append(categorize(ex.command))
        novelties.append(novelty)

        records.append({
            "prompt": prompt,
            "instruction": ex.instruction,
            "gold": ex.command,
            "predicted": predicted,
            "raw_output": gen_result.text,
            "latency_s": round(gen_result.latency_s, 3),
            "mean_logprob": gen_result.mean_logprob,
            "category": categories[-1],
            "novelty": novelty,
        })

        if i % 10 == 0 or i == len(test):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(test)}  ({elapsed:.0f}s elapsed, "
                  f"{elapsed / i:.1f}s/example)", flush=True)

    total_time = time.time() - t_start

    # ------------------------------------------------------------- scoring ---
    overall = score_all(predictions, golds, categories)
    by_novelty = score_all(predictions, golds, novelties)

    print()
    print("#" * 72)
    print(f"# RESULTS — {args.label}")
    print("#" * 72)
    print(overall.format_table())
    print()
    print("BY NOVELTY (the number that matters most is unseen_command)")
    print("-" * 66)
    for nov in sorted(by_novelty.per_category):
        s = by_novelty.per_category[nov]
        n = s["n"]
        print(f"  {nov:<18} n={n:<4} exact {s['exact'] / n:>6.1%}   "
              f"norm {s['normalized'] / n:>6.1%}   func {s['functional'] / n:>6.1%}")
    print("-" * 66)
    print(f"  mean latency  {total_time / max(len(test), 1):.2f}s per example")
    print()

    # Pair judgements back to their records so the miss report can show the actual
    # prompt the model was given, not just the gold command.
    missed = [
        (rec, j)
        for rec, j in zip(records, overall.judgements)
        if not j.functional
    ]
    if missed and args.show_misses:
        print(f"MISSES ({len(missed)} total, showing {min(len(missed), args.show_misses)})")
        print("-" * 72)
        for rec, j in missed[: args.show_misses]:
            print(f"  [{rec['category']}/{rec['novelty']}]")
            print(f"  prompt   {rec['instruction']}")
            print(f"  expected {j.gold}")
            print(f"  got      {j.predicted or '<empty>'}")
            if j.note:
                print(f"  note     {j.note}")
            print()

    # -------------------------------------------------------------- output ---
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    gen_path = args.results_dir / f"{args.label}_generations.jsonl"
    with open(gen_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "label": args.label,
        "timestamp_utc": stamp,
        "backend": backend.describe(),
        "train_file": str(args.train),
        "test_file": str(args.test),
        "n_test": len(test),
        "contamination": report.to_dict(),
        "overall": {
            "exact": overall.rate("exact"),
            "normalized": overall.rate("normalized"),
            "functional": overall.rate("functional"),
        },
        "by_category": {
            c: {
                "n": s["n"],
                "exact": s["exact"] / s["n"],
                "normalized": s["normalized"] / s["n"],
                "functional": s["functional"] / s["n"],
            }
            for c, s in overall.per_category.items()
        },
        "by_novelty": {
            c: {
                "n": s["n"],
                "exact": s["exact"] / s["n"],
                "normalized": s["normalized"] / s["n"],
                "functional": s["functional"] / s["n"],
            }
            for c, s in by_novelty.per_category.items()
        },
        "mean_latency_s": total_time / max(len(test), 1),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    sum_path = args.results_dir / f"{args.label}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"generations -> {gen_path}")
    print(f"summary     -> {sum_path}")

    close = getattr(backend, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
