# nlcli-wizard

[![tests](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml/badge.svg)](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing)
[![Reddit](https://img.shields.io/badge/Reddit-r%2FLocalLLaMA-orange.svg)](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)

**A pipeline for giving any Python CLI tool a `-w` flag that takes plain English.**
Local, offline, one small model per library, trained once and shipped with the package.

## The idea

Every CLI has flags you look up every time. I do it for `docker run`. I do it for
`tar`. The reference is right there in `--help` and it still takes three tries.

So: what if the library shipped with a translator?

```bash
docker -w "show all running containers"
# docker ps

docker -w "run nginx on port 8080 in the background"
# docker run -d -p 8080:80 nginx
```

No cloud call. No API key. The model is a few hundred megabytes, it came down with
`pip install`, and it runs on the CPU you already have.

https://github.com/user-attachments/assets/2d7ca418-d6b2-4449-a81e-417df9666d44

## How the pipeline works

The product here is not the Docker model. It is the four steps that produce one, so a
library maintainer can run them against their own tool.

**1. Generate.** A tool's flag grammar becomes a training set: flag specs with several
intent phrasings each, examples built by sampling flag subsets rather than whole
commands, sentence order randomised. The point is covering flag *combinations*, since
that is where a small model actually struggles.

**2. Audit.** Before anything trains, the dataset is checked against the held-out set
for leakage across four channels. If a test prompt turns up in training, the run stops.
Not a warning, a stop.

**3. Fine-tune.** QLoRA on a 1B base model, one adapter per tool, on a free Colab T4 or
any single GPU. The adapter is small. The base model is shared.

**4. Quantize and ship.** Merge, convert to GGUF, quantize to 4-bit, put it next to the
package. llama.cpp loads it on demand.

Add a tool, you get its adapter. The harness, the scorer and the inference path stay
the same.

## Why local

Because the machines where you most want help with a command are often the ones that
cannot reach a hosted model. Air-gapped build hosts. Customer-site VMs. Networks where
approving a new outbound endpoint costs more than the feature is worth.

There is a duller reason too: something you call fifty times a day should not have a
per-call price or a round trip.

That constraint drives the whole design. No GPU and no network means a small model,
which means the hard part is the data and the evaluation, not the model.

## What runs today

Docker is the worked example and it is the only one. `nlcli_wizard/dataset_v2.py`
generates 5,000 Docker examples over 3,188 distinct commands, and the Colab notebook
trains and scores an adapter against 116 hand-written held-out prompts.

The `-w` flag needs a shell function to intercept it. `scripts/docker-wizard.sh` has
one you can paste into your shell profile. Per-library packaging, where the adapter
arrives with `pip install`, is the direction and is not built yet.

Weights are not downloadable yet either. `MODEL_REGISTRY` in `nlcli_wizard/model.py`
expects `docker_gemma3_4b_q4km.gguf` from `pranavkumaarofficial/nlcli-gemma3-docker`
on HuggingFace, and that repo is still private. For now, bring your own GGUF with
`--model-path`, or train one. If the download fails, `translate` prints the repo, the
filename and every path it looked in.

## Quickstart

```bash
git clone https://github.com/pranavkumaarofficial/nlcli-wizard.git
cd nlcli-wizard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install llama-cpp-python from the prebuilt CPU wheel index first. Without it, pip
# grabs the sdist and compiles llama.cpp, which needs a C++ toolchain and fails on
# Windows on a long path.
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install -e .
```

Fourteen packages, no compiler. `torch` and `transformers` are not runtime
dependencies, because inference goes through GGUF and llama.cpp only.

With a GGUF in `models/`:

```bash
python -m nlcli_wizard.cli translate --cli-tool docker \
    "run nginx on port 8080 in background"
```

```
Input: run nginx on port 8080 in background
Command: docker run -d -p 8080:80 nginx
```

On a four-thread laptop CPU that is 1.09 s to load the model and 1.85 s to answer.

### Adding a tool

```bash
# 1. generate, excluding your held-out set at generation time
python -m nlcli_wizard.dataset_v2 --out data/mytool_train.jsonl \
                                  --exclude data/mytool_test.jsonl

# 2. check it before spending GPU minutes
python -m eval.contamination --train data/mytool_train.jsonl \
                             --test  data/mytool_test.jsonl

# 3. train in the notebook, then score the result
python -m eval.run_eval --model models/mytool.gguf --template gemma3 \
                        --llama-bin /path/to/llama-server --label mytool
```

Step 1 is still Docker-specific. Deriving the flag grammar from a tool's own `--help`
rather than hand-written specs is the open piece of the pipeline.

## Results

Docker, 116 hand-written held-out prompts, no prompt overlap with training, scored by
exact match, flag-order-normalized match and functional equivalence together.

| | accuracy |
|---|---|
| base 1B, 8-shot prompting | 19.0% |
| fine-tuned adapter | 62.9% |

Composition is where it still falls over. Two flags in one command lands at 50%, three
or more at 17%. That looks like a data coverage problem rather than a model size
problem, and it is the next thing to fix.

Reproduce any of it with no GPU and no model download:

```bash
python -m eval.run_eval --replay results/v2_finetune_generations.jsonl
```

Every run in `results/` ships its per-generation records, not just a summary, so the
numbers are checkable. Method and caveats in
[`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md).

## Design notes

**The evaluation is the part I would defend.** `eval/` has no third-party
dependencies, it is the same code in the notebook and on a laptop, and the
contamination audit runs before inference and prints above the accuracy. Splits are by
target command, not by row, because a generator that emits several paraphrases per
command will otherwise put paraphrases of the same command on both sides. Flags that
change behaviour, `-d` and `-i` and `-t` and `-a`, are deliberately not normalized
away.

**Q4_K_M on a narrow model is not quite what the name says.** K-quants work on
256-element superblocks, and this model's embedding width is 1152, which is not a
multiple of 256. Every tensor with a 1152-long row falls back to a non-K quant, so the
file ends up about 60% Q5_0 by parameter count. Worth knowing before picking a
quantization for a small model. Check any GGUF with standard library only:

```bash
python scripts/gguf_header.py models/your-model.gguf
```

## Limitations

- Docker only. One adapter exists.
- `run` and `exec`, the two verbs people most want help with, sit around 50%.
- Three or more flags in one command is largely unsolved.
- The `CONFIDENCE` field the model emits is meaningless. The dataset generator filled
  it with a random number, so that is what the model learned to predict. Ignore it.
- Single seed, no variance estimates. At 116 examples the 95% interval is roughly plus
  or minus 9 points.
- The held-out set is one person's phrasing. Prompts from real users would be stronger.
- The CLI lowercases your input, which breaks build-arg values and env var names.

## Layout

```
nlcli_wizard/   CLI, GGUF loading, prompt formatting, dataset generators
eval/           contamination audit, command-level splits, normalization, scoring,
                inference backends. No third-party dependencies.
tests/          120 tests
data/           training sets and the hand-written held-out prompts
results/        every run, summaries and per-generation records
scripts/        gguf_header.py, shell wrappers
training/       Colab notebook, generated by build_notebook.py
```

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome at
[github.com/pranavkumaarofficial/nlcli-wizard/issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues).

## License

[MIT](LICENSE)
