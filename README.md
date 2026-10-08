# nlcli-wizard

[![tests](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml/badge.svg)](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Translates plain English into Docker commands using a fine-tuned small language
model that runs on CPU, offline. For people who work on machines that cannot reach
a hosted API.

> **This project previously published 94% Docker accuracy. That number was wrong.**
> It was measured on training data. The corrected figure is 46.6%. The full account
> is in [`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md) — what broke, how it
> was found, and what replaced it. Both numbers are kept side by side below rather
> than the old one being quietly deleted.

At 46.6% this is not a tool you should install to get work done. What is worth your
time is `eval/`: a contamination-checked harness that refuses to print an accuracy
without printing a leakage report above it. That harness is what turned 94% into
46.6%, and it is the part of this repo that would survive contact with someone
else's dataset.

## Why it exists

The obvious way to turn English into a Docker command is to ask a hosted model. It
works, and for most people it is the right answer. It stops being the right answer
on a host with no egress: air-gapped build machines, customer-site VMs, restricted
networks where the security review of a new outbound endpoint costs more than the
feature is worth. The commands you most want help composing are the ones you run in
exactly those places.

So the constraint here is offline on CPU, which rules out anything that needs a GPU
or a network call, which in turn rules out model sizes above a couple of billion
parameters. The question the project exists to answer is whether a model small
enough to satisfy that constraint can learn one CLI's flag grammar from its own
documentation. The measured answer so far is no, and the interesting part is why:
not model capacity, but the composition coverage of the training set. See
[Limitations](#limitations).

Rejected along the way:

- **Retrieval over the man pages.** Gets you the right subcommand and no help with
  flag composition, which the failure analysis below shows is the entire problem.
- **A hand-written grammar and intent parser.** Accurate and brittle. It is the
  approach that does not generalize to the next CLI tool, which is the point.
- **Scoring by exact string match**, which the previous harness did. Flag order is
  not semantic. Replaced with three metrics reported together; see
  [Design notes](#design-notes).

## Results

Fine-tuned on 594 templated Docker examples with QLoRA, quantized to GGUF, run on
CPU through llama.cpp. Evaluated on 116 hand-written held-out prompts with zero
prompt overlap with training
([`data/docker_test_handwritten.jsonl`](data/docker_test_handwritten.jsonl)).

| | Legacy harness | Corrected harness |
|--|--|--|
| Overall | 94.0% | **46.6%** |
| Unseen command | — | 38.0% (n=50) |
| Unseen phrasing, known command | — | 53.0% (n=66) |
| Eval set | last 100 lines of the training file | 116 hand-written held-out prompts |
| Prompt leakage | ~90 of 100 rows in train | 0 |
| Metric | exact string match | exact + flag-order-normalized + functional |

Exact, normalized, and functional scoring all returned **46.6%** — the model never
lost a point to flag ordering. The gap is contamination, not scoring.

### Per-category

| Category | n | Corrected | Legacy claim |
|----------|---|-----------|--------------|
| volume | 7 | 100.0% | 100% |
| system | 16 | 68.8% | 100% |
| ps/images | 20 | 60.0% | 87.5% |
| build | 9 | 55.6% | 90.0% |
| network | 9 | 55.6% | 100% |
| compose | 15 | 46.7% | 100% |
| run | 29 | 20.7% | 96.2% |
| exec | 11 | 9.1% | 84.6% |

The ranking inverted. Categories with few distinct commands survive; the
flag-composition-heavy ones collapse.

Of 62 misses, 44 use the right subcommand with wrong flags, 18 pick the wrong
subcommand, and none are malformed. Six of the ten `exec` misses are a single dropped
`-it`: the model learned which phrasings precede `-it`, not that interactive intent
requires it. That is memorization of surface form, and it is what the 94% measured.

**Status of the 1B vs 4B comparison:** withdrawn pending re-measurement. The claimed
"capacity ceiling" at 73–76% was attributed to the 1B model's parameter count. The
corrected results suggest the ceiling was imposed by the dataset — 594 examples over
298 mostly single-flag commands cannot teach flag composition — and that the larger
model hit the same wall unnoticed behind a contaminated metric.

**Which model produced 46.6% is unresolved.** The GGUF is named
`docker_gemma3_4b_q4km.gguf`, but its own header reports `general.architecture
gemma3`, `general.size_label 1000M`, 26 blocks, embedding width 1152, and
999,885,952 parameters summed over its 340 tensors. Those are Gemma 3 1B's numbers,
not 4B's. Until that is settled, treat the parameter count as unknown rather than
trusting the filename. Check it yourself, standard library only:

```bash
python scripts/gguf_header.py models/docker_gemma3_4b_q4km.gguf
```

### Reproducing the number

The numbers above come from
[`results/gemma3_4b_docker_summary.json`](results/gemma3_4b_docker_summary.json).
With the GGUF in `models/` and a llama.cpp release build:

```bash
python -m eval.run_eval \
    --model models/docker_gemma3_4b_q4km.gguf \
    --template gemma3 \
    --llama-bin /path/to/llama-server \
    --label myrun
```

Roughly ten minutes on four CPU threads. `--llama-bin` must be `llama-server`, not
`llama-cli`; the default backend speaks HTTP. Temperature is 0, so the run is
deterministic: re-running it on 2026-09-18 reproduced all three headline metrics to
17 significant figures and all 116 generations byte for byte.

## Quickstart

CI runs the test suite on Python 3.10 and 3.12
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

```bash
git clone https://github.com/pranavkumaarofficial/nlcli-wizard.git
cd nlcli-wizard

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install llama-cpp-python from the prebuilt CPU wheel index FIRST. Without this,
# pip resolves it to an sdist and compiles llama.cpp, which needs a C++ toolchain
# and fails on Windows with a long-path error inside vendor/llama.cpp.
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install -e .
```

Then put `docker_gemma3_4b_q4km.gguf` (806 MB) in `models/` and:

```bash
python -m nlcli_wizard.cli translate --cli-tool docker \
    "run nginx on port 8080 in background"
```

```
Loading model from models\docker_gemma3_4b_q4km.gguf...
Model loaded successfully!
Input: run nginx on port 8080 in background
Command: docker run -d -p 8080:80 nginx
Confidence: 95%
Runs nginx container in detached mode, mapping port 8080:80, container ID: nginx
```

That is the real output, 8.6 s wall on four CPU threads. Two things in it are not
good: the confidence is a sampled random number (see Limitations) and the
explanation trails off into nonsense after the first clause. Neither is scored by
the evaluation, which reads only the `COMMAND:` line.

**The weights are not currently downloadable.** `MODEL_REGISTRY` in
[`nlcli_wizard/model.py`](nlcli_wizard/model.py) points at
`pranavkumaarofficial/nlcli-gemma3-docker` on HuggingFace, and that repo is not
public as of 2026-09-18. `translate` prints the repo id, the filename, and the paths
it searched when the download fails. Until the repo is published, the only working
paths are a file you already have (`--model-path`) or training your own.

The harness itself needs no weights:

```bash
pip install -e ".[dev]"
python -m pytest -q tests                                                    # 92 tests
python -m eval.contamination --train data/docker_training.jsonl              # self-audit
python -m eval.contamination --train data/docker_training.jsonl \
                             --test  data/docker_test_handwritten.jsonl
```

### Training your own

[Colab notebook](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL)
(free T4).

```bash
python -m nlcli_wizard.dataset_docker   # writes data/docker_training.jsonl
# train in the notebook, download the GGUF into models/, then:
python -m eval.run_eval --model models/<your>.gguf --template gemma3 \
    --llama-bin /path/to/llama-server --label yourrun
```

The tracked notebook and the shipped weights are built from different base models.
Read [`training/build_notebook.py`](training/build_notebook.py) before assuming the
notebook reproduces the numbers above; it does not.

## How it works

```
"scale web service to 3 instances"
  |
  v  <start_of_turn>user\nTranslate to docker command: ...<end_of_turn>\n<start_of_turn>model\n
  v
  fine-tuned Gemma 3, Q4_K_M GGUF, llama.cpp, CPU, 4 threads
  |
  v  COMMAND: docker-compose up --scale web=3
  v  CONFIDENCE: 0.92      <- ignore this, see Limitations
  v  EXPLANATION: Scales the web service to 3 replicas
  |
  v
preview -> confirm -> execute
```

```
nlcli_wizard/
  cli.py              click entry point
  model.py            GGUF loading, MODEL_REGISTRY, local discovery
  agent.py            prompt formatting, output parsing, validation
  dataset_docker.py   v1 Docker dataset generator (594 rows)
  dataset_v2.py       v2 composition-first generator (5,000 rows)
eval/
  contamination.py    four-channel leakage audit
  splits.py           command-level splits
  normalize.py        structural command parsing
  metrics.py          exact / normalized / functional scoring
  backends.py         llama-server, llama-cli, llama-cpp-python, transformers, replay
  run_eval.py         entry point
tests/                92 tests
data/
  docker_training.jsonl           v1, 594 rows / 298 unique commands
  docker_train_v2.jsonl           v2, 5,000 rows / 3,188 unique commands
  docker_test_handwritten.jsonl   116 hand-written held-out prompts
results/                          published run summaries
training/
  build_notebook.py               generates the v2 notebook
```

## Design notes

**Contamination is a gate, not a report.** The leakage audit runs before inference
and prints above the accuracy, and `--strict` makes a leaking pair exit non-zero
with no number at all. This is the decision that mattered. The original 94% survived
nine months because the eval logic lived in a private copy inside the training
notebook, where nothing ever compared the two files. Four channels are checked:
verbatim prompt overlap, near-duplicate prompts, shared templates, and target-command
overlap. Only the first two are fatal. Target overlap is disclosed rather than
failed, because generalizing to new phrasings of a known command is the task, and
57% of this test set does exactly that. Reporting a single blended number over both
would hide which of the two the model can do, which is why results are also split by
novelty.

**Normalization gives the model every benefit of the doubt, with one deliberate
exception.** Scoring parses both commands into a subcommand path, a flag set, and
positionals, so `docker run -d -p 8080:80 nginx` and `docker run -p 8080:80 -d nginx`
match, as do `docker ps` and `docker container ls`. What is *not* normalized away is
`-d`, `-i`, `-t` and `-a`. Those change what the command does, and a scorer that
forgives them would have scored the `exec` category at 90% while the model was
producing non-interactive shells. Building the normalizer surfaced two real parser
bugs: `-it` did not expand into `-i -t`, and `-t` was globally value-consuming when
it only takes a value under `build`, so `docker run -t nginx` parsed as a tag with no
image. Both would have mis-scored every interactive example. A harness gets tests
before it gets trusted.

**Q4_K_M on a narrow model is not what the name suggests.** K-quants work on
256-element superblocks, and this model's embedding width is 1152, which is not a
multiple of 256. Every tensor with a 1152-long row falls back to a non-K quant: of
183 two-dimensional tensors, 117 are Q5_0 and 14 are Q8_0, and only 52 are actually
Q4_K or Q6_K. The file is about 60% Q5_0 by parameter count. The correlation holds for
183 of 183 two-dimensional tensors: every one with a row length divisible by 256 got a
K-quant, and every one without got Q5_0 or Q8_0. `general.file_type` still reads 15,
the Q4_K_M tag, so the filename is not wrong so much as uninformative. Worth knowing
before picking a quantization for a narrow model. Reproduce with
`python scripts/gguf_header.py models/docker_gemma3_4b_q4km.gguf`.

**What I would do differently.** Write the held-out set first, before the generator,
and never let the same person write both in the same week. The first build of the v2
dataset leaked 12 test prompts verbatim, because the author reached for the same
phrasings twice. Care does not prevent that; a generation-time blocklist does, and
that is now the default. Second: measure a baseline before fine-tuning anything. With
no zero-shot or few-shot number for the base model, 46.6% says how the fine-tune
does, not whether fine-tuning helped at all.

## Limitations

- **Not usable for the two most common Docker verbs.** `run` scores 20.7% and `exec`
  9.1%. Those are the categories people actually need help with.
- **The dataset is the ceiling, not the model.** 594 rows over 298 mostly single-flag
  commands. Accuracy by flag count in the target: 0 flags 74.0%, 1 flag 47.1%,
  2 flags 5.0%, 3+ flags 0.0%. The v1 set contains 17 distinct flag pairs, 47
  occurrences of which are the trivial `-i`+`-t` bundle. The model was asked to
  compose flags it had never seen composed.
- **Ignore the `CONFIDENCE` field.** The dataset generator filled it with
  `random.uniform(0.90, 0.97)`, so the model was trained to predict a random number.
  Worse: `agent.py` gates its `success` flag on that number being ≥ 0.6, so a correct
  command with a low sampled confidence is reported as a failed translation. Replacing
  it with mean token logprob is Milestone 3 in
  [`notes/PROGRESS.md`](notes/PROGRESS.md); it is not done.
- **The 1B-vs-4B comparison is withdrawn**, and the identity of the model behind
  46.6% is unresolved. See Results.
- **No baseline.** Base-model zero-shot and few-shot have never been measured, so the
  value of the fine-tuning step is unknown.
- **116 examples is small.** A binomial 95% interval at n=116 near 50% is about ±9
  points. Differences smaller than that are noise. Single seed, no variance estimate.
- **The test set is one author's phrasing.** Prompts scraped from real Docker
  questions, or collected from other developers, would be stronger evidence.
- **Functional equivalence is a heuristic, not execution.** Two commands scored
  equivalent may still differ in effect. Execution-based scoring is Milestone 4.
- **The CLI lowercases your input**, which destroys build-arg values, env var names,
  and `Dockerfile.test`. 42 of the 594 training prompts contain such tokens.
- **Windows needs the wheel index** in the Quickstart. Without it the install
  compiles llama.cpp and fails on a long path.
- **Only Docker.** The venvy integration's accuracy was withdrawn for the same
  contamination reason and its weights are not published either.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs:
[github.com/pranavkumaarofficial/nlcli-wizard/issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues).

## License

[MIT](LICENSE)
