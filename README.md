# nlcli-wizard

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing)
[![Reddit Discussion](https://img.shields.io/badge/Reddit-Discussion-orange.svg)](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)

A framework for adding natural language interfaces to CLI tools using locally-trained small language models. No cloud APIs, no subscriptions -- runs offline on CPU.

```bash
# Instead of memorizing flags
docker run -d -p 8080:80 --name web -e NODE_ENV=production nginx

# Just describe what you want
docker -w "run nginx on port 8080 with production env in background"
```

> **📢 Discussion:** See the [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/) for technical discussion and community feedback.


## Demo

[https://github.com/user-attachments/assets/VIDEO_ID_HERE](https://github.com/user-attachments/assets/2d7ca418-d6b2-4449-a81e-417df9666d44)

## Results: Docker CLI translation

> **This project previously published 94% Docker accuracy. That number was wrong.**
> It was measured on training data. The corrected figure is 46.6%. The full account
> is in [`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md) — what broke, how it
> was found, and what replaced it. Both numbers are kept side by side below rather
> than the old one being quietly deleted.

Gemma 3 4B, QLoRA fine-tuned on 594 templated Docker examples, Q4_K_M on CPU.
Evaluated on 116 hand-written held-out prompts with zero prompt overlap with training
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
requires it. That is memorization of surface form — and it is what the 94% measured.

**Status of the 1B vs 4B comparison:** withdrawn pending re-measurement. The claimed
"capacity ceiling" at 73–76% was attributed to the 1B model's parameter count. The
corrected results suggest the ceiling was imposed by the dataset — 594 examples over
298 mostly single-flag commands cannot teach flag composition — and that the 4B model
hit the same wall unnoticed behind a contaminated metric.

## Quick start

### Use the pre-trained Docker model

```bash
# Clone and install
git clone https://github.com/pranavkumaarofficial/nlcli-wizard.git
cd nlcli-wizard
pip install -e .

# Download the 4B GGUF model (~2.5GB) and place in models/
# (HuggingFace repo: pranavkumaarofficial/nlcli-gemma3-docker)

# Translate
python -m nlcli_wizard.cli translate --cli-tool docker "run nginx on port 8080 in background"
# Command: docker run -d -p 8080:80 nginx
# Runs nginx container in detached mode, mapping port 8080 to 80
```

> **Note on the `CONFIDENCE` field.** The model emits one, but it is meaningless: the
> dataset generator filled it with `random.uniform(0.90, 0.97)`, so the model was
> trained to predict a random number. It is being replaced with mean token logprob
> (see [`notes/PROGRESS.md`](notes/PROGRESS.md), Milestone 3). Do not rely on it.

### Train your own model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing)

The training notebook runs on free Colab T4 with step-by-step explanations. No ML experience required.

```bash
# 1. Generate training data for your CLI tool
python -m nlcli_wizard.dataset_docker  # generates data/docker_training.jsonl

# 2. Open the Colab notebook and train (free T4 GPU)
# 3. Download the GGUF model and place in models/
# 4. Run evaluation
python -m eval.run_eval --model models/docker_gemma3_4b_q4km.gguf --template gemma3
```

## How it works

```
User: "scale web service to 3 instances"
  |
  v
Prompt: "<start_of_turn>user\nTranslate to docker command: ...<end_of_turn>\n<start_of_turn>model\n"
  |
  v
Gemma 3 4B (fine-tuned, quantized Q4_K_M, running on CPU via llama.cpp)
  |
  v
COMMAND: docker-compose up --scale web=3
CONFIDENCE: 0.92
EXPLANATION: Scales the web service to 3 replicas
  |
  v
Preview -> Confirm -> Execute
```

The model outputs structured `COMMAND / CONFIDENCE / EXPLANATION` format. The agent parses this and asks for confirmation before executing.

## Architecture

The framework is tool-agnostic. To add support for a new CLI tool:

1. Write a dataset generator -- parse `--help` output, generate NL variations for each command
2. Train on Colab -- swap the dataset file, run the notebook
3. Drop in the GGUF -- place the quantized model in `models/`
4. Register in MODEL_REGISTRY -- add an entry in `model.py`

```
nlcli-wizard/
  nlcli_wizard/
    cli.py              # CLI interface
    model.py            # Model loading, MODEL_REGISTRY
    agent.py            # Prompt formatting, output parsing
    dataset.py          # Venvy dataset generator
    dataset_docker.py   # Docker dataset generator (594 examples)
  training/
    nlcli_wizard_training_[PUBLIC].ipynb   # Colab training notebook
  eval/
    contamination.py    # Train/test leakage auditing
    splits.py           # Command-level (non-leaking) splits
    metrics.py          # exact / normalized / functional scoring
    run_eval.py         # Evaluation entry point
  tests/                # pytest suite for the harness
  data/
    docker_training.jsonl        # Generated training data
    docker_test_handwritten.jsonl # Hand-written held-out set (116)
  models/
    *.gguf              # Quantized models (gitignored)
  scripts/
    docker-wizard.sh    # Shell wrapper
    docker-wizard.ps1   # PowerShell wrapper
    plot_comparison.py  # Generate comparison charts
```

## Technical stack

- **Base model**: Gemma 3 4B-Instruct (via Unsloth)
- **Training**: QLoRA with Unsloth on free Colab T4
- **Quantization**: GGUF Q4_K_M with importance matrix via llama.cpp
- **Inference**: llama.cpp (llama-server / llama-cpp-python), CPU, 4 threads
- **Output format**: Structured COMMAND/CONFIDENCE/EXPLANATION

## Supported tools

| Tool | Dataset | Model | Accuracy | Status |
|------|---------|-------|----------|--------|
| Docker | 594 rows / 298 unique cmds | Gemma 3 4B | 46.6% | Available |
| [Venvy](https://github.com/pranavkumaarofficial/venvy) | 1,500 rows / 230 unique | Gemma 3 1B | withdrawn | Needs re-eval |
| Kubernetes | -- | -- | -- | Planned |
| Git | -- | -- | -- | Planned |

### Venvy (proof-of-concept)

The first tool integrated was [venvy](https://github.com/pranavkumaarofficial/venvy), a fast Python virtual environment manager:

```
"show my environments sorted by size"  ->  venvy ls --sort size
"register this project as myenv"       ->  venvy register --name myenv
"clean up old venvs"                   ->  venvy cleanup --days 90
```

Trained on Gemma 3 1B. The previously published 83% accuracy is **withdrawn**: the
venvy dataset has 1,500 rows but only 230 unique instructions (6.5x duplication), and
95 of the 100 evaluation prompts appeared verbatim in training. The figure measured
memorization recall, and the model got 17% of memorized items wrong even so. A
venvy equivalent of `data/docker_test_handwritten.jsonl` is needed before any number
is republished. See [`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md).

## Roadmap

- [x] Venvy proof-of-concept (Gemma 3 1B) — accuracy withdrawn, needs a clean test set
- [x] Docker support (Gemma 3 4B) — 46.6% on the corrected harness
- [x] 1B vs 4B comparison — withdrawn, see docs/EVAL_METHODOLOGY.md
- [x] Training notebook with step-by-step explanations
- [x] Contamination-free eval harness (`eval/`) + methodology writeup
- [ ] Baselines: base model zero-shot / few-shot vs fine-tuned
- [ ] Auto-ingestion pipeline: `--help` docs in, training data out, weights packaged
- [ ] Error correction feedback loop (command fails -> suggest fix)
- [ ] PyPI package release
- [ ] Kubernetes and Git datasets

The end goal: any CLI tool maintainer can point this at their docs, generate training data, fine-tune a model, and ship weights alongside their package. Their users get `tool -w "what I want to do"` for free.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Adding new CLI tool support
- Improving dataset quality
- Testing and evaluation

## Community

- [Reddit: r/LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)
- [GitHub Issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues)

## License

[MIT License](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pranavkumaarofficial/nlcli-wizard&type=date&legend=top-left)](https://www.star-history.com/#pranavkumaarofficial/nlcli-wizard&type=date&legend=top-left)

---

Built by [Pranav Kumaar](https://github.com/pranavkumaarofficial) | [nlcli-wizard](https://github.com/pranavkumaarofficial/nlcli-wizard) | [venvy](https://github.com/pranavkumaarofficial/venvy)
