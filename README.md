# nlcli-wizard

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing)
[![Reddit Discussion](https://img.shields.io/badge/Reddit-Discussion-orange.svg)](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)

Natural language control for Python CLI tools using locally-trained SLMs. No cloud, no API keys, runs offline on CPU.

> **📢 Discussion:** See the [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/) for technical discussion and community feedback.
>
> **🚀 Quick Start:** Train your own model in [Google Colab](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing) (free T4 GPU, ~2.5 hours)

```bash
# Instead of memorizing flags
venvy register --project /path/to/project --name myenv

# Just describe what you want
venvy -w "register this project as myenv"
```

**Portfolio project** demonstrating local LLM fine-tuning for developer tooling.

## Results

- **83.3% accuracy** on venvy command translation
- **810MB model** (Q4_K_M quantized from 2.01GB)
- **~1.5s inference** on CPU (4 threads)
- **Zero fabrication** - all training data verified against source code
- **1,500 training examples** generated programmatically

## Technical Stack

- **Base**: google/gemma-3-1b-it (March 2025)
- **Training**: Unsloth 2025.1+ with QLoRA (2x speed, 70% less VRAM)
- **Quantization**: GGUF Q4_K_M with smart fallback (Q4_K/Q5_0/Q6_K mix)
- **Inference**: llama.cpp for CPU execution
- **Data**: 1,500 verified examples in Alpaca format
- **Loss**: 0.135 (train), 0.142 (val) - no overfitting

## How It Works

```
User: "show my venvs sorted by size"
  ↓
Gemma 3 1B (fine-tuned)
  ↓
Command: venvy ls --sort size
  ↓
Preview → Confirm → Execute
```

Training pipeline:
1. Generate synthetic dataset from CLI help docs
2. Fine-tune Gemma 3 1B with QLoRA on Colab T4
3. Quantize to Q4_K_M with llama.cpp
4. Run locally with llama-cpp-python

## Project Status

- [x] Dataset generation (1,500 verified examples)
- [x] Fine-tuned Gemma 3 1B with QLoRA
- [x] Quantized to Q4_K_M (810MB)
- [x] Validated 83.3% accuracy
- [ ] CLI integration with venvy
- [ ] PyPI package release

## Technical Deep Dive

**Key learnings for local LLM deployment:**

1. **QLoRA Training**: Only 1.29% of parameters trainable (14M/1.1B) using low-rank adapters
2. **Dynamic 4-bit**: Unsloth's dynamic quantization saves 70% VRAM during training
3. **Smart Quantization**: Mixed precision (Q4_K/Q5_0/Q6_K) adapts per-layer based on tensor dimensions
4. **Zero Fabrication**: Verified all commands against source code to prevent hallucination
5. **Alpaca Format**: Instruction/input/output structure for consistent fine-tuning

**Training metrics:**
- 3 epochs @ 2.5 hours (Colab T4)
- Final loss: 0.135 (train), 0.142 (val)
- Manual test: 6/6 correct (100%)
- Validation: 125/150 correct (83.3%)

### 🎓 Train Your Own Model

**Interactive Tutorial:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL?usp=sharing)

Complete training pipeline with step-by-step explanations:
- Fine-tune Gemma 3 1B with QLoRA
- Generate importance matrix for smart quantization
- Convert to GGUF for CPU inference
- Test locally with llama-cpp-python

No ML experience required - runs on free Colab T4 GPU in ~2.5 hours.

See [training/](training/) for additional notebooks and scripts.

## Use Case: venvy

Proof-of-concept integration with [venvy](https://github.com/pranavkumaarofficial/venvy) - a fast Python virtual environment manager.

Supports 7 commands: `register`, `ls`, `scan`, `current`, `cleanup`, `shell-hook`, `stats`

**Example translations:**
```
"show my environments" → venvy ls
"register this project as myenv" → venvy register --name myenv
"clean up old venvs" → venvy cleanup --days 90
```

## Files of Interest

- [dataset.py](nlcli_wizard/dataset.py) - Synthetic data generation with zero fabrication
- [train_gemma3_colab.ipynb](training/train_gemma3_colab.ipynb) - Complete training pipeline (coming soon)
- [evaluate_accuracy.py](test/evaluate_accuracy.py) - Validation script
- [TRAINING_GUIDE_COLAB.md](docs/TRAINING_GUIDE_COLAB.md) - Technical deep dive

## Contributing

Interested in contributing? Check out [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Adding support for new CLI tools
- Improving accuracy and data quality
- Mobile deployment and benchmarking
- Testing and validation

## Community & Discussion

- **Reddit:** [r/LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)
- **Issues:** [GitHub Issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues)
- **Related:** [venvy](https://github.com/pranavkumaarofficial/venvy) - Virtual environment manager

## License

[MIT License](LICENSE) - Pranav Kumaar

