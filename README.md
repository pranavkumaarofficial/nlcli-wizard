# nlcli-wizard

**Natural Language Interface for Python CLI Tools**

A framework that enables natural language interaction with any Python command-line tool using locally-running Small Language Models (SLMs). No cloud dependencies, no API keys - everything runs on your machine.

## What It Does

Instead of memorizing complex CLI flags and syntax:

```bash
# Traditional CLI
venvy create --python 3.10 --name myenv --location ./envs/

# With nlcli-wizard
venvy -w "create a Python 3.10 environment called myenv"
```

The wizard understands your intent, translates it to the correct command, and executes it - all using a tiny, locally-trained language model.

## Key Features

- **Locally Running**: Fine-tuned SLM runs entirely on your CPU (~650MB)
- **Zero Cloud Calls**: No API keys, no data leaves your machine
- **Fast Inference**: <2s response time on modern CPUs
- **Extensible Framework**: Easy to adapt for any Python CLI tool
- **Safe Execution**: Always previews commands before execution
- **Fallback Support**: If model fails, falls back to standard CLI

## Technical Stack (2025 Latest)

- **Base Model**: Gemma 3 1B (phone-optimized, released March 2025)
- **Training**: Unsloth 2025.1+ (2x faster fine-tuning, 70% less memory, Dynamic 4-bit)
- **Inference**: llama-cpp-python for local CPU execution
- **Quantization**: GGUF Q4_K_M with importance matrix for optimal quality
- **Optimization**: QLoRA + PEFT for efficient fine-tuning
- **Training Platform**: Google Colab (T4 GPU)

## Architecture

```
┌─────────────────┐
│  User Input     │  "create a Python 3.10 venv called test"
└────────┬────────┘
         │
┌────────▼────────┐
│  NL Parser      │  Tokenize & normalize
└────────┬────────┘
         │
┌────────▼────────┐
│  SLM Inference  │  TinyLlama 1.1B (quantized)
└────────┬────────┘
         │
┌────────▼────────┐
│  Command Gen    │  "venvy create --python 3.10 --name test"
└────────┬────────┘
         │
┌────────▼────────┐
│  Preview/Exec   │  Show command → Confirm → Execute
└─────────────────┘
```

## Use Cases

### Demonstrated (venvy)
Virtual environment management with natural language

### Future Extensions
- **pytest-wizard**: "run all tests in the auth module"
- **git-wizard**: "create a new branch from main for the login feature"
- **pip-wizard**: "install the latest version of requests"

## Project Status

**Current Phase**: Dataset Generation & Model Training

- [x] Framework architecture design
- [x] Technical analysis & stack selection
- [ ] Generate training dataset (1,500+ examples)
- [ ] Fine-tune TinyLlama with Unsloth
- [ ] Integrate with venvy as proof-of-concept
- [ ] Quantize & optimize for CPU inference
- [ ] Package for PyPI distribution

## Implementation Plan

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the complete 5-week roadmap.

## Why This Project?

This started as a way to make CLI tools more accessible, but evolved into a technical showcase demonstrating:

- **ML Engineering**: Fine-tuning SLMs with modern techniques (Unsloth, QLoRA)
- **Local-First AI**: Running LLMs efficiently on consumer hardware
- **Software Architecture**: Designing extensible, framework-level solutions
- **UX Innovation**: Bridging the gap between natural language and technical tools

## Installation (Coming Soon)

```bash
pip install nlcli-wizard

# For venvy integration
pip install venvy
venvy -w "your natural language command"
```

## Training Your Own

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for instructions on adapting nlcli-wizard for your own CLI tool.

## Performance Expectations

- **Model Size**: ~650MB (4-bit quantized)
- **Inference Time**: <2s on CPU
- **Accuracy**: 75-85% on domain-specific commands
- **RAM Usage**: ~2GB during inference
- **Training Time**: ~2-3 hours on Colab T4 GPU

## License

MIT

## Related Projects

- [venvy](https://github.com/pranavkumar2004/venvy) - Fast virtual environment manager (first use-case for nlcli-wizard)

## Author

Pranav Kumaar

---

**Note**: This is a portfolio/research project exploring the intersection of SLMs and developer tooling. While functional, it's optimized for learning and demonstration rather than production-scale deployment.
