# Contributing to nlcli-wizard

Thanks for your interest! This project demonstrates local LLM fine-tuning for CLI command translation.

## 🎯 High-Impact Contributions

### 1. New CLI Tool Support

Add support for popular CLI tools:

**High Priority:**
- `docker` - Container management (high complexity, 3B model recommended)
- `git` - Version control (medium complexity)
- `pytest` - Python testing (medium complexity)
- `npm`/`pip` - Package managers (low complexity)

**How to add:**
1. Create dataset generator in `nlcli_wizard/dataset_<tool>.py`
2. Verify all commands against source code (zero fabrication!)
3. Generate 1,000-1,500 examples
4. Follow the pattern in [dataset.py](nlcli_wizard/dataset.py)

### 2. Data Quality Improvements

**Current accuracy:** 83.3% (125/150)
**Goal:** 90%+

Ideas:
- Add more diverse phrasing variations
- Error correction pairs (wrong command → right command)
- Multi-step command chains
- Edge case handling for ambiguous queries

### 3. Mobile Deployment

Test and optimize for mobile:
- Android deployment with llama.cpp
- iOS deployment with llama.cpp
- Inference benchmarks on ARM processors
- Compare Q4_0 vs Q4_K_M on mobile

### 4. Accuracy Analysis

Analyze the 16.7% error rate:
- Categorize failed examples (ambiguous, missing context, hallucination)
- Identify patterns in failures
- Propose data improvements
- Test with larger models (3B, 7B)

## 🛠️ How to Contribute

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/nlcli-wizard.git
cd nlcli-wizard
```

### 2. Install Dependencies

```bash
# Basic dependencies
pip install -e .

# Training dependencies (for dataset generation)
pip install -e ".[training]"

# Development dependencies (for testing)
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

**For dataset changes:**
- Verify ALL commands against source code
- Run generation and inspect output
- Check for fabricated commands
- Validate output format (Alpaca)

**For code changes:**
- Follow existing code style
- Add docstrings
- Add type hints where helpful

### 5. Test Your Changes

```bash
# Generate dataset
python -m nlcli_wizard.dataset

# Run validation
python test/evaluate_accuracy.py

# Run tests (if available)
pytest
```

### 6. Submit Pull Request

- Clear description of changes
- Link to related issue (if any)
- Include test results/validation
- Update docs if needed

## 📋 Current Needs

**High Priority:**
- [ ] Docker command dataset (500+ examples, verified)
- [ ] Git command dataset (500+ examples, verified)
- [ ] Mobile inference benchmarks (Android/iOS)
- [ ] Accuracy analysis on failed cases

**Medium Priority:**
- [ ] Integration with venvy CLI (`-w` flag)
- [ ] Larger model comparison (1B vs 3B vs 7B)
- [ ] More training epochs (3 → 5) for accuracy boost
- [ ] HuggingFace model card and hosting

**Nice to Have:**
- [ ] Demo video/GIF
- [ ] Blog post (technical deep dive)
- [ ] Web interface for testing
- [ ] Comparison with shellgpt/other tools

## 💡 Dataset Guidelines

**Zero Fabrication Rule:**
- EVERY command must exist in the actual CLI tool
- Verify against source code or help docs
- No hallucinated flags or options
- No invented commands

**Example Verification:**
```python
# BAD - Fabricated command
{"instruction": "create a venv", "output": "venvy create"}  # 'create' doesn't exist!

# GOOD - Verified command
{"instruction": "register this venv", "output": "venvy register"}  # exists in CLI
```

**Diversity:**
- Multiple phrasings per command
- Different contexts (with/without parameters)
- Natural variations ("list", "show", "display")
- Include edge cases

## 🐛 Bug Reports

Found a bug? Open an issue with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version)

## 💬 Questions?

- **Technical questions:** [GitHub Issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues)
- **General discussion:** [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/)

## 📜 Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Help others learn
- No spam or self-promotion
- By submitting a Pull Request, contributors agree that:
  - Their contributions are made under the same **MIT License** as this repository.
  - Copyright and overall ownership of the project remain with **Pranav Kumaar** (the project maintainer).
  - Contributions may be modified, merged, or redistributed as part of the project under the MIT terms.



## 🙏 Recognition

Contributors will be:
- Listed in README acknowledgments
- Mentioned in release notes
- Credited in any papers/posts about the project

---

**Thank you for contributing to local-first AI tooling!** 🚀
