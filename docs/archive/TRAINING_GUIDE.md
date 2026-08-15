# Training Guide: Adapting nlcli-wizard for Your CLI Tool

This guide shows how to train nlcli-wizard for your own Python CLI tool.

## Overview

The training process:
1. Generate training dataset (NL → CLI command pairs)
2. Fine-tune TinyLlama using Unsloth on Google Colab
3. Quantize to GGUF format for CPU inference
4. Integrate with your CLI tool

## Step 1: Generate Dataset

### Option A: Use the DatasetGenerator

For venvy (already implemented):

```python
from nlcli_wizard.dataset import DatasetGenerator

generator = DatasetGenerator(cli_tool="venvy")
examples = generator.generate_examples(num_examples=1500)
generator.save_to_jsonl(examples, Path("data/training_examples.jsonl"))
```

### Option B: Create Custom Generator

For your own tool, extend `DatasetGenerator`:

```python
class MyToolGenerator(DatasetGenerator):
    def __init__(self):
        super().__init__(cli_tool="mytool")

    def _generate_mytool_examples(self, count: int):
        templates = [
            # (natural_language, command, confidence, explanation)
            ("run all tests", "mytool test --all", 0.95, "Runs all test suites"),
            ("test the {module} module", "mytool test {module}", 0.92, "Tests specific module"),
        ]

        # Generate variations with different values
        # Return list of example dicts
```

### Dataset Quality Tips

- **Aim for 1,500-2,000 examples** minimum
- **Diverse phrasing**: formal, casual, abbreviated
- **Cover all commands**: 80/20 rule (common commands get more examples)
- **Real-world patterns**: how users actually talk
- **Include edge cases**: ambiguous inputs, synonyms

### Example Format

Each example should be in Alpaca format:

```json
{
  "instruction": "Translate to venvy command: create a python 3.10 environment",
  "input": "",
  "output": "COMMAND: venvy create --python 3.10\nCONFIDENCE: 0.95\nEXPLANATION: Creates a new virtual environment with Python 3.10\n"
}
```

## Step 2: Fine-Tune Model in Google Colab

### Setup Colab Notebook

Use the provided `training/colab_training.ipynb` or create your own:

```python
# Install dependencies
!pip install unsloth transformers datasets peft trl bitsandbytes accelerate

from unsloth import FastLanguageModel
import torch

# Load base model (TinyLlama)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length = 512,
    dtype = None,  # Auto-detect
    load_in_4bit = True,  # Use 4-bit quantization
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)
```

### Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="training_examples.jsonl")

# Format for TinyLlama chat template
def format_prompts(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"<s>[INST] {instruction} [/INST]\n{output}</s>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)
```

### Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = 512,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs",
    ),
)

trainer.train()
```

### Expected Training Time

- **Dataset size**: 1,500 examples
- **Hardware**: Google Colab T4 GPU (free tier)
- **Time**: ~2-3 hours for 3 epochs
- **Memory**: ~15GB VRAM with 4-bit quantization

## Step 3: Quantize to GGUF

After training, convert to GGUF format for llama.cpp:

```python
# Save trained model
model.save_pretrained("models/nlcli-tinyllama-mytool")
tokenizer.save_pretrained("models/nlcli-tinyllama-mytool")

# Convert to GGUF (using llama.cpp tools)
# Download: https://github.com/ggerganov/llama.cpp

!python llama.cpp/convert.py models/nlcli-tinyllama-mytool \
    --outfile models/nlcli-tinyllama-mytool-f16.gguf \
    --outtype f16

# Quantize to 4-bit
!./llama.cpp/quantize models/nlcli-tinyllama-mytool-f16.gguf \
    models/nlcli-tinyllama-mytool-q4_k_m.gguf q4_k_m
```

### Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="models/nlcli-tinyllama-mytool-q4_k_m.gguf",
    path_in_repo="nlcli-tinyllama-mytool-q4_k_m.gguf",
    repo_id="YOUR_USERNAME/nlcli-tinyllama-mytool",
    repo_type="model",
)
```

## Step 4: Integrate with Your CLI

### Update Model Path

In `nlcli_wizard/model.py`, update:

```python
DEFAULT_MODEL_REPO = "YOUR_USERNAME/nlcli-tinyllama-mytool"
DEFAULT_MODEL_FILE = "nlcli-tinyllama-mytool-q4_k_m.gguf"
```

### Add Wizard Flag to Your CLI

In your CLI tool's main file:

```python
import click
from nlcli_wizard.agent import NLCLIAgent
import subprocess

@click.group()
@click.option('-w', '--wizard', is_flag=True, help='Use natural language wizard')
@click.pass_context
def cli(ctx, wizard):
    """Your CLI tool"""
    ctx.obj = {'wizard': wizard}

@cli.command()
@click.argument('args', nargs=-1)
@click.pass_context
def create(ctx, args):
    """Create something"""
    if ctx.obj.get('wizard'):
        # Wizard mode - args is natural language
        nl_input = " ".join(args)
        agent = NLCLIAgent(cli_tool="mytool")

        result = agent.translate(nl_input)

        if result["success"]:
            print(f"Command: {result['command']}")
            print(f"Explanation: {result['explanation']}")

            confirm = input("Execute? (y/n): ")
            if confirm.lower() == 'y':
                # Parse and execute
                cmd_parts = result['command'].split()[1:]  # Remove tool name
                # Execute with parsed args
        else:
            print("Could not understand. Try regular syntax.")
    else:
        # Regular mode
        # ... your normal command logic
```

### Usage

```bash
# Regular mode
mytool create --name myproject --type web

# Wizard mode
mytool -w create a new web project called myproject
```

## Step 5: Testing & Iteration

### Test Accuracy

```python
from nlcli_wizard.agent import NLCLIAgent

agent = NLCLIAgent(cli_tool="mytool")

test_cases = [
    ("create a new project", "mytool create"),
    ("run all tests", "mytool test --all"),
    # ... more test cases
]

correct = 0
for nl, expected_cmd in test_cases:
    result = agent.translate(nl)
    if result["command"] == expected_cmd:
        correct += 1

print(f"Accuracy: {correct/len(test_cases)*100:.1f}%")
```

### Common Issues & Fixes

**Low Accuracy (<70%)**
- Add more training examples (aim for 2,000+)
- Increase training epochs (3 → 5)
- Check dataset quality (typos, inconsistencies)

**Slow Inference (>5s)**
- Use smaller quantization (q4_k_m → q4_0)
- Reduce context window (512 → 256)
- Ensure llama-cpp-python compiled with optimizations

**Model Hallucinates Commands**
- Add more negative examples (unclear inputs → error)
- Increase confidence threshold (0.6 → 0.75)
- Add command validation/whitelisting

## Performance Benchmarks

Expected results with 1,500 training examples:

| Metric | Value |
|--------|-------|
| Accuracy | 75-85% |
| Inference Time | <2s (CPU) |
| Model Size | ~650MB |
| RAM Usage | ~2GB |
| Training Time | 2-3 hours (T4 GPU) |

## Next Steps

- **Week 1-2**: Generate dataset, validate quality
- **Week 3**: Train model in Colab, evaluate
- **Week 4**: Quantize, integrate with CLI
- **Week 5**: Test, document, create demo

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for full roadmap.
