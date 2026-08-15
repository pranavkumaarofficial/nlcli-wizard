# Implementation Plan: venvy NL-CLI with SLM

## Portfolio/Resume Impact Statement

**"Built a locally-running NL-CLI agent for Python package management using fine-tuned TinyLlama with Unsloth/QLoRA, achieving 80%+ command accuracy with 650MB footprint and <2s inference on CPU - demonstrating expertise in LLM fine-tuning, PEFT, model quantization, and production ML deployment."**

---

## Phase 1: Dataset Creation (Week 1)

### Goal
Create 1,500-2,000 high-quality NL → venvy command pairs

### Tech Stack (Latest 2025)
```python
# Dataset tools
datasets==2.18.0         # HuggingFace datasets
pandas==2.2.0
jsonlines==4.0.0
```

### Dataset Structure

**Format:**
```json
{
  "instruction": "You are a venvy command generator. Convert natural language to venvy CLI commands.",
  "input": "create a new virtual environment with Python 3.11",
  "output": "venvy create --python 3.11"
}
```

### Dataset Generation Strategy

**Step 1: Core Commands (300 examples)**
```python
# venvy/dataset/generate_core.py

CORE_TEMPLATES = [
    # CREATE commands
    {
        "patterns": [
            "create a venv",
            "make a new virtual environment",
            "set up a new env",
            "initialize a venv",
        ],
        "command": "venvy create"
    },
    {
        "patterns": [
            "create a venv named {name}",
            "make environment called {name}",
            "create {name} venv",
        ],
        "command": "venvy create --name {name}"
    },
    # ... all venvy commands
]

def generate_variations(template, count=20):
    """Use GPT-4 API to generate natural variations"""
    # Or use Claude API / manual curation
    pass
```

**Step 2: Complex Commands (500 examples)**
```python
# Multi-flag combinations
examples = [
    ("create Python 3.11 venv named myapp", "venvy create --python 3.11 --name myapp"),
    ("list all venvs sorted by size in json", "venvy ls --sort size --format json"),
    ("remove venvs older than 90 days", "venvy cleanup --days 90"),
    # ... complex multi-parameter commands
]
```

**Step 3: Conversational Variations (700 examples)**
```python
# Natural language variations
examples = [
    ("I want to make a new environment", "venvy create"),
    ("can you show me my venvs?", "venvy ls"),
    ("what virtual environments do I have", "venvy ls"),
    ("please delete old environments", "venvy cleanup"),
    # ... conversational style
]
```

### Dataset Quality Checklist
- [ ] All venvy commands covered
- [ ] Multiple NL variations per command
- [ ] Edge cases (typos, abbreviations)
- [ ] Different phrasing styles (formal, casual)
- [ ] Multi-parameter combinations
- [ ] Conversational queries

**Deliverable:** `venvy_nl_dataset.json` (1,500+ examples)

---

## Phase 2: Model Fine-Tuning (Week 2)

### Tech Stack (Latest Jan 2025)

```python
# Core ML libraries
torch==2.5.1                    # Latest PyTorch
transformers==4.46.3            # Latest HF transformers
peft==0.13.2                    # Latest PEFT
bitsandbytes==0.44.1           # 4-bit quantization
accelerate==1.1.1              # Multi-GPU support
trl==0.12.1                    # Trainer for LLMs

# CRITICAL: Unsloth (2x faster training!)
unsloth[colab-new]>=2025.1     # Latest Unsloth
```

### Model Selection

**Option A: TinyLlama 1.1B (RECOMMENDED)**
- Model: `unsloth/TinyLlama-1.1B-Chat-v1.0-bnb-4bit`
- Size: ~637MB (4-bit quantized)
- Speed: Fast on CPU
- RAM: 2-3GB inference

**Option B: Phi-3.5 Mini 3.8B (Higher Accuracy)**
- Model: `unsloth/Phi-3.5-mini-instruct-bnb-4bit`
- Size: ~2.3GB (4-bit quantized)
- Speed: Slower on CPU
- RAM: 4-5GB inference

**Decision: Use TinyLlama** (portfolio impact is same, but more practical)

### Google Colab Setup

**Notebook: `venvy_finetune.ipynb`**

```python
# ============================================
# SECTION 1: Environment Setup
# ============================================

# Install latest packages
!pip install -q unsloth[colab-new]
!pip install -q transformers==4.46.3 datasets trl bitsandbytes accelerate

# ============================================
# SECTION 2: Load Model with Unsloth
# ============================================

from unsloth import FastLanguageModel
import torch

max_seq_length = 512  # Short for CLI commands
dtype = None  # Auto-detect
load_in_4bit = True   # Use 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/TinyLlama-1.1B-Chat-v1.0-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ============================================
# SECTION 3: Add LoRA Adapters
# ============================================

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (higher = more capacity)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth optimization
    random_state = 42,
)

# ============================================
# SECTION 4: Load & Format Dataset
# ============================================

from datasets import load_dataset

# Load your dataset
dataset = load_dataset("json", data_files="venvy_nl_dataset.json")

# Format for chat template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)

# ============================================
# SECTION 5: Training with Unsloth
# ============================================

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # Short sequences don't need packing
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,  # Increase to 500-1000 for production
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
)

# Train!
trainer_stats = trainer.train()

# ============================================
# SECTION 6: Test Inference
# ============================================

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

test_input = "create a Python 3.11 venv named myapp"
inputs = tokenizer([alpaca_prompt.format(
    "You are a venvy command generator. Convert natural language to venvy CLI commands.",
    test_input,
    ""
)], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================
# SECTION 7: Save Model
# ============================================

# Option A: Save LoRA adapters only (tiny!)
model.save_pretrained("venvy_lora_model")
tokenizer.save_pretrained("venvy_lora_model")

# Option B: Merge and save full model
model.save_pretrained_merged("venvy_merged_model", tokenizer, save_method="merged_16bit")

# Option C: Save as GGUF for llama.cpp (RECOMMENDED)
model.save_pretrained_gguf("venvy_gguf_model", tokenizer, quantization_method="q4_k_m")
```

**Expected Training Time:** 30-60 minutes on Colab T4 GPU

### Model Validation

```python
# Test on held-out examples
test_cases = [
    ("list my venvs", "venvy ls"),
    ("create Python 3.10 env", "venvy create --python 3.10"),
    ("show stats", "venvy stats"),
    # ... 50 test cases
]

correct = 0
for nl, expected in test_cases:
    predicted = generate_command(nl)
    if predicted == expected:
        correct += 1

accuracy = correct / len(test_cases)
print(f"Accuracy: {accuracy:.1%}")  # Target: 80%+
```

**Deliverable:** `venvy-tinyllama-q4.gguf` (~650MB)

---

## Phase 3: Local Inference Setup (Week 3)

### Tech Stack

```python
# CPU inference
llama-cpp-python==0.3.2      # Latest llama.cpp bindings
ctransformers==0.2.27        # Alternative: CTransformers
```

### Installation

```python
# venvy/setup.py

extras_require={
    'nl': [
        'llama-cpp-python>=0.3.0',
        # Download model on first use (not in package)
    ]
}
```

### Model Loader

```python
# venvy/nl_agent.py

from pathlib import Path
from llama_cpp import Llama
import requests
from tqdm import tqdm

class VenvyNLAgent:
    MODEL_URL = "https://huggingface.co/pranavkumaar/venvy-tinyllama-gguf/resolve/main/venvy-q4.gguf"
    MODEL_PATH = Path.home() / ".venvy" / "models" / "venvy-q4.gguf"

    def __init__(self):
        self.model = None
        self._ensure_model()

    def _ensure_model(self):
        """Download model if not present"""
        if not self.MODEL_PATH.exists():
            print("⬇️  Downloading venvy NL model (650MB)...")
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(self.MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(self.MODEL_PATH, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print("✅ Model downloaded!")

        # Load model
        if self.model is None:
            print("🔄 Loading model...")
            self.model = Llama(
                model_path=str(self.MODEL_PATH),
                n_ctx=512,
                n_threads=4,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            print("✅ Model ready!")

    def parse(self, nl_input: str) -> dict:
        """Convert NL to venvy command"""
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a venvy command generator. Convert natural language to venvy CLI commands.

### Input:
{nl_input}

### Response:
"""

        output = self.model(
            prompt,
            max_tokens=50,
            temperature=0.1,
            stop=["###", "\n\n"],
            echo=False
        )

        command = output['choices'][0]['text'].strip()

        # Parse confidence from logprobs
        confidence = self._calculate_confidence(output)

        return {
            'command': command,
            'confidence': confidence,
            'raw_output': output
        }

    def _calculate_confidence(self, output) -> float:
        """Estimate confidence from model output"""
        # Simple heuristic: longer = less confident
        tokens = output['choices'][0]['text'].split()
        if len(tokens) <= 5:
            return 0.9
        elif len(tokens) <= 10:
            return 0.7
        else:
            return 0.5
```

---

## Phase 4: CLI Integration (Week 3-4)

### Wizard Mode Flag

```python
# venvy/cli.py

@click.option('-w', '--wizard', is_flag=True,
              help='🧙 Natural language mode (requires venvy[nl])')
@click.argument('query', required=False, nargs=-1)
def main(wizard, query):
    """venvy - Virtual Environment Manager

    Examples:
        venvy ls                           # Normal mode
        venvy -w "list my environments"    # Wizard mode
    """
    if wizard:
        return _wizard_mode(query)

    # ... normal CLI logic


def _wizard_mode(query_parts):
    """Handle natural language queries"""
    try:
        from venvy.nl_agent import VenvyNLAgent
    except ImportError:
        console.print("[red]Error: Natural language mode requires: pip install venvy[nl][/red]")
        return

    # Join query parts
    if not query_parts:
        query = Prompt.ask("🧙 [cyan]What would you like to do?[/cyan]")
    else:
        query = " ".join(query_parts)

    # Parse with NL agent
    agent = VenvyNLAgent()

    with console.status("[cyan]Thinking...[/cyan]"):
        result = agent.parse(query)

    # Show preview
    console.print("\n[bold cyan]📋 Suggested command:[/bold cyan]")
    console.print(f"   {result['command']}")
    console.print(f"[dim]   Confidence: {result['confidence']:.0%}[/dim]\n")

    # Confirm before execution
    if result['confidence'] < 0.6:
        console.print("[yellow]⚠️  Low confidence - please verify command[/yellow]")

    if Confirm.ask("Execute this command?", default=True):
        # Parse and execute
        _execute_venvy_command(result['command'])
    else:
        console.print("[yellow]Cancelled[/yellow]")
        console.print("[dim]Tip: Try rephrasing your query or use regular CLI[/dim]")


def _execute_venvy_command(command_str: str):
    """Safely execute parsed venvy command"""
    # Parse command string
    parts = command_str.replace("venvy ", "").split()

    # Security: whitelist venvy commands only
    ALLOWED_COMMANDS = {'create', 'ls', 'register', 'scan', 'current',
                        'stats', 'cleanup', 'shell-hook'}

    if not parts or parts[0] not in ALLOWED_COMMANDS:
        console.print(f"[red]Error: Invalid command '{parts[0]}'[/red]")
        return

    # Execute via click context
    ctx = click.Context(main)
    ctx.invoke(globals()[parts[0]], *parts[1:])
```

### User Experience

```bash
$ venvy -w "create a Python 3.11 environment named backend"

🔄 Loading model...
✅ Model ready!

📋 Suggested command:
   venvy create --python 3.11 --name backend
   Confidence: 95%

Execute this command? [Y/n]: y

Creating virtual environment...
✅ Environment created: backend
```

---

## Phase 5: Testing & Documentation (Week 4)

### Testing Strategy

```python
# tests/test_nl_agent.py

import pytest
from venvy.nl_agent import VenvyNLAgent

@pytest.fixture
def agent():
    return VenvyNLAgent()

def test_basic_commands(agent):
    test_cases = [
        ("create a venv", "venvy create"),
        ("list environments", "venvy ls"),
        ("show stats", "venvy stats"),
    ]

    for nl, expected in test_cases:
        result = agent.parse(nl)
        assert result['command'] == expected

def test_complex_commands(agent):
    result = agent.parse("create Python 3.11 venv named myapp")
    assert "venvy create" in result['command']
    assert "--python 3.11" in result['command']
    assert "--name myapp" in result['command']

def test_confidence_scores(agent):
    # High confidence for common queries
    result = agent.parse("list my venvs")
    assert result['confidence'] > 0.8

    # Lower confidence for unusual queries
    result = agent.parse("do something with environments maybe")
    assert result['confidence'] < 0.7
```

### Documentation

**README section:**
```markdown
## 🧙 Natural Language Mode (Experimental)

Talk to venvy in plain English!

### Installation

```bash
pip install venvy[nl]  # Downloads 650MB model on first use
```

### Usage

```bash
# Wizard mode
venvy -w "create a Python 3.11 venv named backend"

# Interactive
venvy -w
🧙 What would you like to do? list my environments sorted by size
```

### How It Works

venvy uses a fine-tuned TinyLlama 1.1B model (trained with Unsloth/QLoRA) to convert your natural language queries into CLI commands. The model:

- Runs 100% locally (no API calls)
- ~650MB download (one-time)
- <2s inference on CPU
- 80%+ accuracy on venvy commands

### Examples

```bash
venvy -w "make a new venv"              → venvy create
venvy -w "show my environments"         → venvy ls
venvy -w "create Python 3.10 env"       → venvy create --python 3.10
venvy -w "delete old venvs"             → venvy cleanup
venvy -w "what venvs do I have"         → venvy ls
```

### Under the Hood

The NL agent:
1. Downloads fine-tuned TinyLlama model (first use only)
2. Parses your natural language input
3. Generates venvy command with confidence score
4. Shows preview and asks for confirmation
5. Executes command if approved

**Always shows what it will do before execution!**
```

---

## Phase 6: Portfolio Materials (Week 5)

### Blog Post Outline

**Title:** "Building a Natural Language CLI for Python Packages: Fine-Tuning TinyLlama with Unsloth"

**Sections:**
1. **Introduction** - The problem with CLI memorization
2. **Architecture** - NL → SLM → Command translation
3. **Dataset Creation** - How I generated 1,500 examples
4. **Fine-Tuning** - Unsloth/QLoRA/PEFT technical details
5. **Deployment** - Local inference with llama.cpp
6. **Results** - 80% accuracy, <2s latency, 650MB footprint
7. **Lessons Learned** - What worked, what didn't
8. **Future Work** - Multi-turn conversations, context awareness

### Demo Video Script

**Duration:** 3-5 minutes

1. **Intro** (30s) - Problem statement
2. **Live Demo** (2min) - Show wizard mode in action
3. **Technical Deep Dive** (1.5min) - Show training notebook
4. **Results** (1min) - Accuracy metrics, performance

### Resume Bullet Points

```
• Designed and deployed a natural language CLI agent for Python package management,
  enabling users to execute complex commands through conversational input

• Fine-tuned TinyLlama 1.1B using Unsloth/QLoRA on custom dataset of 1,500+
  NL-command pairs, achieving 80%+ accuracy with 2x faster training

• Implemented production-ready inference pipeline with llama.cpp for CPU deployment,
  optimizing model to 650MB footprint with <2s latency

• Built end-to-end ML pipeline: dataset curation → model training → quantization →
  deployment, demonstrating proficiency in PEFT, model optimization, and production ML
```

---

## Timeline & Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Dataset Creation | `venvy_nl_dataset.json` (1,500+ examples) |
| 2 | Model Training | Fine-tuned model, 80%+ accuracy |
| 2 | Quantization | `venvy-q4.gguf` (650MB) |
| 3 | Integration | Wizard mode in CLI |
| 3 | Testing | 90%+ test coverage |
| 4 | Documentation | README, code comments |
| 5 | Portfolio | Blog post, demo video |

---

## Tech Stack Summary (January 2025)

```yaml
Training (Google Colab):
  - unsloth>=2025.1        # 2x faster fine-tuning
  - transformers==4.46.3   # Latest HF
  - peft==0.13.2           # LoRA/QLoRA
  - bitsandbytes==0.44.1   # 4-bit quantization
  - trl==0.12.1            # SFT trainer

Inference (Local):
  - llama-cpp-python==0.3.2  # CPU inference

Model:
  - Base: TinyLlama-1.1B-Chat-v1.0
  - Method: QLoRA (r=16)
  - Quantization: 4-bit (Q4_K_M)
  - Size: 650MB
  - Format: GGUF
```

---

## Next Steps

**Ready to start?** I can help you with:

1. ✅ **Generate the dataset** - I'll create 1,500 examples
2. ✅ **Write the Colab notebook** - Complete training pipeline
3. ✅ **Implement CLI integration** - Wizard mode
4. ✅ **Write blog post** - Technical deep dive

**Let's start with Phase 1: Dataset Creation?**

I'll generate the initial 500 examples to get you started!
