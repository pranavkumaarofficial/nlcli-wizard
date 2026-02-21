# Docker Training Guide - Colab Notebook Changes

## Quick Summary

**Only 3 changes needed** in your existing Colab notebook to train on docker dataset:

1. Change dataset filename
2. Update model output name
3. Adjust training parameters (optional - can keep same)

---

## Step-by-Step Changes

### 1. Dataset Loading (Cell ~10-12)

**CHANGE THIS:**
```python
# Old - venvy dataset
dataset = load_dataset('json', data_files='data/venvy_training.jsonl', split='train')
```

**TO THIS:**
```python
# New - docker dataset
dataset = load_dataset('json', data_files='data/docker_training.jsonl', split='train')
```

**Verify dataset size:**
```python
print(f"Dataset size: {len(dataset)}")
# Should print: Dataset size: 574
```

---

### 2. Model Output Names (Multiple cells)

**CHANGE ALL occurrences of `venvy_gemma3` TO `docker_gemma3`:**

#### LoRA Model Save (Cell ~18-20):
```python
# Old
model.save_pretrained("venvy_gemma3_lora")
tokenizer.save_pretrained("venvy_gemma3_lora")

# New
model.save_pretrained("docker_gemma3_lora")
tokenizer.save_pretrained("docker_gemma3_lora")
```

#### Merged Model Save (Cell ~22-25):
```python
# Old
model.save_pretrained_merged("venvy_gemma3_merged", tokenizer, save_method="merged_16bit")

# New
model.save_pretrained_merged("docker_gemma3_merged", tokenizer, save_method="merged_16bit")
```

#### GGUF Conversion (Cell ~28-30):
```python
# Old
model.save_pretrained_gguf("venvy_gemma3_gguf", tokenizer)

# New
model.save_pretrained_gguf("docker_gemma3_gguf", tokenizer)
```

#### Quantization Output (Cell ~35-40):
```python
# Old - in llama.cpp quantization command
!./llama.cpp/build/bin/llama-quantize \
    venvy_gemma3_gguf/gemma-3-1b-it-Q8_0.gguf \
    venvy_gemma3_q4km.gguf \
    Q4_K_M

# New
!./llama.cpp/build/bin/llama-quantize \
    docker_gemma3_gguf/gemma-3-1b-it-Q8_0.gguf \
    docker_gemma3_q4km.gguf \
    Q4_K_M
```

#### Final Model Copy to Repo (Cell ~45):
```python
# Old
!cp docker_gemma3_q4km.gguf /content/nlcli-wizard/models/venvy_gemma3_q4km.gguf

# New
!cp docker_gemma3_q4km.gguf /content/nlcli-wizard/models/docker_gemma3_q4km.gguf
```

---

### 3. Training Parameters (OPTIONAL - Current values work well)

**Current settings work great** (based on venvy 83.3% accuracy), but you can adjust:

```python
# Training arguments (Cell ~15-17)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,  # Keep same
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Keep same
        gradient_accumulation_steps=4,   # Keep same
        warmup_steps=5,                  # Keep same
        num_train_epochs=3,              # Keep same (or try 4 for docker complexity)
        learning_rate=2e-4,              # Keep same
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

**Recommendation**: Start with same parameters (3 epochs). If accuracy <80%, try 4 epochs.

---

### 4. Test/Inference Prompts (Cell ~42-45)

Update test examples to docker commands:

```python
# Old - venvy tests
test_queries = [
    "show me my virtual environments",
    "register this venv",
    "scan for venvs in current folder"
]

# New - docker tests
test_queries = [
    "run nginx on port 8080 in background",
    "build myapp version 2.0",
    "show all running containers",
    "start compose detached",
    "create network backend-net",
    "scale web to 3 instances"
]

for query in test_queries:
    prompt = f"""<start_of_turn>user
Translate to docker command: {query}<end_of_turn>
<start_of_turn>model
"""
    response = llm(prompt, max_tokens=128, temperature=0.1, stop=["<end_of_turn>"])
    print(f"Query: {query}")
    print(f"Command: {response['choices'][0]['text'].strip()}\n")
```

---

## Complete Cell-by-Cell Checklist

| Cell # | Description | Change Required |
|--------|-------------|-----------------|
| 1-5 | Setup, install dependencies | ❌ No change |
| 6-8 | Clone repo, load model | ❌ No change |
| 10-12 | **Load dataset** | ✅ Change to `docker_training.jsonl` |
| 13-14 | Format dataset | ❌ No change |
| 15-17 | **Training config** | ⚠️ Optional (keep same works) |
| 18-20 | **Save LoRA model** | ✅ Change to `docker_gemma3_lora` |
| 22-25 | **Save merged model** | ✅ Change to `docker_gemma3_merged` |
| 28-30 | **GGUF conversion** | ✅ Change to `docker_gemma3_gguf` |
| 32-34 | Build llama.cpp | ❌ No change |
| 35-40 | **Quantization** | ✅ Change input/output names |
| 42-45 | **Test inference** | ✅ Update test prompts |
| 46-48 | **Copy to repo** | ✅ Change to `docker_gemma3_q4km.gguf` |

---

## Quick Find-Replace (Ctrl+F in Colab)

**Find:** `venvy_gemma3`
**Replace:** `docker_gemma3`

**Find:** `venvy_training.jsonl`
**Replace:** `docker_training.jsonl`

**Find:** `Translate to venvy command:`
**Replace:** `Translate to docker command:`

---

## Expected Training Results

Based on venvy training (1,500 examples, 83.3% accuracy):

| Metric | Venvy (1,500 examples) | Docker (574 examples) | Notes |
|--------|------------------------|------------------------|-------|
| Training time | ~2.5 hours | **~1.5-2 hours** | Smaller dataset |
| Model size (Q4_K_M) | 810MB | **~810MB** | Same base model |
| Inference speed | ~1.5s | **~1.5s** | Same quantization |
| Expected accuracy | 83.3% | **80-85%** | Similar complexity |
| Epoch time | ~50min | **~30-35min** | Proportional to size |

---

## Training Timeline (Colab T4)

```
[00:00] Install dependencies (5 min)
[00:05] Load model + dataset (3 min)
[00:08] Training Epoch 1/3 (30 min)
[00:38] Training Epoch 2/3 (30 min)
[01:08] Training Epoch 3/3 (30 min)
[01:38] Save + merge models (5 min)
[01:43] GGUF conversion (3 min)
[01:46] Quantization to Q4_K_M (5 min)
[01:51] Testing inference (2 min)
[01:53] DONE ✓
```

**Total: ~1h 50min**

---

## After Training - Validation

Create `test/test_docker_model.py`:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/docker_gemma3_q4km.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=False,
)

test_cases = [
    # Basic run
    ("run nginx in background", "docker run -d nginx"),
    ("run redis on port 6379", "docker run -p 6379:6379 redis"),

    # Build
    ("build image tagged myapp", "docker build -t myapp ."),
    ("build without cache", "docker build --no-cache ."),

    # Compose
    ("start compose detached", "docker-compose up -d"),
    ("scale web to 3", "docker-compose up -d --scale web=3"),

    # Management
    ("show running containers", "docker ps"),
    ("list all images", "docker images"),
    ("clean up docker system", "docker system prune"),
]

correct = 0
for query, expected in test_cases:
    prompt = f"""<start_of_turn>user
Translate to docker command: {query}<end_of_turn>
<start_of_turn>model
"""
    response = llm(prompt, max_tokens=128, temperature=0.1, stop=["<end_of_turn>"])
    predicted = response['choices'][0]['text'].strip()

    is_correct = predicted == expected
    correct += is_correct

    print(f"✓" if is_correct else "✗", f"Query: {query}")
    print(f"  Expected:  {expected}")
    print(f"  Predicted: {predicted}\n")

print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
```

Run: `python test/test_docker_model.py`

---

## Git Commands (After Training)

```bash
# Add new files
git add models/docker_gemma3_q4km.gguf
git add data/docker_training.jsonl
git add data/DOCKER_DATASET_README.md

# Check file size (should use Git LFS for model)
git lfs track "*.gguf"

# Commit
git commit -m "Add Docker command translation support

- 574 verified docker command examples
- Trained Gemma 3 1B model (Q4_K_M quantized)
- Covers docker run/build/compose/network/volume/system
- Expected accuracy: 80-85%

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push
git push origin main
```

---

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size from 2 to 1
```python
per_device_train_batch_size=1,
gradient_accumulation_steps=8,  # Double this to compensate
```

### Issue: Training too slow
**Solution**: Use T4 GPU (not CPU), enable gradient checkpointing
```python
model = FastLanguageModel.get_peft_model(
    model,
    gradient_checkpointing="unsloth",  # Enable this
    ...
)
```

### Issue: Model not generating
**Solution**: Check prompt format matches training exactly
```python
# Must use Gemma 3 format
prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
```

---

## Summary

**Minimum changes**: 3 things
1. Dataset: `docker_training.jsonl`
2. Model names: `docker_gemma3_*`
3. Test prompts: docker commands

**Training time**: ~1h 50min (vs 2h 30min for venvy)

**Expected result**: 80-85% accuracy, 810MB model, ~1.5s inference

✅ Ready to train!
