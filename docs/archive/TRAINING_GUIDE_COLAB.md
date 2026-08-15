# Training Guide: Fine-tuning Gemma 3 1B on Google Colab

**Project**: nlcli-wizard
**Date**: October 24, 2025
**Target**: ML/AI engineers learning SLM fine-tuning

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Understanding the Tech Stack](#understanding-the-tech-stack)
4. [Training Pipeline](#training-pipeline)
5. [Deep Dive: Key Concepts](#deep-dive-key-concepts)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Overview

This guide walks you through fine-tuning **Gemma 3 1B** for natural language → CLI command translation using:

- **Unsloth** (2x faster training, 70% less VRAM)
- **QLoRA** (efficient fine-tuning with low-rank adapters)
- **4-bit Quantization** (compress model without accuracy loss)
- **GGUF Format** (CPU-optimized inference with llama.cpp)

### Training Goals

```
Input:  "list all environments sorted by size"
Output: "venvy ls --sort size"

Target Accuracy: 80-90%
Training Time:   8-10 minutes on Colab T4 GPU
Model Size:      ~600MB (quantized for CPU)
```

---

## Prerequisites

### Before You Start

1. **GitHub Repository**: Push your code to GitHub
   ```bash
   git add .
   git commit -m "Ready for training"
   git push origin main
   ```

2. **Google Colab Account**: Free tier with T4 GPU access
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Sign in with Google account

3. **GitHub Personal Access Token**: For cloning private repos
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create token with `repo` scope
   - Save it securely (you'll need it in Colab)

### Files You Need

Ensure these are in your repository:
- `data/venvy_training.jsonl` - Training dataset (1,500 examples)
- `training/train_gemma3_colab.ipynb` - Training notebook

---

## Understanding the Tech Stack

### 1. Why Gemma 3 1B?

**Gemma 3 1B** (March 2025) is Google's latest small language model:

| Feature | Benefit |
|---------|---------|
| **Phone-optimized** | Runs efficiently on consumer hardware |
| **1.1B parameters** | Small enough for CPU inference (~600MB quantized) |
| **Instruction-tuned** | Pre-trained for following instructions |
| **Chat format** | Supports multi-turn conversations |

**vs TinyLlama 1.1B** (2023):
- ✅ 20-30% better reasoning on benchmarks
- ✅ More stable training (better gradient flow)
- ✅ Optimized attention mechanism

### 2. What is Unsloth?

**Unsloth** is a library that makes LLM fine-tuning dramatically faster:

#### Traditional Fine-tuning:
```
Load Model (FP16)        → 2.2GB VRAM
Compute Gradients        → All 1.1B parameters
Update Weights           → Slow, memory-intensive
Result: ~20 minutes, 12GB VRAM needed
```

#### Unsloth + QLoRA:
```
Load Model (4-bit)       → 650MB VRAM
Add LoRA Adapters        → +16MB
Compute Gradients        → Only adapters (8M params)
Update Weights           → Fast, memory-efficient
Result: ~8 minutes, 4GB VRAM needed
```

#### How Unsloth Achieves 2x Speed:

1. **Custom CUDA Kernels**:
   ```python
   # Standard PyTorch (slow)
   out = torch.matmul(x, W) + torch.matmul(x, A @ B)

   # Unsloth (fast)
   out = unsloth.fused_lora_matmul(x, W, A, B)
   # ↑ Single kernel, 2x faster
   ```

2. **Dynamic 4-bit Quantization**:
   ```python
   for layer in model.layers:
       if layer.is_critical():  # Attention, embeddings, norms
           keep_fp16()  # Don't quantize
       else:
           quantize_4bit()  # Safe to compress
   ```

   Result: **10% more VRAM, 15-20% better accuracy**

3. **Gradient Checkpointing**:
   - Only store activations at checkpoint layers
   - Recompute intermediate activations during backward pass
   - Trades computation for memory (still net faster due to less swapping)

### 3. QLoRA (Quantized Low-Rank Adaptation)

**The Problem**: Traditional fine-tuning is expensive
- Must update all 1.1B parameters
- Requires storing gradients for all params (×3 memory)
- Easy to overfit on small datasets

**The Solution**: LoRA (Low-Rank Adaptation)

Instead of modifying original weights `W`, add small adapter matrices:

```
Original: y = Wx
LoRA: y = Wx + α(AB)x
         ↑     ↑
      frozen  trainable

Where:
W: [1024 × 1024] = 1,048,576 params (frozen)
A: [1024 × 16]   = 16,384 params (trainable)
B: [16 × 1024]   = 16,384 params (trainable)

Total trainable: 32,768 params (32x smaller!)
```

**Why Low-Rank Works**:
- Fine-tuning changes are typically low-rank (don't need full matrix)
- Most adaptation happens in a low-dimensional subspace
- Rank 16 captures 95%+ of necessary changes

**QLoRA = LoRA + 4-bit Quantization**:
```
1. Quantize base model W to 4-bit (650MB instead of 2.2GB)
2. Add FP16 LoRA adapters (16MB)
3. Train only the adapters
4. Total VRAM: 650 + 16 + overhead = ~1.5GB
```

### 4. 4-bit Quantization Explained

**Quantization** = Reducing numerical precision to save memory:

#### Full Precision (FP16):
```
Weight: 0.123456789
Binary: 0011110000111110... (16 bits)
Memory: 2 bytes per parameter
Total: 1.1B × 2 bytes = 2.2GB
```

#### 4-bit Quantization:
```
Weight: 0.123456789 → Quantized to bin 11 (0-15)
Binary: 1011 (4 bits)
Memory: 0.5 bytes per parameter
Total: 1.1B × 0.5 bytes = 550MB
```

#### NF4 (Normal Float 4-bit):

Standard 4-bit uses uniform bins:
```
Bins: [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, ...]
      ↑ Equal spacing
```

NF4 uses non-uniform bins optimized for normal distribution:
```
Bins: [-1.0, -0.67, -0.44, -0.29, -0.17, -0.08, 0, 0.08, 0.17, ...]
      ↑ More precision near 0 (where most weights are)
```

Neural network weights follow normal distribution → NF4 is optimal!

### 5. Q4_K_M with Importance Matrix

**Q4_K_M** = 4-bit quantization with K-means clustering:

#### Standard Q4:
```
All weights use same 16 bins:
[-1.0, -0.75, -0.5, ..., 0.75, 1.0]
```

#### Q4_K_M:
```
Layer 1 bins: [-0.9, -0.6, -0.3, ..., 0.7, 1.1]
Layer 2 bins: [-1.2, -0.8, -0.4, ..., 0.6, 0.9]
              ↑ Optimized per layer using K-means
```

**Importance Matrix (imatrix)**:

1. Run inference on your dataset
2. Measure activation magnitudes per layer
3. Identify critical vs non-critical layers
4. Quantize less important layers more aggressively

```python
if importance_score(layer) > threshold:
    use_6bit()  # Critical layer, preserve quality
else:
    use_4bit()  # Non-critical, compress more
```

Result: **Same model size, 15-20% better perplexity**

### 6. GGUF Format

**GGUF** (GPT-Generated Unified Format) is the successor to GGML:

#### Why GGUF?

| Feature | Benefit |
|---------|---------|
| **Memory-mapped** | Fast loading, no full load into RAM |
| **Quantization support** | 2-bit to 8-bit in single file |
| **Cross-platform** | CPU, GPU, Metal, Vulkan |
| **Efficient inference** | Optimized kernels in llama.cpp |

#### GGUF vs HuggingFace Format:

```
HuggingFace (safetensors):
├── model.safetensors (2.2GB FP16)
├── config.json
├── tokenizer.json
└── Requires PyTorch/transformers

GGUF:
├── model.gguf (600MB Q4_K_M)
└── Standalone, no dependencies
    Works with llama.cpp (C++)
```

---

## Training Pipeline

### Step-by-Step Process

```
┌─────────────────────────────────────────────────┐
│ 1. Setup: Install Unsloth + dependencies       │
│    Time: 3-5 minutes                            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 2. Clone Repo: Get dataset from GitHub         │
│    Time: 30 seconds                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 3. Load Model: Gemma 3 1B (4-bit quantized)    │
│    Download: 2.2GB                              │
│    Time: 2-3 minutes                            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 4. Add LoRA: Attach adapter matrices           │
│    Memory: +16MB                                │
│    Time: 5 seconds                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 5. Prepare Data: Format as Gemma chat turns    │
│    Split: 90% train, 10% validation            │
│    Time: 10 seconds                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 6. Train: Fine-tune on 1,350 examples          │
│    Epochs: 3                                    │
│    Time: 8-10 minutes ← MAIN TRAINING           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 7. Test: Verify accuracy on sample queries     │
│    Time: 30 seconds                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 8. Merge: Combine base + LoRA adapters         │
│    Output: FP16 model (2.2GB)                   │
│    Time: 1 minute                               │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 9. Quantize: Convert to GGUF Q4_K_M            │
│    Steps:                                       │
│    a) Convert to GGUF FP16                      │
│    b) Generate importance matrix                │
│    c) Quantize with imatrix                     │
│    Output: 600MB GGUF                           │
│    Time: 10-15 minutes                          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 10. Test GGUF: Verify CPU inference works      │
│     Time: 1 minute                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│ 11. Download: Save to local machine            │
│     Files: model.gguf (600MB)                   │
│     Time: 2-5 minutes                           │
└─────────────────────────────────────────────────┘

TOTAL TIME: ~30-40 minutes (mostly automated)
```

---

## Deep Dive: Key Concepts

### Training Hyperparameters

#### Learning Rate Schedule

```python
learning_rate = 2e-4  # Peak learning rate
warmup_steps = 50     # Gradual increase
lr_scheduler_type = "cosine"  # Gradual decrease

Visualization:
    |
2e-4|        _______________
    |      /                 \
    |    /                     \
    |  /                         \
  0 |_/____________________________\___
     0   50   100  ...  250   300  steps
     └─Warmup─┘  └─────Cosine──────┘
```

**Why warmup?**
- At start, model hasn't adapted to new task
- Large learning rate + random gradients = instability
- Gradually increasing LR allows model to "warm up"

**Why cosine decay?**
- At end, model is close to optimum
- Large LR causes oscillation around optimum
- Gradually decreasing LR allows precise convergence

#### Batch Size and Gradient Accumulation

```python
per_device_batch_size = 4      # Process 4 examples
gradient_accumulation_steps = 4  # Accumulate 4 batches
effective_batch_size = 4 × 4 = 16

How it works:
Step 1: Process batch 1 (4 examples) → Compute grads, don't update
Step 2: Process batch 2 (4 examples) → Add to grads, don't update
Step 3: Process batch 3 (4 examples) → Add to grads, don't update
Step 4: Process batch 4 (4 examples) → Add to grads, NOW UPDATE
```

**Why not just batch_size=16?**
- T4 GPU has 16GB VRAM
- Batch size 16 would require ~8GB (might cause OOM)
- Batch size 4 uses ~2GB (safe)
- Gradient accumulation gives us effective batch of 16 without memory issues

#### Weight Decay

```python
weight_decay = 0.01  # L2 regularization

Loss function:
L = CrossEntropy(predictions, targets) + 0.01 × Σ(w²)
                                          ↑
                               Penalty on large weights
```

**Why weight decay?**
- Prevents overfitting by penalizing complex models
- Encourages many small weights instead of few large weights
- Particularly important with small datasets (1,350 examples)

### Training Metrics

#### Loss (Cross-Entropy)

```python
# For each token, calculate:
loss = -log(P(correct_token))

Example:
Target: "venvy ls"
Model outputs probabilities:
  "venvy": 0.95 → -log(0.95) = 0.05
  " ls":   0.80 → -log(0.80) = 0.22
Average: 0.135

Lower is better!
Target: <0.5 for good performance
```

#### Perplexity

```python
perplexity = exp(loss)

If loss = 0.5:
  perplexity = exp(0.5) = 1.65

Interpretation:
  "On average, model is confused between 1.65 choices"
  Lower is better!
```

#### Learning Rate

Watch the learning rate change over time:
```
Steps 0-50:   0 → 2e-4    (warmup)
Steps 50-250: 2e-4 → 0    (cosine decay)
```

If you see constant LR, warmup/scheduler isn't working!

#### Gradient Norm

```python
grad_norm = sqrt(Σ(gradient²))
```

**What to watch:**
- Normal: 0.5 - 5.0 (stable training)
- Warning: 10-100 (might be learning too fast)
- Critical: >100 or NaN (training diverged!)

If grad_norm explodes:
1. Lower learning rate (try 1e-4)
2. Increase warmup steps (try 100)
3. Enable gradient clipping (max_grad_norm=1.0)

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solutions**:
```python
# A) Reduce batch size
per_device_batch_size = 2  # Instead of 4
gradient_accumulation_steps = 8  # Instead of 4

# B) Enable more gradient checkpointing
gradient_checkpointing = True

# C) Reduce max_seq_length
max_seq_length = 256  # Instead of 512
```

#### 2. Training Diverges (Loss → NaN)

**Symptoms**:
- Loss suddenly jumps to NaN
- Gradients become very large (>100)

**Solutions**:
```python
# A) Lower learning rate
learning_rate = 1e-4  # Instead of 2e-4

# B) Increase warmup
warmup_steps = 100  # Instead of 50

# C) Enable gradient clipping
max_grad_norm = 0.5  # Instead of 1.0
```

#### 3. Model Not Learning (Loss Stays High)

**Symptoms**:
- Loss stays above 2.0 after first epoch
- Validation loss not decreasing

**Solutions**:
```python
# A) Check dataset formatting
print(train_dataset[0]['text'])  # Should be properly formatted

# B) Increase learning rate
learning_rate = 3e-4  # Instead of 2e-4

# C) Train longer
num_train_epochs = 5  # Instead of 3
```

#### 4. Overfitting (Val Loss > Train Loss)

**Symptoms**:
- Train loss: 0.3, Val loss: 1.5
- Model memorizing training data

**Solutions**:
```python
# A) Increase weight decay
weight_decay = 0.05  # Instead of 0.01

# B) Add dropout
lora_dropout = 0.1  # Instead of 0

# C) Reduce LoRA rank
r = 8  # Instead of 16
```

#### 5. Slow Training

**Expected**: ~1-2 seconds per step on T4 GPU

**If slower**:
```python
# A) Check GPU is being used
!nvidia-smi  # Should show Python process using GPU

# B) Enable FP16
fp16 = True  # Should already be enabled

# C) Reduce logging
logging_steps = 50  # Instead of 10
```

### GGUF Conversion Issues

#### 1. Conversion Fails

**Error**: `KeyError: 'model.embed_tokens.weight'`

**Solution**: Model structure mismatch. Ensure you're using compatible llama.cpp version:
```bash
cd llama.cpp
git pull
make clean && make
```

#### 2. Quantized Model Gives Poor Results

**Symptoms**: GGUF model produces gibberish

**Solutions**:
```bash
# A) Try higher quantization level
Q5_K_M  # Instead of Q4_K_M

# B) Regenerate importance matrix with more data
--chunks 200  # Instead of 100

# C) Verify FP16 model works first
# Test before quantizing
```

---

## Next Steps

### After Training Completes

1. **Test Accuracy** on validation set
   ```python
   # Calculate accuracy on 150 validation examples
   correct = 0
   for example in eval_dataset:
       prediction = model.generate(example['input'])
       if prediction == example['output']:
           correct += 1
   accuracy = correct / len(eval_dataset)
   print(f"Accuracy: {accuracy:.2%}")
   ```

2. **Integrate with venvy CLI**
   - Copy `venvy_gemma3_q4km.gguf` to `nlcli-wizard/models/`
   - Update `nlcli_wizard/model.py` to load GGUF
   - Test: `venvy -w "list all environments"`

3. **Create Demo Video**
   - Show natural language → command translation
   - Highlight accuracy and speed
   - Perfect for portfolio/resume

4. **Write Blog Post**
   Topics:
   - Fine-tuning Gemma 3 1B with Unsloth
   - QLoRA for efficient SLM training
   - Quantization techniques (Q4_K_M with imatrix)
   - Building NL interfaces for CLI tools

5. **Experiment with Improvements**
   - Try different prompts (zero-shot, few-shot)
   - Test on other CLI tools (git, docker, kubectl)
   - Explore distillation (compress further)

### Portfolio Presentation

**For Recruiters/Hiring Managers**:

```markdown
## Natural Language CLI Agent (nlcli-wizard)

**Tech Stack**: Gemma 3 1B, Unsloth, QLoRA, GGUF Q4_K_M

**Problem**: CLI tools have complex syntax that users must memorize

**Solution**: Fine-tuned SLM to translate natural language → commands

**Key Achievements**:
- 80-90% accuracy on domain-specific commands
- ~600MB model (runs on CPU with <2s latency)
- Used Unsloth for 2x training speedup
- Applied Q4_K_M quantization with importance matrix

**Technical Skills Demonstrated**:
- SLM fine-tuning (QLoRA, PEFT)
- Model quantization (4-bit NF4, K-means, imatrix)
- Production deployment (CPU inference, GGUF format)
- Dataset creation (1,500 high-quality examples)

**Code**: github.com/your-username/nlcli-wizard
```

---

## Additional Resources

### Learn More

**Unsloth**:
- GitHub: https://github.com/unslothai/unsloth
- Blog: https://unsloth.ai/blog

**QLoRA Paper**:
- Title: "QLoRA: Efficient Finetuning of Quantized LLMs"
- Link: https://arxiv.org/abs/2305.14314

**llama.cpp**:
- GitHub: https://github.com/ggerganov/llama.cpp
- Quantization guide: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md

**Gemma 3**:
- Model card: https://huggingface.co/google/gemma-3-1b-it
- Paper: https://ai.google.dev/gemma

### Community

- **Unsloth Discord**: Community for questions/support
- **r/LocalLLaMA**: Reddit for local model deployment
- **HuggingFace Forums**: Technical discussions on fine-tuning

---

**Good luck with your training! 🚀**

*Questions? Issues? Check the troubleshooting section or open a GitHub issue.*
