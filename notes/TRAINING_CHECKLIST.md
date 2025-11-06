# Training Checklist: Gemma 3 1B Fine-tuning

**Quick reference guide for training workflow**

---

## Pre-Training Checklist

### ☐ 1. Verify Dataset
```bash
# Check dataset exists and is valid
python -c "import json; f=open('data/venvy_training.jsonl'); print(f'✅ {len(f.readlines())} examples'); f.close()"
```
**Expected**: ✅ 1500 examples

### ☐ 2. Commit and Push to GitHub
```bash
git add .
git commit -m "Ready for training: dataset + notebook"
git push origin main
```

### ☐ 3. Get GitHub Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scope: `repo` (Full control of private repositories)
4. Copy token and save it (you'll need it in Colab)

### ☐ 4. Open Google Colab
1. Go to: https://colab.research.google.com
2. Click "Upload" → Select `training/train_gemma3_colab.ipynb`
3. Or: File → Upload notebook

### ☐ 5. Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU**
3. Save

---

## Training Workflow (In Colab)

### ☐ Step 1: Install Unsloth (Cell 1)
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
**Time**: 3-5 minutes
**Watch for**: ✅ Unsloth installed successfully

### ☐ Step 2: Verify GPU (Cell 2)
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
**Expected**: ✅ GPU Available: Tesla T4

### ☐ Step 3: Clone Repository (Cell 3)
```python
# Enter your GitHub token when prompted
GITHUB_TOKEN = getpass()
!git clone https://{GITHUB_TOKEN}@github.com/pranavkumar2004/nlcli-wizard.git
```
**Time**: 30 seconds
**Watch for**: ✅ Repository cloned successfully!

### ☐ Step 4: Verify Dataset (Cell 4)
```python
dataset_path = Path("data/venvy_training.jsonl")
```
**Expected**: ✅ Dataset loaded: 1500 examples

### ☐ Step 5: Load Gemma 3 1B (Cell 5)
```python
model, tokenizer = FastLanguageModel.from_pretrained(...)
```
**Time**: 2-3 minutes (downloading ~2.2GB)
**Watch for**: ✅ Model loaded successfully!
**Memory**: Should show ~0.65GB allocated

### ☐ Step 6: Add LoRA Adapters (Cell 6)
```python
model = FastLanguageModel.get_peft_model(model, r=16, ...)
```
**Expected**: Trainable %: ~0.70%
**Watch for**: ✅ LoRA adapters added!

### ☐ Step 7: Prepare Dataset (Cells 7-9)
```python
dataset = load_dataset('json', ...)
```
**Watch for**:
- ✅ Dataset loaded: 1500 examples
- ✅ Dataset split (1350 train, 150 val)
- ✅ Dataset formatted for Gemma 3 chat

### ☐ Step 8: Configure Training (Cell 10)
```python
training_args = TrainingArguments(...)
```
**Verify settings**:
- Epochs: 3
- Effective batch: 16
- Learning rate: 2e-4
- FP16: True

### ☐ Step 9: Train! (Cells 11-12)
```python
trainer.train()
```
**Time**: 8-10 minutes
**Watch metrics**:
- ✅ Loss should decrease (2.5 → 0.5)
- ✅ No NaN values
- ✅ Speed: ~1-2 sec/step

**Expected Final**:
- Train loss: ~0.4-0.6
- Val loss: ~0.5-0.7

### ☐ Step 10: Test Model (Cells 13-14)
```python
test_command_translation("list all environments")
```
**Watch for**: Correct venvy commands in responses

### ☐ Step 11: Save Models (Cells 15-16)
```python
# LoRA adapters
model.save_pretrained("venvy_gemma3_lora")

# Merged model
model.save_pretrained_merged("venvy_gemma3_merged", ...)
```
**Expected**:
- ✅ LoRA saved (~16MB)
- ✅ Merged saved (~2.2GB)

### ☐ Step 12: Convert to GGUF (Cells 17-20)
```python
# Install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make

# Convert to GGUF FP16
!python llama.cpp/convert_hf_to_gguf.py venvy_gemma3_merged ...

# Generate importance matrix
!cd llama.cpp && ./llama-imatrix ...

# Quantize to Q4_K_M
!cd llama.cpp && ./llama-quantize ... Q4_K_M --imatrix ...
```
**Time**: 10-15 minutes
**Expected**: venvy_gemma3_q4km.gguf (~600MB)

### ☐ Step 13: Test GGUF (Cells 21-22)
```python
from llama_cpp import Llama
llm = Llama(model_path="venvy_gemma3_q4km.gguf")
```
**Watch for**: Correct responses on test queries

### ☐ Step 14: Download Model (Cell 23)
```python
from google.colab import files
files.download('venvy_gemma3_q4km.gguf')
```
**Time**: 2-5 minutes (downloading 600MB)

---

## Post-Training Checklist

### ☐ 1. Save Model Locally
```bash
# On your machine
mkdir models/
mv ~/Downloads/venvy_gemma3_q4km.gguf models/
```

### ☐ 2. Test Locally (Optional)
```python
from llama_cpp import Llama
llm = Llama(model_path="models/venvy_gemma3_q4km.gguf")
response = llm("Translate to venvy command: list all environments")
print(response['choices'][0]['text'])
```

### ☐ 3. Upload to GitHub with Git LFS
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.gguf"
git add .gitattributes

# Add model
git add models/venvy_gemma3_q4km.gguf
git commit -m "Add trained Gemma 3 1B model (Q4_K_M)"
git push
```

**Alternative**: Upload to HuggingFace Hub (see notebook Cell 24)

### ☐ 4. Document Results
Create `TRAINING_RESULTS.md`:
```markdown
# Training Results

**Date**: YYYY-MM-DD
**Model**: Gemma 3 1B (Q4_K_M)

## Metrics
- Final train loss: X.XXX
- Final val loss: X.XXX
- Training time: XX minutes
- Model size: 600MB

## Sample Outputs
[Paste test examples here]

## Next Steps
- Integrate with venvy CLI
- Test accuracy on held-out examples
- Create demo video
```

---

## Troubleshooting Quick Reference

### ❌ Out of Memory
```python
# Reduce batch size
per_device_batch_size = 2  # Instead of 4
gradient_accumulation_steps = 8  # Instead of 4
```

### ❌ Loss → NaN
```python
# Lower learning rate
learning_rate = 1e-4  # Instead of 2e-4
warmup_steps = 100  # Instead of 50
```

### ❌ Model Not Learning
```python
# Check dataset formatting
print(train_dataset[0]['text'])

# Increase LR
learning_rate = 3e-4
```

### ❌ Slow Training
```bash
# Verify GPU
!nvidia-smi  # Should show Python using GPU

# Check FP16 is enabled
fp16 = True
```

---

## Success Criteria

### ✅ Training Successful If:
1. **Loss decreases smoothly**: 2.5 → 0.5 over 3 epochs
2. **No NaN values**: All metrics remain valid
3. **Val loss ≈ Train loss**: Not overfitting (diff <0.3)
4. **Test outputs correct**: >70% of test queries give valid commands
5. **GGUF model works**: CPU inference produces sensible outputs

### ✅ Model Ready for Production If:
1. **Accuracy >80%**: On validation set
2. **Inference <2s**: On CPU with llama.cpp
3. **Model size ~600MB**: Q4_K_M quantized
4. **No hallucination**: Only generates valid venvy commands

---

## Time Estimates

| Phase | Time |
|-------|------|
| Setup (install, clone) | 5 min |
| Load model | 3 min |
| Training | 10 min |
| GGUF conversion | 15 min |
| Download | 5 min |
| **Total** | **~40 min** |

---

## Files to Save

After training, you should have:

```
models/
└── venvy_gemma3_q4km.gguf          # 600MB - MOST IMPORTANT

training/
├── venvy_gemma3_lora/              # 16MB - LoRA adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── venvy_gemma3_merged/            # 2.2GB - Full FP16 (optional)
└── venvy_imatrix.dat               # Importance matrix (optional)

logs/
└── training_logs.txt               # Training metrics
```

**Minimum to keep**: `venvy_gemma3_q4km.gguf` (600MB)

---

## Next Steps After Training

1. ☐ Integrate GGUF model into `nlcli_wizard/model.py`
2. ☐ Test CLI: `venvy -w "list all environments"`
3. ☐ Calculate accuracy on validation set
4. ☐ Create demo video/GIF
5. ☐ Write blog post about training process
6. ☐ Update README with results
7. ☐ Add to portfolio/resume

---

**Ready to train? Open the Colab notebook and follow the cells!** 🚀

**Questions?** Check [TRAINING_GUIDE_COLAB.md](docs/TRAINING_GUIDE_COLAB.md) for detailed explanations.
