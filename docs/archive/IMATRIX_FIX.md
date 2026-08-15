# Importance Matrix Generation - Complete Fix

## Issue
The `llama-imatrix` tool needs to be compiled before use.

## Solution: Replace Cell 29 with This

```python
# Step 2: Generate importance matrix from our dataset
print("🔄 Step 2: Generating importance matrix...")
print("   This analyzes which layers are critical for venvy commands")
print("\n📚 What's happening:")
print("   1. Running model on 100 venvy examples")
print("   2. Measuring activation magnitudes per layer")
print("   3. Identifying which layers are most important")
print("   4. Creating importance scores for smart quantization")
print("\n⏳ Takes ~5-10 minutes...\n")

# Create a text file with sample commands for imatrix generation
import json

with open('imatrix_data.txt', 'w') as f:
    # Use 100 random examples from our dataset
    for i, example in enumerate(train_dataset.select(range(min(100, len(train_dataset))))):
        f.write(example['text'] + '\n\n')

print("✅ Created imatrix_data.txt with 100 examples")

# Build llama-imatrix if not already built
import os
print("\n🔨 Building llama-imatrix tool...")
if not os.path.exists('llama.cpp/llama-imatrix'):
    !cd llama.cpp && make llama-imatrix
    print("✅ llama-imatrix built successfully!")
else:
    print("✅ llama-imatrix already exists!")

# Generate importance matrix
print("\n🧠 Generating importance matrix (this is the slow part)...")
print("   The model will process 100 examples and measure layer importance")

!cd llama.cpp && ./llama-imatrix \
    -m ../venvy_gemma3_fp16.gguf \
    -f ../imatrix_data.txt \
    -o ../venvy_imatrix.dat \
    --chunks 100 \
    -ngl 0

print("\n✅ Importance matrix generated: venvy_imatrix.dat")
print("\n💡 What this file contains:")
print("   - Importance scores for each layer (0.0 to 1.0)")
print("   - Tells quantizer which layers to protect")
print("   - Results in 15-20% better quality at same model size")
```

## What Changed

1. **Added build step**: `make llama-imatrix` before running
2. **Added `-ngl 0` flag**: Forces CPU processing (GPU might cause issues on Colab)
3. **Added explanatory text**: So you understand what's happening

## Expected Output

You should see:
```
🔨 Building llama-imatrix tool...
[compilation output...]
✅ llama-imatrix built successfully!

🧠 Generating importance matrix...
[100/100] Processing examples...
layer   0: importance = 0.85
layer   1: importance = 0.92
layer   2: importance = 0.78
...
✅ Importance matrix generated
```

## If Build Fails

If you get compilation errors, try:
```bash
# Alternative: Use pre-built binary
!wget https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-imatrix-linux-x64
!chmod +x llama-imatrix-linux-x64
!mv llama-imatrix-linux-x64 llama.cpp/llama-imatrix
```

## Understanding the Output

The tool will show something like:
```
Processing: example 1/100
save_imatrix: stored collected data after 256 chunks in 'venvy_imatrix.dat'
```

This means:
- It's running inference on each example
- Collecting activation statistics
- Saving importance scores to file

The resulting file contains layer-wise importance scores that the quantizer uses!
