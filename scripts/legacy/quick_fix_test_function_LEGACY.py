# CORRECTED TEST FUNCTION FOR COLAB NOTEBOOK
# Replace Cell 21 in train_gemma3_colab.ipynb with this code

# Enable inference mode (faster, less memory)
FastLanguageModel.for_inference(model)

def test_command_translation(nl_query):
    """
    Test the model's ability to translate natural language to venvy commands.
    """
    # Format as instruction
    instruction = f"Translate to venvy command: {nl_query}"

    # Format as Gemma chat turn - IMPORTANT: Match training format exactly
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate with proper parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,  # Low temperature for deterministic output
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Important: set pad token
        eos_token_id=tokenizer.eos_token_id,  # Important: set EOS token
    )

    # Decode WITHOUT skipping special tokens first (to see full output)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the model's response (after "<start_of_turn>model\n")
    if "<start_of_turn>model\n" in full_response:
        response = full_response.split("<start_of_turn>model\n")[-1]
        # Remove end token if present
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        response = response.strip()
    else:
        # Fallback: try without special tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract after "model" if present
        if "model" in response:
            response = response.split("model")[-1].strip()

    return response

print("✅ Inference mode enabled!")
print("\n🧪 Testing model on example queries...\n")
print("="*80)
