# evaluate_accuracy.py
from llama_cpp import Llama
import json

llm = Llama(model_path="assets/models/gemma3/base/venvy_gemma3_q4km.gguf", n_ctx=512, n_threads=4)

# Load validation set
with open('assets/data/venvy_training.jsonl') as f:
    examples = [json.loads(line) for line in f][-150:]  # Last 150 = validation

correct = 0
total = len(examples)

for ex in examples:
    query = ex['instruction'].replace('Translate to venvy command: ', '')
    expected_cmd = ex['output'].split('COMMAND: ')[1].split('\n')[0]
    
    # Get model prediction
    prompt = f"<start_of_turn>user\n{ex['instruction']}<end_of_turn>\n<start_of_turn>model\n"
    response = llm(prompt, max_tokens=128, temperature=0.1, stop=["<end_of_turn>"])
    predicted = response['choices'][0]['text'].strip()
    
    # Extract command
    if 'COMMAND:' in predicted:
        predicted_cmd = predicted.split('COMMAND: ')[1].split('\n')[0].strip()
    else:
        predicted_cmd = predicted.split('\n')[0].strip()
    
    # Check if correct
    if predicted_cmd == expected_cmd:
        correct += 1
    else:
        print(f"❌ Query: {query}")
        print(f"   Expected: {expected_cmd}")
        print(f"   Got: {predicted_cmd}\n")

accuracy = correct / total
print(f"\n📊 Accuracy: {correct}/{total} = {accuracy:.1%}")