# evaluate_docker.py - Docker model accuracy evaluation with per-category breakdown
from llama_cpp import Llama
import json
import time

llm = Llama(model_path="models/docker_gemma3_4b_q4km.gguf", n_ctx=512, n_threads=4, verbose=False)

# Load validation set (last ~100 examples)
with open('data/docker_training.jsonl') as f:
    examples = [json.loads(line) for line in f][-100:]

correct = 0
total = len(examples)
category_stats = {}
inference_times = []

for ex in examples:
    query = ex['instruction'].replace('Translate to docker command: ', '')
    expected_cmd = ex['output'].split('COMMAND: ')[1].split('\n')[0]

    # Determine category
    if expected_cmd.startswith('docker-compose'):
        category = 'compose'
    elif expected_cmd.startswith('docker run'):
        category = 'run'
    elif expected_cmd.startswith('docker build'):
        category = 'build'
    elif expected_cmd.startswith('docker exec'):
        category = 'exec'
    elif expected_cmd.startswith('docker network'):
        category = 'network'
    elif expected_cmd.startswith('docker volume'):
        category = 'volume'
    elif expected_cmd.startswith(('docker system', 'docker info', 'docker version',
                                  'docker container', 'docker image')):
        category = 'system'
    else:
        category = 'ps_images'

    if category not in category_stats:
        category_stats[category] = {'correct': 0, 'total': 0}
    category_stats[category]['total'] += 1

    # Get model prediction using Gemma 3 format
    prompt = f"<start_of_turn>user\n{ex['instruction']}<end_of_turn>\n<start_of_turn>model\n"
    start = time.time()
    response = llm(prompt, max_tokens=128, temperature=0.1, stop=["<end_of_turn>"])
    inference_times.append(time.time() - start)
    predicted = response['choices'][0]['text'].strip()

    # Extract command from structured output
    if 'COMMAND:' in predicted:
        predicted_cmd = predicted.split('COMMAND: ')[1].split('\n')[0].strip()
    else:
        predicted_cmd = predicted.split('\n')[0].strip()

    if predicted_cmd == expected_cmd:
        correct += 1
        category_stats[category]['correct'] += 1
    else:
        print(f"MISS | {category:10} | Query: {query}")
        print(f"     | Expected:  {expected_cmd}")
        print(f"     | Got:       {predicted_cmd}\n")

accuracy = correct / total
avg_time = sum(inference_times) / len(inference_times)

print(f"\n{'='*60}")
print(f"DOCKER MODEL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1%}")
print(f"Avg Inference:    {avg_time:.2f}s per query")
print(f"{'='*60}")
print(f"\nPer-category breakdown:")
print(f"{'Category':12} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
print(f"{'-'*38}")
for cat in sorted(category_stats.keys()):
    stats = category_stats[cat]
    cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{cat:12} {stats['correct']:>5}/{stats['total']:<3}    {cat_acc:>7.1%}")
