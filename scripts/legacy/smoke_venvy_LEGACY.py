from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/venvy_gemma3_q4km.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=False,
)

def translate(nl_query):
    """Translate natural language to venvy command."""
    prompt = f"""<start_of_turn>user
Translate to venvy command: {nl_query}<end_of_turn>
<start_of_turn>model
"""
    
    response = llm(prompt, max_tokens=128, temperature=0.1, stop=["<end_of_turn>"])
    return response['choices'][0]['text'].strip()

# Demo
queries = [
    "list all environments",
    "register this venv as myproject",
    "show current environment",
    "cleanup old venvs",
    "scan home directory for environments",
    "show statistics"
]

print("🤖 Gemma 3 1B - venvy Command Translator\n")
print("="*80)

for query in queries:
    result = translate(query)
    print(f"\n💬 Query: \"{query}\"")
    print(f"⚡ Output:")
    print(result)
    print("-"*80)