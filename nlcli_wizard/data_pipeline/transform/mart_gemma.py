import json
from pathlib import Path
from typing import List, Dict, Any
import random

# Run with: python -m nlcli_wizard.data_pipeline.transform.mart_gemma
# This will produce the final training file at `assets/data/mart/gemma.jsonl`.
# The final file will be a JSONL file where each line is a single JSON object containing only the required `"text"` field for the `SFTTrainer`.
# This also shuffles the data

# Gemma 3 Chat Template Constants
# This template is used to format each Q&A pair into a single, cohesive chat turn 
# string required by the SFTTrainer's 'dataset_text_field'.
GEMMA_PROMPT = """<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

# Files to load from staging to convert to the final mart layer
INPUT_FILES = [
    "assets/data/staging/dockerNLcommands/stg_06102023.jsonl",
    "assets/data/staging/virtual_environments/stg_venvy_training.jsonl"
]
OUTPUT_DIR = "assets/data/mart"
OUTPUT_STEM = "gemma"

# Using a generic EOS token as the specific tokenizer is not available here.
# When this data is loaded for training, the tokenizer will replace this with its own EOS.
GENERIC_EOS_TOKEN = "<|end_of_text|>"


class StagingToMartTransformer:
    """
    Loads multiple staged JSONL files, merges them, and formats the records
    into the required Gemma 3 chat template for SFT training.
    """

    def __init__(self, input_paths: List[str], mart_dir: str, output_file_stem: str):
        """
        Initializes the transformer.

        Args:
            input_paths: List of full paths to the staged JSONL files.
            mart_dir: Directory where the final mart file will be saved.
            output_file_stem: The name stem for the output file (e.g., 'gemma').
        """
        self.input_paths = [Path(p) for p in input_paths]
        self.mart_dir = Path(mart_dir)
        self.mart_file_path = self.mart_dir / f"{output_file_stem}.jsonl"

    def load_and_merge_data(self) -> List[Dict[str, Any]]:
        """Loads and merges all input JSONL files vertically."""
        all_data = []
        for path in self.input_paths:
            print(f"Loading data from: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # Read line by line for JSONL format
                    data = [json.loads(line) for line in f]
                    all_data.extend(data)
                    print(f"Loaded {len(data)} records from {path.name}.")
            except FileNotFoundError:
                print(f"ERROR: Input file not found at {path.resolve()}. Skipping.")
            except json.JSONDecodeError as e:
                print(f"ERROR: Error decoding JSONL file {path.name}: {e}. Skipping.")
        
        print(f"\nSuccessfully merged a total of {len(all_data)} records.")
        return all_data

    @staticmethod
    def format_record_for_gemma(record: Dict[str, str]) -> Dict[str, str]:
        """
        Converts a single staged record (instruction, input, output) into 
        the Gemma 3 chat format string.
        """
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output = record.get("output", "")
        
        # Combine instruction and input (input is expected to be empty, but we handle it)
        full_instruction = instruction + ("\n" + input_text if input_text else "")
        
        # Format as chat turns
        text = GEMMA_PROMPT.format(full_instruction, output) + GENERIC_EOS_TOKEN
        
        # Return the record with the new "text" field
        return {"text": text}

    def run(self, seed: int = 42):
        """Executes the full Mart transformation pipeline."""
        
        print("-" * 40)
        print(f"Starting Mart Transformation: Gemma Format Generation")
        print("-" * 40)
        
        merged_data = self.load_and_merge_data()
        if not merged_data:
            print("Mart transformation aborted due to empty dataset.")
            return

        # 1. Shuffle the dataset (reproducibly) for ML training
        print(f"\nShuffling merged dataset with random seed = {seed} ...")
        random.Random(seed).shuffle(merged_data)
        print(f"Shuffling complete. Sample record after shuffle:")
        print(json.dumps(merged_data[0], indent=2, ensure_ascii=False))

        # 2. Transform all records
        formatted_data = [self.format_record_for_gemma(record) for record in merged_data]
        
        # 3. Save the mart layer file
        self.mart_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final mart data to: {self.mart_file_path}")
        with open(self.mart_file_path, 'w', encoding='utf-8') as outfile:
            for record in formatted_data:
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')
        print("-" * 40)
        print(f"SUCCESS: Mart pipeline finished. Total records: {len(formatted_data)}")
        print(f"Output saved to: {self.mart_file_path.resolve()}")
        print("-" * 40)
        
        print("\n📝 Formatted Example:")
        print("-" * 80)
        print(formatted_data[0]['text'])
        print("-" * 80)


def execute_mart_transformation():
    # Define the input files and output paths relative to the project root

    transformer = StagingToMartTransformer(
        input_paths=INPUT_FILES,
        mart_dir=OUTPUT_DIR,
        output_file_stem=OUTPUT_STEM
    )
    transformer.run()


if __name__ == "__main__":
    print(f"INFO: Current Working Directory is: {Path.cwd().resolve()}")
    execute_mart_transformation()