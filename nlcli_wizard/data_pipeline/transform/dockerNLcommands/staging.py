import json
from pathlib import Path

# Run with: python -m nlcli_wizard.data_pipeline.transform.dockerNLcommands.staging

# Define the input and output paths relative to the project root
INPUT_FILE = "assets/data/base/dockerNLcommands/06102023.json"
OUTPUT_DIR = "assets/data/staging/dockerNLcommands" 


class BaseToStagingTransformer:
    """
    Transforms the base `DockerNL commands` JSON data into the staging JSONL format
    suitable for training, using the venvy command structure as a template.
    """

    def __init__(self, base_file_path: str, staging_dir: str):
        """
        Initializes the transformer with source file and target directory paths.

        Args:
            base_file_path: Full path to the input raw JSON file.
            staging_dir: Directory where the output JSONL file will be saved.
        """
        self.base_file_path = Path(base_file_path)
        self.staging_dir = Path(staging_dir)
        # Define the output file name based on the dataset
        self.output_file_name = f"stg_{self.base_file_path.stem}.jsonl"
        self.staging_file_path = self.staging_dir / self.output_file_name

    def load_data(self) -> list:
        """Loads the raw JSON data from the base layer."""
        
        abs_path = self.base_file_path.resolve()
        print(f"DEBUG: Attempting to load from absolute path: {abs_path}")
        
        try:
            with open(self.base_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} records.")
            return data
        except FileNotFoundError:
            print(f"ERROR: Input file not found at {self.base_file_path} (resolved: {abs_path}).")
            print("Ensure you are running the command from the project root directory (the parent of 'assets' and 'nlcli_wizard').")
            return []
        except json.JSONDecodeError as e:
            print(f"ERROR: Error decoding JSON file: {e}")
            return []

    def transform_record(self, record: dict) -> dict:
        """
        Transforms a single raw record into the target JSONL format.
        
        Input fields: 'input', 'instruction', 'output'
        Target fields: 'instruction', 'input', 'output' (with prefix)
        """
        
        # Combine the high-level instruction and the input query
        # Since the 'input' in the target format is always empty, we combine the 
        # original 'input' and 'instruction' into the new 'instruction'.
        if record.get("input"):
            # Example: "translate this sentence in docker command: Give me a list..."
            new_instruction = f"{record['instruction']}: {record['input']}"
        else:
            new_instruction = record['instruction']

        # Format the output field with the required COMMAND prefix
        new_output = f"COMMAND: {record['output']}"
        
        # Create the final structured object
        transformed_record = {
            "instruction": new_instruction,
            # The target format requires an empty 'input' field
            "input": "", 
            "output": new_output
            # CONFIDENCE and EXPLANATION are not present in the source data, 
            # so we omit them to match original source data structure.
        }
        
        return transformed_record

    def run(self):
        """Executes the full transformation pipeline: Load, Transform, Save."""
        
        print("-" * 40)
        print(f"Starting Base to Staging Transformation for {self.base_file_path.stem}")
        print("-" * 40)
        
        raw_data = self.load_data()
        if not raw_data:
            print("Transformation aborted due to loading error.")
            return

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Open the output file in write mode to start fresh
        print(f"Saving staged data to: {self.staging_file_path}")
        with open(self.staging_file_path, 'w', encoding='utf-8') as outfile:
            
            for record in raw_data:
                transformed_record = self.transform_record(record)
                
                # Write each transformed record as a single line of JSON (JSONL format)
                json.dump(transformed_record, outfile, ensure_ascii=False)
                outfile.write('\n') # Add newline to complete the JSONL format

        print("-" * 40)
        print("SUCCESS: Transformation pipeline finished.")
        print("-" * 40)

def execute_transformation():
    transformer = BaseToStagingTransformer(
        base_file_path=INPUT_FILE,
        staging_dir=OUTPUT_DIR
    )
    transformer.run()


if __name__ == "__main__":
    print(f"INFO: Current Working Directory is: {Path.cwd().resolve()}")
    execute_transformation()