"""
Dataset generation utilities for training the NL-CLI model
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import random


class DatasetGenerator:
    """
    Generate training datasets for NL → CLI command translation.

    Creates diverse examples with variations in:
    - Phrasing (formal/casual)
    - Detail level (explicit/implicit)
    - Synonyms and paraphrasing
    """

    def __init__(self, cli_tool: str = "venvy"):
        """
        Initialize dataset generator.

        Args:
            cli_tool: Name of the CLI tool to generate data for
        """
        self.cli_tool = cli_tool

    def generate_examples(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate training examples.

        Returns:
            List of examples in format:
            {
                "instruction": "natural language instruction",
                "command": "cli command",
                "confidence": 0.95,
                "explanation": "brief explanation"
            }
        """
        examples = []

        # Generate examples based on CLI tool
        if self.cli_tool == "venvy":
            examples = self._generate_venvy_examples(num_examples)
        else:
            raise ValueError(f"No generator implemented for {self.cli_tool}")

        return examples

    def _generate_venvy_examples(self, num_examples: int) -> List[Dict[str, Any]]:
        """
        Generate examples for venvy CLI.

        Categories:
        - Creating venvs (30%)
        - Listing/searching venvs (20%)
        - Managing packages (20%)
        - Activating/using venvs (15%)
        - Cleanup/maintenance (15%)
        """
        examples = []

        # Distribution of example types
        create_count = int(num_examples * 0.30)
        list_count = int(num_examples * 0.20)
        packages_count = int(num_examples * 0.20)
        activate_count = int(num_examples * 0.15)
        cleanup_count = int(num_examples * 0.15)

        # Generate each category
        examples.extend(self._venvy_create_examples(create_count))
        examples.extend(self._venvy_list_examples(list_count))
        examples.extend(self._venvy_package_examples(packages_count))
        examples.extend(self._venvy_activate_examples(activate_count))
        examples.extend(self._venvy_cleanup_examples(cleanup_count))

        # Shuffle to mix categories
        random.shuffle(examples)

        return examples[:num_examples]

    def _venvy_create_examples(self, count: int) -> List[Dict[str, Any]]:
        """Generate examples for creating venvs."""
        templates = [
            # Basic creation
            ("create a new environment", "venvy create", 0.95),
            ("make a new venv", "venvy create", 0.90),
            ("set up a virtual environment", "venvy create", 0.85),

            # With name
            ("create an environment called {name}", "venvy create --name {name}", 0.95),
            ("make a venv named {name}", "venvy create --name {name}", 0.92),
            ("set up environment {name}", "venvy create --name {name}", 0.88),

            # With Python version
            ("create a python {version} environment", "venvy create --python {version}", 0.95),
            ("make a venv with python {version}", "venvy create --python {version}", 0.93),
            ("set up a {version} environment", "venvy create --python {version}", 0.85),

            # With name and version
            (
                "create a python {version} environment called {name}",
                "venvy create --python {version} --name {name}",
                0.98
            ),
            (
                "make {name} with python {version}",
                "venvy create --name {name} --python {version}",
                0.95
            ),
        ]

        examples = []
        names = ["myenv", "testenv", "devenv", "project_env", "app", "backend", "frontend"]
        versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]

        for i in range(count):
            template = random.choice(templates)
            instruction_template, command_template, confidence = template

            # Fill in placeholders
            name = random.choice(names)
            version = random.choice(versions)

            instruction = instruction_template.format(name=name, version=version)
            command = command_template.format(name=name, version=version)
            explanation = f"Creates a new virtual environment"

            if "{name}" in instruction_template:
                explanation += f" named '{name}'"
            if "{version}" in instruction_template:
                explanation += f" using Python {version}"

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence,
                "explanation": explanation.strip(),
            })

        return examples

    def _venvy_list_examples(self, count: int) -> List[Dict[str, Any]]:
        """Generate examples for listing venvs."""
        templates = [
            ("list all environments", "venvy ls", 0.95, "Lists all registered virtual environments"),
            ("show my venvs", "venvy ls", 0.92, "Shows all registered virtual environments"),
            ("what environments do i have", "venvy ls", 0.88, "Lists available virtual environments"),
            ("display all virtual environments", "venvy ls", 0.90, "Displays all registered venvs"),
        ]

        examples = []
        for i in range(count):
            template = random.choice(templates)
            examples.append({
                "instruction": template[0],
                "command": template[1],
                "confidence": template[2],
                "explanation": template[3],
            })

        return examples

    def _venvy_package_examples(self, count: int) -> List[Dict[str, Any]]:
        """Generate examples for package management."""
        # Note: This depends on venvy's package management features
        # Adjust based on actual venvy commands
        templates = [
            ("save requirements", "venvy req save", 0.95, "Saves current environment requirements"),
            ("export dependencies", "venvy req save", 0.90, "Exports environment dependencies"),
            ("freeze packages", "venvy req save", 0.85, "Freezes installed packages to file"),
        ]

        examples = []
        for i in range(count):
            template = random.choice(templates)
            examples.append({
                "instruction": template[0],
                "command": template[1],
                "confidence": template[2],
                "explanation": template[3],
            })

        return examples

    def _venvy_activate_examples(self, count: int) -> List[Dict[str, Any]]:
        """Generate examples for activation commands."""
        templates = [
            ("show current environment", "venvy current", 0.95, "Shows currently active environment"),
            ("which env am i using", "venvy current", 0.90, "Displays the active virtual environment"),
            ("what environment is active", "venvy current", 0.92, "Shows which venv is currently active"),
        ]

        examples = []
        for i in range(count):
            template = random.choice(templates)
            examples.append({
                "instruction": template[0],
                "command": template[1],
                "confidence": template[2],
                "explanation": template[3],
            })

        return examples

    def _venvy_cleanup_examples(self, count: int) -> List[Dict[str, Any]]:
        """Generate examples for cleanup/maintenance."""
        templates = [
            ("clean up old environments", "venvy cleanup", 0.95, "Removes old/unused virtual environments"),
            ("remove unused venvs", "venvy cleanup", 0.92, "Cleans up unused environments"),
            ("delete old environments", "venvy cleanup --days 30", 0.90, "Removes environments not used in 30 days"),
        ]

        examples = []
        for i in range(count):
            template = random.choice(templates)
            examples.append({
                "instruction": template[0],
                "command": template[1],
                "confidence": template[2],
                "explanation": template[3],
            })

        return examples

    def save_to_jsonl(self, examples: List[Dict[str, Any]], output_path: Path):
        """
        Save examples to JSONL file for training.

        Args:
            examples: List of training examples
            output_path: Path to output JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in examples:
                # Convert to training format (Alpaca-style)
                training_example = {
                    "instruction": f"Translate to {self.cli_tool} command: {example['instruction']}",
                    "input": "",
                    "output": (
                        f"COMMAND: {example['command']}\n"
                        f"CONFIDENCE: {example['confidence']}\n"
                        f"EXPLANATION: {example['explanation']}\n"
                    ),
                }
                f.write(json.dumps(training_example) + '\n')

        print(f"Saved {len(examples)} examples to {output_path}")

    def validate_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Validate a dataset file.

        Returns:
            Statistics about the dataset
        """
        examples = []

        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        # Calculate statistics
        total = len(examples)
        unique_commands = len(set(ex["output"].split("COMMAND: ")[1].split("\n")[0] for ex in examples))

        avg_instruction_len = sum(len(ex["instruction"]) for ex in examples) / total
        avg_output_len = sum(len(ex["output"]) for ex in examples) / total

        return {
            "total_examples": total,
            "unique_commands": unique_commands,
            "avg_instruction_length": round(avg_instruction_len, 2),
            "avg_output_length": round(avg_output_len, 2),
        }


def main():
    """Generate a sample dataset for venvy."""
    generator = DatasetGenerator(cli_tool="venvy")

    # Generate 1500 examples
    examples = generator.generate_examples(num_examples=1500)

    # Save to data directory
    output_path = Path(__file__).parent.parent / "data" / "training_examples.jsonl"
    generator.save_to_jsonl(examples, output_path)

    # Validate
    stats = generator.validate_dataset(output_path)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
