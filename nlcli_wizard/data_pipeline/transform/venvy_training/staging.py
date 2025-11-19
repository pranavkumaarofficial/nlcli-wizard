"""
Staging layer dataset generation for venvy CLI commands.

ALL COMMANDS VERIFIED against venvy/cli.py (lines 528-866).
ZERO FABRICATION - only commands that actually exist in venvy.

Based on VENVY_AUDIT_REPORT.md verification (October 24, 2025).
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import random


class VenvyDatasetGenerator:
    """
    Generate training dataset for venvy CLI.

    VERIFIED COMMANDS (from venvy/cli.py):
    1. venvy register [path] [--project PATH] [--name NAME]
    2. venvy ls [--sort {name,recent,size,project}] [--format {table,json,simple}]
    3. venvy scan [--home] [--path PATH] [--depth N]
    4. venvy current
    5. venvy cleanup [--days N] [--dry-run]
    6. venvy shell-hook [--shell {bash,zsh,fish,powershell}]
    7. venvy stats
    8. venvy track [path]  (internal command, rarely used directly)
    """

    def __init__(self):
        self.cli_tool = "venvy"

        # Target distribution based on expected usage patterns
        self.distribution = {
            'register': 0.25,   # 375 examples - common operation
            'ls': 0.20,         # 300 examples - frequently used
            'scan': 0.10,       # 150 examples - occasional use
            'current': 0.15,    # 225 examples - checking status
            'cleanup': 0.15,    # 225 examples - maintenance
            'shell-hook': 0.05, # 75 examples - one-time setup
            'stats': 0.10,      # 150 examples - informational
        }

    def generate_examples(self, num_examples: int = 1500) -> List[Dict[str, Any]]:
        """
        Generate training examples with ZERO fabrication.

        Args:
            num_examples: Total number of examples to generate

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

        # Calculate counts for each command type
        counts = {
            cmd: int(num_examples * prob)
            for cmd, prob in self.distribution.items()
        }

        # Generate examples for each command type
        examples.extend(self._register_examples(counts['register']))
        examples.extend(self._ls_examples(counts['ls']))
        examples.extend(self._scan_examples(counts['scan']))
        examples.extend(self._current_examples(counts['current']))
        examples.extend(self._cleanup_examples(counts['cleanup']))
        examples.extend(self._shell_hook_examples(counts['shell-hook']))
        examples.extend(self._stats_examples(counts['stats']))

        # Shuffle to mix command types
        random.shuffle(examples)

        return examples[:num_examples]

    def _register_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy register [path] [--project PATH] [--name NAME]

        Verified from venvy/cli.py lines 528-573
        """
        templates = [
            # Basic registration (no args - registers .venv in current dir)
            ("register this venv", "venvy register", 0.95,
             "Registers the .venv in current directory"),
            ("add this environment to registry", "venvy register", 0.92,
             "Registers the .venv in current directory"),
            ("track this venv", "venvy register", 0.90,
             "Registers the .venv in current directory"),
            ("register current environment", "venvy register", 0.93,
             "Registers the .venv in current directory"),

            # Register with explicit path
            ("register the venv at {path}", "venvy register {path}", 0.95,
             "Registers virtual environment at {path}"),
            ("add environment from {path}", "venvy register {path}", 0.92,
             "Registers virtual environment at {path}"),
            ("track venv in {path}", "venvy register {path}", 0.90,
             "Registers virtual environment at {path}"),

            # Register with custom name
            ("register this venv as {name}", "venvy register --name {name}", 0.95,
             "Registers .venv with custom name '{name}'"),
            ("add this environment named {name}", "venvy register --name {name}", 0.93,
             "Registers .venv with custom name '{name}'"),
            ("track this venv called {name}", "venvy register -n {name}", 0.91,
             "Registers .venv with custom name '{name}'"),

            # Register with project link
            ("register venv for project in {project}", "venvy register --project {project}", 0.95,
             "Registers .venv and links to project at {project}"),
            ("add venv linked to {project}", "venvy register --project {project}", 0.92,
             "Registers .venv and links to project at {project}"),
            ("register environment for {project} project", "venvy register -p {project}", 0.90,
             "Registers .venv and links to project at {project}"),

            # Complex combinations
            ("register {path} as {name}", "venvy register {path} --name {name}", 0.96,
             "Registers venv at {path} with name '{name}'"),
            ("register {path} for project {project}", "venvy register {path} --project {project}", 0.95,
             "Registers venv at {path} linked to {project}"),
        ]

        examples = []

        # Sample paths and names for variety
        paths = [
            ".venv", "venv", "./env", "../project/venv",
            "/home/user/envs/myenv", "C:/projects/app/venv",
            "./backend/.venv", "~/environments/test"
        ]
        names = [
            "myproject", "webapp", "api_env", "test-env",
            "production", "dev_environment", "ml_project", "backend_env"
        ]
        projects = [
            ".", "./myapp", "../project", "/home/user/projects/webapp",
            "~/code/backend", "C:/dev/myproject"
        ]

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            # Fill placeholders
            path = random.choice(paths)
            name = random.choice(names)
            project = random.choice(projects)

            instruction = instruction.format(path=path, name=name, project=project)
            command = command.format(path=path, name=name, project=project)
            explanation = explanation.format(path=path, name=name, project=project)

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),  # Add slight variation
                "explanation": explanation,
            })

        return examples

    def _ls_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy ls [--sort {name,recent,size,project}] [--format {table,json,simple}]

        Verified from venvy/cli.py lines 595-673
        """
        templates = [
            # Basic listing
            ("list all environments", "venvy ls", 0.95,
             "Lists all registered virtual environments"),
            ("show my venvs", "venvy ls", 0.93,
             "Lists all registered virtual environments"),
            ("what environments do i have", "venvy ls", 0.88,
             "Lists all registered virtual environments"),
            ("display all venvs", "venvy ls", 0.91,
             "Lists all registered virtual environments"),
            ("show registered environments", "venvy ls", 0.92,
             "Lists all registered virtual environments"),

            # Sorted listings
            ("list venvs by name", "venvy ls --sort name", 0.95,
             "Lists environments sorted alphabetically by name"),
            ("show environments sorted by recent use", "venvy ls --sort recent", 0.94,
             "Lists environments sorted by most recently used"),
            ("list venvs by size", "venvy ls --sort size", 0.95,
             "Lists environments sorted by disk space used"),
            ("show venvs sorted by project", "venvy ls --sort project", 0.93,
             "Lists environments sorted by linked project path"),
            ("sort environments by size", "venvy ls -s size", 0.94,
             "Lists environments sorted by disk space used"),

            # Different output formats
            ("list venvs in json", "venvy ls --format json", 0.96,
             "Lists environments in JSON format"),
            ("show environments as table", "venvy ls --format table", 0.95,
             "Lists environments in table format"),
            ("list venvs in simple format", "venvy ls --format simple", 0.93,
             "Lists environments in simple text format"),
            ("show venvs as json", "venvy ls -f json", 0.95,
             "Lists environments in JSON format"),

            # Combined options
            ("list venvs by size in json", "venvy ls --sort size --format json", 0.96,
             "Lists environments sorted by size in JSON format"),
            ("show recent venvs as table", "venvy ls --sort recent --format table", 0.95,
             "Lists environments sorted by recent use in table format"),
        ]

        examples = []

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def _scan_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy scan [--home] [--path PATH] [--depth N]

        Verified from venvy/cli.py lines 675-713
        """
        templates = [
            # Basic scan
            ("scan for venvs", "venvy scan", 0.93,
             "Scans current directory for virtual environments"),
            ("find virtual environments", "venvy scan", 0.91,
             "Scans current directory for virtual environments"),
            ("search for venvs in current directory", "venvy scan", 0.90,
             "Scans current directory for virtual environments"),

            # Scan home directory
            ("scan home directory for venvs", "venvy scan --home", 0.95,
             "Scans home directory for virtual environments"),
            ("find all venvs in home", "venvy scan --home", 0.93,
             "Scans home directory for virtual environments"),
            ("search home for environments", "venvy scan --home", 0.91,
             "Scans home directory for virtual environments"),

            # Scan specific path
            ("scan {path} for venvs", "venvy scan --path {path}", 0.95,
             "Scans {path} for virtual environments"),
            ("find venvs in {path}", "venvy scan --path {path}", 0.93,
             "Scans {path} for virtual environments"),
            ("search for environments in {path}", "venvy scan -p {path}", 0.91,
             "Scans {path} for virtual environments"),

            # Scan with depth limit
            ("scan with depth {depth}", "venvy scan --depth {depth}", 0.94,
             "Scans current directory to depth {depth} for venvs"),
            ("find venvs up to {depth} levels deep", "venvy scan --depth {depth}", 0.92,
             "Scans current directory to depth {depth} for venvs"),
            ("search {depth} directories deep", "venvy scan -d {depth}", 0.90,
             "Scans current directory to depth {depth} for venvs"),

            # Combined options
            ("scan {path} with depth {depth}", "venvy scan --path {path} --depth {depth}", 0.95,
             "Scans {path} to depth {depth} for virtual environments"),
        ]

        examples = []

        paths = [
            "/home/user/projects", "~/code", "C:/dev",
            "./projects", "../workspace", "/opt/apps"
        ]
        depths = [1, 2, 3, 4, 5]

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            path = random.choice(paths)
            depth = random.choice(depths)

            instruction = instruction.format(path=path, depth=depth)
            command = command.format(path=path, depth=depth)
            explanation = explanation.format(path=path, depth=depth)

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def _current_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy current

        Verified from venvy/cli.py lines 715-739
        """
        templates = [
            ("show current environment", "venvy current", 0.95,
             "Shows currently active virtual environment"),
            ("which venv am i using", "venvy current", 0.93,
             "Shows currently active virtual environment"),
            ("what environment is active", "venvy current", 0.92,
             "Shows currently active virtual environment"),
            ("show active venv", "venvy current", 0.94,
             "Shows currently active virtual environment"),
            ("display current environment", "venvy current", 0.91,
             "Shows currently active virtual environment"),
            ("which environment is activated", "venvy current", 0.90,
             "Shows currently active virtual environment"),
            ("what venv am i in", "venvy current", 0.89,
             "Shows currently active virtual environment"),
            ("show me the active environment", "venvy current", 0.92,
             "Shows currently active virtual environment"),
        ]

        examples = []

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def _cleanup_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy cleanup [--days N] [--dry-run]

        Verified from venvy/cli.py lines 741-801
        """
        templates = [
            # Basic cleanup (default 90 days)
            ("clean up old environments", "venvy cleanup", 0.95,
             "Removes virtual environments unused for 90 days"),
            ("remove unused venvs", "venvy cleanup", 0.93,
             "Removes virtual environments unused for 90 days"),
            ("delete old environments", "venvy cleanup", 0.91,
             "Removes virtual environments unused for 90 days"),
            ("cleanup stale venvs", "venvy cleanup", 0.92,
             "Removes virtual environments unused for 90 days"),

            # Cleanup with custom days
            ("cleanup venvs older than {days} days", "venvy cleanup --days {days}", 0.95,
             "Removes virtual environments unused for {days} days"),
            ("remove environments not used in {days} days", "venvy cleanup --days {days}", 0.93,
             "Removes virtual environments unused for {days} days"),
            ("delete venvs unused for {days} days", "venvy cleanup -d {days}", 0.91,
             "Removes virtual environments unused for {days} days"),

            # Dry run
            ("preview cleanup", "venvy cleanup --dry-run", 0.96,
             "Shows which environments would be removed without deleting"),
            ("show what would be cleaned up", "venvy cleanup --dry-run", 0.94,
             "Shows which environments would be removed without deleting"),
            ("dry run cleanup", "venvy cleanup --dry-run", 0.93,
             "Shows which environments would be removed without deleting"),

            # Combined options
            ("preview cleanup of {days} day old venvs", "venvy cleanup --days {days} --dry-run", 0.96,
             "Shows environments older than {days} days without deleting"),
        ]

        examples = []

        days_options = [7, 14, 30, 60, 90, 120, 180, 365]

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            days = random.choice(days_options)

            instruction = instruction.format(days=days)
            command = command.format(days=days)
            explanation = explanation.format(days=days)

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def _shell_hook_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy shell-hook [--shell {bash,zsh,fish,powershell}]

        Verified from venvy/cli.py lines 803-843
        """
        templates = [
            # Auto-detect shell
            ("setup shell integration", "venvy shell-hook", 0.95,
             "Generates shell integration hook (auto-detects shell)"),
            ("install shell hook", "venvy shell-hook", 0.93,
             "Generates shell integration hook (auto-detects shell)"),
            ("generate shell integration", "venvy shell-hook", 0.91,
             "Generates shell integration hook (auto-detects shell)"),

            # Specific shells
            ("setup bash integration", "venvy shell-hook --shell bash", 0.96,
             "Generates bash shell integration hook"),
            ("install zsh hook", "venvy shell-hook --shell zsh", 0.95,
             "Generates zsh shell integration hook"),
            ("generate fish integration", "venvy shell-hook --shell fish", 0.94,
             "Generates fish shell integration hook"),
            ("setup powershell hook", "venvy shell-hook --shell powershell", 0.95,
             "Generates PowerShell integration hook"),
        ]

        examples = []

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def _stats_examples(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate examples for: venvy stats

        Verified from venvy/cli.py lines 845-863
        """
        templates = [
            ("show statistics", "venvy stats", 0.95,
             "Shows statistics about registered virtual environments"),
            ("display venv stats", "venvy stats", 0.93,
             "Shows statistics about registered virtual environments"),
            ("show environment statistics", "venvy stats", 0.92,
             "Shows statistics about registered virtual environments"),
            ("what are my venv stats", "venvy stats", 0.90,
             "Shows statistics about registered virtual environments"),
            ("show me the stats", "venvy stats", 0.91,
             "Shows statistics about registered virtual environments"),
            ("display statistics", "venvy stats", 0.93,
             "Shows statistics about registered virtual environments"),
            ("how much space are venvs using", "venvy stats", 0.88,
             "Shows statistics about registered virtual environments"),
            ("show disk usage statistics", "venvy stats", 0.89,
             "Shows statistics about registered virtual environments"),
        ]

        examples = []

        for i in range(count):
            template = random.choice(templates)
            instruction, command, confidence, explanation = template

            examples.append({
                "instruction": instruction,
                "command": command,
                "confidence": confidence + random.uniform(-0.03, 0.02),
                "explanation": explanation,
            })

        return examples

    def save_to_jsonl(self, examples: List[Dict[str, Any]], output_path: Path):
        """
        Save examples to JSONL file in Alpaca format for fine-tuning.

        Args:
            examples: List of training examples
            output_path: Path to output JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in examples:
                # Convert to Alpaca-style training format
                training_example = {
                    "instruction": f"Translate to {self.cli_tool} command: {example['instruction']}",
                    "input": "",
                    "output": (
                        f"COMMAND: {example['command']}\n"
                        f"CONFIDENCE: {example['confidence']:.2f}\n"
                        f"EXPLANATION: {example['explanation']}\n"
                    ),
                }
                f.write(json.dumps(training_example) + '\n')

        print(f"Saved {len(examples)} examples to {output_path}")

    def validate_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Validate dataset for quality and correctness.

        Returns:
            Statistics about the dataset
        """
        examples = []

        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        # Extract commands from outputs
        commands = []
        for ex in examples:
            output = ex["output"]
            cmd_line = [line for line in output.split("\n") if line.startswith("COMMAND: ")][0]
            command = cmd_line.replace("COMMAND: ", "").strip()
            commands.append(command)

        # Count command distribution
        command_bases = [cmd.split()[1] if len(cmd.split()) > 1 else cmd for cmd in commands]
        from collections import Counter
        distribution = Counter(command_bases)

        # Calculate statistics
        total = len(examples)
        unique_commands = len(set(commands))
        avg_instruction_len = sum(len(ex["instruction"]) for ex in examples) / total
        avg_output_len = sum(len(ex["output"]) for ex in examples) / total

        return {
            "total_examples": total,
            "unique_commands": unique_commands,
            "command_distribution": dict(distribution),
            "avg_instruction_length": round(avg_instruction_len, 2),
            "avg_output_length": round(avg_output_len, 2),
        }


def main():
    """Generate training dataset for venvy."""
    generator = VenvyDatasetGenerator()

    print("Generating 1500 training examples for venvy...")
    print("All commands verified against venvy/cli.py")
    print()

    # Generate examples
    examples = generator.generate_examples(num_examples=1500)

    # Save to data directory
    output_path = Path(__file__).parent.parent / "assets" / "data" / "staging" / "virutal_environments" / "stg_venvy_training.jsonl"
    generator.save_to_jsonl(examples, output_path)

    # Validate and show statistics
    print()
    print("="*50)
    print("Dataset Statistics:")
    print("="*50)

    stats = generator.validate_dataset(output_path)

    print(f"  total_examples: {stats['total_examples']}")
    print(f"  unique_commands: {stats['unique_commands']}")
    print()
    print("command_distribution:")
    for cmd, count in sorted(stats['command_distribution'].items()):
        percentage = (count / stats['total_examples']) * 100
        print(f"  {cmd}: {count} examples ({percentage:.1f}%)")
    print()
    print(f"  avg_instruction_length: {stats['avg_instruction_length']} chars")
    print(f"  avg_output_length: {stats['avg_output_length']} chars")
    print("="*50)

    # Show sample examples
    print()
    print("Sample Examples (first 5):")
    print("-"*50)
    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            example = json.loads(line)
            print(f"\n{i+1}. Instruction: {example['instruction']}")
            print(f"   Output: {example['output'].strip()}")
    print("-"*50)


if __name__ == "__main__":
    main()
