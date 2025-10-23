"""
NL-CLI Agent - Core translation logic from natural language to CLI commands
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import re

from nlcli_wizard.model import ModelManager


class NLCLIAgent:
    """
    Translates natural language instructions into CLI commands using a fine-tuned SLM.

    This is the core agent that:
    1. Accepts natural language input
    2. Loads and runs the appropriate SLM
    3. Generates CLI command suggestions
    4. Provides confidence scores and fallback options
    """

    def __init__(
        self,
        cli_tool: str,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.6,
        auto_execute: bool = False,
    ):
        """
        Initialize the NL-CLI agent.

        Args:
            cli_tool: Name of the CLI tool (e.g., "venvy", "pytest")
            model_path: Path to fine-tuned model (downloads if None)
            confidence_threshold: Minimum confidence to suggest command (0-1)
            auto_execute: Whether to auto-execute without preview (dangerous!)
        """
        self.cli_tool = cli_tool
        self.confidence_threshold = confidence_threshold
        self.auto_execute = auto_execute

        # Initialize model manager
        self.model_manager = ModelManager(
            cli_tool=cli_tool,
            model_path=model_path,
        )

    def translate(self, natural_language: str) -> Dict[str, Any]:
        """
        Translate natural language to CLI command.

        Args:
            natural_language: User's natural language instruction

        Returns:
            Dictionary with:
                - command: Generated CLI command string
                - confidence: Model confidence score (0-1)
                - explanation: Brief explanation of what command does
                - alternatives: List of alternative interpretations
                - success: Whether translation succeeded
        """
        # Normalize input
        nl_cleaned = self._normalize_input(natural_language)

        # Generate command using SLM
        result = self.model_manager.generate_command(nl_cleaned)

        # Parse model output
        parsed = self._parse_model_output(result)

        # Validate command
        is_valid = self._validate_command(parsed.get("command", ""))

        return {
            "command": parsed.get("command", ""),
            "confidence": parsed.get("confidence", 0.0),
            "explanation": parsed.get("explanation", ""),
            "alternatives": parsed.get("alternatives", []),
            "success": is_valid and parsed.get("confidence", 0.0) >= self.confidence_threshold,
        }

    def _normalize_input(self, text: str) -> str:
        """
        Normalize natural language input.

        - Strip whitespace
        - Lowercase (model was trained on lowercase)
        - Remove extra spaces
        """
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _parse_model_output(self, model_output: str) -> Dict[str, Any]:
        """
        Parse the SLM's output into structured format.

        Expected format:
        COMMAND: <cli command>
        CONFIDENCE: <0.0-1.0>
        EXPLANATION: <brief explanation>
        ALTERNATIVES: <alternative1> | <alternative2>
        """
        result = {
            "command": "",
            "confidence": 0.0,
            "explanation": "",
            "alternatives": [],
        }

        lines = model_output.strip().split('\n')

        for line in lines:
            if line.startswith("COMMAND:"):
                result["command"] = line.replace("COMMAND:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    result["confidence"] = float(conf_str)
                except ValueError:
                    result["confidence"] = 0.0
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("ALTERNATIVES:"):
                alts_str = line.replace("ALTERNATIVES:", "").strip()
                result["alternatives"] = [a.strip() for a in alts_str.split('|') if a.strip()]

        return result

    def _validate_command(self, command: str) -> bool:
        """
        Validate that generated command is safe and well-formed.

        Safety checks:
        - Starts with expected CLI tool name
        - No shell injection characters (;, &&, ||, >, <, `)
        - No suspicious patterns
        """
        if not command:
            return False

        # Must start with CLI tool name
        if not command.startswith(self.cli_tool):
            return False

        # Check for shell injection patterns
        dangerous_chars = [';', '&&', '||', '`', '$']
        for char in dangerous_chars:
            if char in command:
                return False

        # Check for redirection (could be legitimate, but risky)
        if '>' in command or '<' in command:
            return False

        return True

    def get_command_preview(self, natural_language: str) -> str:
        """
        Get a formatted preview of the command for user confirmation.

        Returns a rich-formatted string suitable for CLI display.
        """
        result = self.translate(natural_language)

        if not result["success"]:
            return f"[red]Could not translate:[/red] {natural_language}\n[yellow]Try rephrasing or use standard CLI syntax.[/yellow]"

        preview = f"""
[green]Natural Language:[/green] {natural_language}
[blue]Generated Command:[/blue] {result['command']}
[dim]Confidence: {result['confidence']:.0%}[/dim]
[dim]{result['explanation']}[/dim]
"""

        if result["alternatives"]:
            preview += f"\n[yellow]Alternatives:[/yellow]\n"
            for alt in result["alternatives"]:
                preview += f"  - {alt}\n"

        return preview.strip()


class CommandHistory:
    """
    Track history of NL → command translations for feedback loop.
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize command history tracker.

        Args:
            history_file: Path to JSON file storing history
        """
        self.history_file = history_file or Path.home() / ".nlcli_wizard" / "history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        """Load history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.history = []

    def save_translation(
        self,
        natural_language: str,
        generated_command: str,
        executed: bool,
        successful: bool,
    ):
        """
        Record a translation event.

        Args:
            natural_language: Original NL input
            generated_command: Generated CLI command
            executed: Whether user executed the command
            successful: Whether command succeeded (if executed)
        """
        entry = {
            "nl": natural_language,
            "command": generated_command,
            "executed": executed,
            "successful": successful,
            "timestamp": str(Path.ctime(Path(__file__))),
        }

        self.history.append(entry)

        # Save to disk
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except IOError:
            pass  # Silently fail if can't write

    def get_success_rate(self) -> float:
        """Calculate success rate of executed commands."""
        if not self.history:
            return 0.0

        executed = [h for h in self.history if h.get("executed", False)]
        if not executed:
            return 0.0

        successful = [h for h in executed if h.get("successful", False)]
        return len(successful) / len(executed)
