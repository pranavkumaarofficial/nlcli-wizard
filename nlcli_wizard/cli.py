"""
CLI interface for nlcli-wizard - mainly for testing and model management
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

from nlcli_wizard import __version__
from nlcli_wizard.agent import NLCLIAgent
from nlcli_wizard.model import ModelManager

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    nlcli-wizard - Natural Language interface for Python CLI tools
    """
    pass


@main.command()
@click.option(
    "--cli-tool",
    default="venvy",
    help="CLI tool to translate for (e.g., venvy, docker)",
)
@click.option(
    "--model-path",
    default=None,
    type=click.Path(exists=True),
    help="Path to GGUF model file (auto-detected if not specified)",
)
@click.argument("instruction", nargs=-1, required=True)
def translate(cli_tool: str, model_path: str, instruction: tuple):
    """
    Translate natural language to CLI command.

    Examples:
        nlcli-wizard translate list all environments
        nlcli-wizard translate --cli-tool docker show running containers
    """
    nl_instruction = " ".join(instruction)

    console.print(f"[dim]Translating for {cli_tool}...[/dim]\n")

    model_path_obj = Path(model_path) if model_path else None
    agent = NLCLIAgent(cli_tool=cli_tool, model_path=model_path_obj)

    try:
        result = agent.translate(nl_instruction)

        if result["success"]:
            console.print(f"[green]Input:[/green] {nl_instruction}")
            console.print(f"[blue]Command:[/blue] {result['command']}")
            console.print(f"[dim]Confidence: {result['confidence']:.0%}[/dim]")
            console.print(f"[dim]{result['explanation']}[/dim]")

            if result["alternatives"]:
                console.print("\n[yellow]Alternatives:[/yellow]")
                for alt in result["alternatives"]:
                    console.print(f"  - {alt}")
        else:
            console.print(f"[red]Could not translate:[/red] {nl_instruction}")
            console.print("[yellow]Try rephrasing or use standard CLI syntax.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@main.command()
@click.option(
    "--cli-tool",
    default="venvy",
    help="CLI tool model is for (default: venvy)",
)
def model_info(cli_tool: str):
    """
    Show information about the loaded model.
    """
    manager = ModelManager(cli_tool=cli_tool)
    info = manager.get_model_info()

    table = Table(title="Model Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", info["status"])
    table.add_row("CLI Tool", info.get("cli_tool", "N/A"))
    table.add_row("Path", str(info["path"]) if info["path"] else "Not downloaded")
    table.add_row("Size (MB)", str(info["size_mb"]))

    console.print(table)


@main.command()
def cache_dir():
    """
    Show the model cache directory.
    """
    cache_path = Path.home() / ".cache" / "nlcli-wizard" / "models"
    console.print(f"[cyan]Model cache directory:[/cyan] {cache_path}")

    if cache_path.exists():
        models = list(cache_path.glob("*.gguf"))
        if models:
            console.print(f"\n[green]Found {len(models)} model(s):[/green]")
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                console.print(f"  - {model.name} ({size_mb:.2f} MB)")
        else:
            console.print("[yellow]No models cached yet.[/yellow]")
    else:
        console.print("[yellow]Cache directory does not exist yet.[/yellow]")


@main.command()
def list_tools():
    """
    Show available CLI tools and their model status.
    """
    table = Table(title="Available CLI Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Model File", style="green")
    table.add_column("Status", style="yellow")

    for tool, info in ModelManager.MODEL_REGISTRY.items():
        manager = ModelManager(cli_tool=tool)
        local_path = manager._find_local_model()
        if local_path:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            status = f"Ready ({size_mb:.0f} MB)"
        else:
            status = "Not found"
        table.add_row(tool, info["filename"], status)

    console.print(table)


if __name__ == "__main__":
    main()
