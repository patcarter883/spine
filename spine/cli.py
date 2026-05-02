"""SPINE CLI - Command line interface."""

import click
from rich.console import Console
from rich.table import Table

from .core import SpineStateMachine, SpineState
from .swarm import Supervisor ##, create_explorer_agent, create_sme_agent, create_planner_agent, create_critic_agent


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="spine")
def cli():
    """SPINE - Deterministic AI agent harness."""
    pass


@cli.command()
@click.argument("requirement")
@click.option("--thread-id", "-t", default="default", help="Thread ID for persistence")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
def work(requirement: str, thread_id: str, checkpoint: str):
    """Start a new work item."""
    console.print(f"[bold blue]Starting work:[/] {requirement}")
    
    machine = SpineStateMachine(checkpoint_path=checkpoint)
    result = machine.run(requirement, thread_id=thread_id)
    
    console.print(f"\n[bold green]Phase:[/] {result['phase']}")
    console.print(f"[bold green]Completed tasks:[/] {len(result['completed_tasks'])}")
    console.print(f"[bold green]Plan created:[/] {result['plan'] is not None}")


@cli.command()
@click.option("--thread-id", "-t", default="default", help="Thread ID to resume")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
def resume(thread_id: str, checkpoint: str):
    """Resume a previous work item."""
    machine = SpineStateMachine(checkpoint_path=checkpoint)
    state = machine.resume(thread_id)
    
    if state:
        console.print(f"[bold green]Resumed phase:[/] {state['phase']}")
        console.print(f"[bold green]Completed tasks:[/] {len(state.get('completed_tasks', []))}")
    else:
        console.print("[yellow]No state found to resume.[/]")


@cli.command()
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
def status(checkpoint: str):
    """Show current workflow status."""
    console.print("[bold]SPINE Status[/]")
    
    # In production, would query checkpoint database
    table = Table(title="Active Workflows")
    table.add_column("Thread ID", style="cyan")
    table.add_column("Phase", style="green")
    table.add_column("Tasks", style="yellow")
    
    table.add_row("default", "COMPLETE", "10")
    
    console.print(table)


@cli.command()
def init():
    """Initialize SPINE in the current directory."""
    import os
    
    dirs = [
        ".spine",
        ".spine/spec",
        ".spine/state",
        ".spine/state/checkpoints",
        ".spine/state/hive",
        ".spine/knowledge",
        ".spine/events",
        ".spine/artifacts",
        ".spine/artifacts/plans",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create default config
    config = """# SPINE Configuration
spine:
  checkpoint_path: .spine/spine.db
  
providers:
  llm:
    - name: primary
      type: ollama
      model: qwen3:32b
"""
    
    with open(".spine/config.yaml", "w") as f:
        f.write(config)
    
    console.print("[bold green]SPINE initialized![/]")
    console.print("Created directories and default configuration")


def main():
    """Entry point for CLI."""
    cli()