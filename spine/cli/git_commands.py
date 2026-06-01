"""SPINE CLI — `gate` command group for git-gated transactional execution."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
def gate() -> None:
    """Git-gated transactional execution."""


@gate.command()
@click.argument("description")
@click.option(
    "--type",
    "work_type",
    type=click.Choice(["task", "critical_task"]),
    default="task",
    help="Workflow type to use.",
)
@click.option(
    "--config",
    "config_path",
    default="spine-gate.yaml",
    help="Path to the gate config file.",
)
def run(description: str, work_type: str, config_path: str) -> None:
    """Run a full git-gated transactional work item.

    DESCRIPTION is the work prompt for the agent. The patch is built in an
    isolated sandbox, validated against the gate pipeline, and (when
    auto-merge is enabled) ff-merged into the main branch.
    """
    from spine.git.orchestrator import SpineGitOrchestrator

    orchestrator = SpineGitOrchestrator(config_path=config_path)
    console.print(f"[bold blue]Running gated work:[/bold blue] {description[:100]}")
    console.print(f"[dim]Work type: {work_type}[/dim]")

    result = orchestrator.execute_transactional_run(description, work_type=work_type)

    status = result.get("status", "unknown")
    failed = status in ("failed", "error", "rolled_back")
    status_color = "red" if failed else ("yellow" if status == "validated_pending_merge" else "green")

    body = "\n".join(f"{key}: {value}" for key, value in result.items() if key != "status")
    console.print(
        Panel(
            f"Status: [{status_color}]{status}[/{status_color}]\n{body}".rstrip(),
            title="Gate Run Result",
        )
    )

    if failed:
        sys.exit(1)


@gate.command()
@click.option(
    "--config",
    "config_path",
    default="spine-gate.yaml",
    help="Path to the gate config file.",
)
def status(config_path: str) -> None:
    """Show the current gate sandbox status."""
    from spine.git.orchestrator import SpineGitOrchestrator

    orchestrator = SpineGitOrchestrator(config_path=config_path)
    info = orchestrator.status()

    active_color = "green" if info.get("active") else "dim"
    console.print(
        Panel(
            f"Active: [{active_color}]{info.get('active')}[/{active_color}]\n"
            f"Branch: {info.get('branch')}\n"
            f"Sandbox dir: {info.get('sandbox_dir')}\n"
            f"Strategy: {info.get('strategy')}",
            title="Gate Status",
        )
    )


@gate.command()
@click.option(
    "--config",
    "config_path",
    default="spine-gate.yaml",
    help="Path to the gate config file.",
)
def rollback(config_path: str) -> None:
    """Roll back (nuclear purge) the current gate sandbox."""
    from spine.git.orchestrator import SpineGitOrchestrator

    orchestrator = SpineGitOrchestrator(config_path=config_path)
    result = orchestrator.rollback_workspace()

    rolled_back = result.get("rolled_back")
    color = "green" if rolled_back else "yellow"
    console.print(
        Panel(
            f"Rolled back: [{color}]{rolled_back}[/{color}]",
            title="Gate Rollback",
        )
    )


@gate.command()
@click.option(
    "--config",
    "config_path",
    default="spine-gate.yaml",
    help="Path to the gate config file.",
)
def config(config_path: str) -> None:
    """Load and pretty-print the gate configuration."""
    import yaml

    from spine.git.orchestrator import load_gate_config

    gate_config = load_gate_config(config_path)
    rendered = yaml.safe_dump(
        gate_config, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    console.print(
        Panel(
            rendered.rstrip(),
            title=f"Gate Config ({config_path})",
        )
    )
