"""SPINE CLI — Click commands for run, status, resume, and list."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spine.config import SpineConfig

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="spine")
def main() -> None:
    """SPINE — Deterministic AI Agent Harness."""
    pass


@main.command()
@click.argument("description")
@click.option(
    "--type",
    "work_type",
    type=click.Choice(["quick", "critical_quick", "spec", "critical_spec"]),
    default="spec",
    help="Workflow type to use.",
)
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
@click.option(
    "--debug-llm",
    is_flag=True,
    default=False,
    help="Log all chat model messages (sent and received) to the console.",
)
def run(description: str, work_type: str, config_path: str, debug_llm: bool) -> None:
    """Submit a new work item and run the workflow.

    DESCRIPTION is the work prompt for the agent.
    """
    import os

    if debug_llm:
        os.environ["SPINE_DEBUG_LLM"] = "1"
        from spine.agents.debug_callback import install_global

        install_global()

    config = SpineConfig.load(path=config_path)
    console.print(f"[bold blue]Submitting work:[/bold blue] {description[:100]}")
    console.print(f"[dim]Work type: {work_type}[/dim]")

    from spine.work.dispatcher import submit_work

    result = asyncio.run(submit_work(description, work_type, config))

    if "error" in result:
        console.print(f"[bold red]Failed:[/bold red] {result['error']}")
        sys.exit(1)

    status_color = "green" if result["status"] == "completed" else "yellow"
    console.print(
        Panel(
            f"Work ID: {result['work_id']}\n"
            f"Status: [{status_color}]{result['status']}[/{status_color}]\n"
            f"Type: {result['work_type']}",
            title="Work Result",
        )
    )


@main.command(name="status")
@click.argument("work_id")
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
def status_cmd(work_id: str, config_path: str) -> None:
    """Get the status of a work item.

    WORK_ID is the unique identifier for the work item.
    """
    config = SpineConfig.load(path=config_path)

    from spine.work.dispatcher import get_work_status

    entry = get_work_status(work_id, config)
    if entry is None:
        console.print(f"[bold red]Work item '{work_id}' not found.[/bold red]")
        sys.exit(1)

    status_color = {
        "running": "blue",
        "completed": "green",
        "needs_review": "yellow",
        "failed": "red",
    }.get(entry.get("status", ""), "white")

    console.print(
        Panel(
            f"ID: {entry['id']}\n"
            f"Status: [{status_color}]{entry.get('status', 'unknown')}[/{status_color}]\n"
            f"Phase: {entry.get('current_phase', 'N/A')}\n"
            f"Type: {entry.get('work_type', 'N/A')}\n"
            f"Created: {entry.get('created_at', 'N/A')}\n"
            f"Updated: {entry.get('updated_at', 'N/A')}",
            title=f"Work: {work_id}",
        )
    )

    result = entry.get("result", {})
    if isinstance(result, dict) and result.get("artifacts"):
        console.print("\n[bold]Artifacts:[/bold]")
        for phase, names in result["artifacts"].items():
            console.print(f"  {phase}: {', '.join(names)}")


@main.command(name="list")
@click.option(
    "--status",
    "status_filter",
    type=click.Choice(["running", "completed", "needs_review", "failed"]),
    default=None,
    help="Filter by status.",
)
@click.option("--limit", default=20, help="Maximum items to show.")
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
def list_cmd(status_filter: str | None, limit: int, config_path: str) -> None:
    """List work items."""
    config = SpineConfig.load(path=config_path)

    from spine.work.dispatcher import list_work

    items = list_work(status=status_filter, limit=limit, config=config)

    if not items:
        console.print("[dim]No work items found.[/dim]")
        return

    table = Table(title="Work Items")
    table.add_column("ID", style="bold")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Phase")
    table.add_column("Created")

    status_colors = {
        "running": "blue",
        "completed": "green",
        "needs_review": "yellow",
        "failed": "red",
    }

    for item in items:
        color = status_colors.get(item.get("status", ""), "white")
        table.add_row(
            item.get("id", ""),
            item.get("work_type", ""),
            f"[{color}]{item.get('status', '')}[/{color}]",
            item.get("current_phase", ""),
            item.get("created_at", "")[:19],
        )

    console.print(table)


@main.command()
@click.argument("work_id")
@click.option("--input", "human_input", help="Human input for a prompt request.")
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
def resume(work_id: str, human_input: str | None, config_path: str) -> None:
    """Resume a paused work item (e.g. after human review).

    WORK_ID is the unique identifier for the paused work item.
    """
    config = SpineConfig.load(path=config_path)

    from spine.work.dispatcher import get_work_status

    entry = get_work_status(work_id, config)
    if entry is None:
        console.print(f"[bold red]Work item '{work_id}' not found.[/bold red]")
        sys.exit(1)

    if entry.get("status") != "needs_review":
        console.print(f"[yellow]Work item is not paused (status: {entry.get('status')}).[/yellow]")
        return

    feedback = human_input or "Approved — proceed with the workflow."
    action = "rework" if human_input else "approve"

    console.print(f"[blue]Resuming work item {work_id} ({action})...[/blue]")

    from spine.work.dispatcher import resume_work

    result = asyncio.run(resume_work(work_id, feedback, action, config))

    if "error" in result:
        console.print(f"[bold red]Resume failed:[/bold red] {result['error']}")
        sys.exit(1)

    status_color = "green" if result["status"] == "completed" else "yellow"
    console.print(
        Panel(
            f"Work ID: {result['work_id']}\n"
            f"Status: [{status_color}]{result['status']}[/{status_color}]\n"
            f"Action: {action}",
            title="Resume Result",
        )
    )


@main.command()
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
def worker(config_path: str) -> None:
    """Start the RalphLoopWorker background processor."""
    config = SpineConfig.load(path=config_path)

    from spine.work.ralph_worker import get_worker

    w = get_worker(config)
    w.start()
    console.print("[bold green]RalphLoopWorker started.[/bold green] Press Ctrl+C to stop.")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
        console.print("[yellow]Worker stopped.[/yellow]")


@main.command()
@click.option("--port", default=8501, help="Port to serve the UI on.")
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
@click.option(
    "--debug-llm",
    is_flag=True,
    default=False,
    help="Log all chat model messages (sent and received) to the console.",
)
def ui(port: int, config_path: str, debug_llm: bool) -> None:
    """Start the SPINE Streamlit dashboard."""
    import subprocess
    import sys

    import pathlib
    import spine
    from spine.config import SpineConfig

    SpineConfig.load(path=config_path)

    app_path = str(pathlib.Path(spine.__file__).parent / "ui" / "app.py")
    console.print(f"[bold blue]Starting SPINE UI on http://localhost:{port}[/bold blue]")
    if debug_llm:
        console.print("[dim]LLM debug logging: ON (chat messages will appear on console)[/dim]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    env = None
    if debug_llm:
        import os

        env = {**os.environ, "SPINE_DEBUG_LLM": "1"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(port),
            "--server.runOnSave",
            "false",
        ],
        env=env,
    )
    sys.exit(result.returncode)
