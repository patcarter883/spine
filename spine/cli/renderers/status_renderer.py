"""Status renderer - renders thread status information using Rich."""

from rich.console import Console
from rich.table import Table


def render_threads(threads: list[dict], console: Console) -> None:
    """Render thread status information to the console.

    Args:
        threads: List of thread dicts with keys: thread_id, phase,
                 completed_tasks, requirement, plan_exists.
        console: Rich Console instance.
    """
    if not threads:
        console.print("[yellow]No active workflows found.[/]")
        console.print("[dim]Start a new work item with: spine work <requirement>[/]")
        return

    table = Table(title="Active Workflows")
    table.add_column("Thread ID", style="cyan", no_wrap=True)
    table.add_column("Phase", style="green")
    table.add_column("Tasks", style="yellow", justify="right")
    table.add_column("Requirement")

    for t in threads:
        tid = t.get("thread_id", "?")
        phase = t.get("phase", "?")
        tasks = str(t.get("completed_tasks", 0))
        requirement = t.get("requirement", "") or ""

        table.add_row(tid, phase, tasks, requirement)

    console.print(table)
    console.print(f"[dim]Total: {len(threads)} thread(s)[/]")
