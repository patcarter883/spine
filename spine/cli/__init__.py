"""SPINE CLI — Click commands for run, status, resume, and list."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spine.config import SpineConfig
from spine.log import configure_logging

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="spine")
@click.option("--verbose/-v", is_flag=True, help="Enable verbose (DEBUG) logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """SPINE — Deterministic AI Agent Harness."""
    configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
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
    default=".spine/config.yaml",
    help="Path to config file.",
)
@click.option(
    "--project",
    "project_id",
    default=None,
    help="Associate this work item with a project (membership back-reference; "
    "independent of plan_id). The project must already exist.",
)
@click.option(
    "--debug-llm",
    is_flag=True,
    default=False,
    help="Log all chat model messages (sent and received) to the console.",
)
def run(
    description: str,
    work_type: str,
    config_path: str,
    project_id: str | None,
    debug_llm: bool,
) -> None:
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
    if project_id:
        console.print(f"[dim]Project: {project_id}[/dim]")

    from spine.work.dispatcher import submit_work

    result = asyncio.run(
        submit_work(description, work_type, config, project_id=project_id)
    )

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


@main.command()
@click.option(
    "--workspace",
    "workspace_root",
    default=None,
    help="Workspace root to index (default: from config).",
)
@click.option("--config", "config_path", default=".spine/config.yaml", help="Path to config file.")
@click.option(
    "--wipe",
    is_flag=True,
    default=False,
    help="Drop all rows and the incremental ledger before indexing, forcing a "
    "full re-index. Use after a model swap or a change to the indexing logic.",
)
def index(workspace_root: str | None, config_path: str, wipe: bool) -> None:
    """Index the workspace into the vector store for RAG.

    Discovers source files (.py, .php, .ts, .tsx), extracts per-symbol
    byte slices via tree-sitter, summarizes with LLM, and stores
    embeddings for hybrid search.

    Indexing is incremental: only files whose content changed since the
    last run are re-processed, and files removed from the tree are pruned.
    Use ``--wipe`` to force a full rebuild (e.g. after an embedding-model
    swap or a change to how documents are built).
    """
    config = SpineConfig.load(path=config_path)

    if wipe:
        from spine.persistence.vector_store import VectorStore

        console.print("[bold yellow]Wiping existing vector rows...[/bold yellow]")
        store = VectorStore(config.checkpoint_path)
        store.ensure_schema()
        store.delete_all()
        store.close()

    console.print("[bold blue]Indexing workspace for RAG...[/bold blue]")

    from spine.workflow.workers.vector_indexer import run_indexing_job

    result = asyncio.run(run_indexing_job(workspace_root))

    console.print(
        Panel(
            f"Files: {result.get('files_total', 0)} "
            f"({result.get('files_changed', 0)} changed, "
            f"{result.get('files_skipped', 0)} unchanged, "
            f"{result.get('files_removed', 0)} removed)\n"
            f"Symbols indexed: {result.get('symbols_indexed', 0)}\n"
            f"Errors: {result.get('errors', 0)}",
            title="Indexing Result",
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
        from spine.ui.utils import normalize_artifacts

        console.print("\n[bold]Artifacts:[/bold]")
        for label, text in normalize_artifacts(result["artifacts"]):
            console.print(f"  {label}: {text}" if label else f"  {text}")


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
@click.argument("work_id")
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
@click.option(
    "--clear-artifacts",
    is_flag=True,
    default=False,
    help="Delete all on-disk artifacts so phases regenerate from scratch.",
)
def restart(work_id: str, config_path: str, clear_artifacts: bool) -> None:
    """Restart a running/stalled/paused work item from phase 0.

    WORK_ID is the unique identifier of the work item to restart.
    The item must be in \"running\", \"stalled\", or \"needs_review\" status.
    """
    import asyncio

    config = SpineConfig.load(path=config_path)

    from spine.work.dispatcher import restart_work

    console.print(f"[blue]Restarting work item {work_id}...[/blue]")

    result = asyncio.run(restart_work(work_id, config, clear_artifacts=clear_artifacts))

    if "error" in result:
        console.print(f"[bold red]Restart failed:[/bold red] {result['error']}")
        sys.exit(1)

    status_color = "green" if result["status"] == "completed" else "yellow"
    console.print(
        Panel(
            f"Work ID: {result['work_id']}\\n"
            f"Status: [{status_color}]{result['status']}[/{status_color}]\\n"
            f"Action: restarted",
            title="Restart Result",
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


@main.command()
@click.argument("work_id")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write export to a file instead of stdout.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format.",
)
@click.option(
    "--config",
    "config_path",
    default=".spine/config.yaml",
    help="Path to config file.",
)
def export(work_id: str, output: str | None, output_format: str, config_path: str) -> None:
    """Export work item data for external analysis.

    WORK_ID is the unique identifier for the work item.
    Outputs specification, plan, research data, and prompts.
    """
    config = SpineConfig.load(path=config_path)

    from spine.work.dispatcher import get_work_status

    entry = get_work_status(work_id, config)
    if entry is None:
        console.print(f"[bold red]Work item '{work_id}' not found.[/bold red]")
        sys.exit(1)

    from spine.workflow.export import export_work_item, format_export_markdown

    data = export_work_item(work_id, config)
    if "error" in data:
        console.print(f"[bold red]{data['error']}[/bold red]")
        sys.exit(1)

    if output_format == "json":
        import json

        out = json.dumps(data, indent=2, default=str)
    else:
        out = format_export_markdown(data)

    if output:
        from pathlib import Path

        Path(output).write_text(out, encoding="utf-8")
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(out)


@main.command()
@click.argument("path", default=".")
@click.option(
    "--tech-stack",
    "tech_stack",
    multiple=True,
    help="Technology tag for the config header (repeatable, e.g. --tech-stack python --tech-stack langgraph).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing .spine/config.yaml even if its content differs.",
)
def init(path: str, tech_stack: tuple[str, ...], force: bool) -> None:
    """Initialize SPINE in a project directory.

    Creates .spine/skills/, .spine/artifacts/, and a baseline .spine/config.yaml
    under PATH (default: current directory). Existing src/ and tests/ are left
    untouched. Re-running is idempotent; an existing config is preserved unless
    --force is passed.
    """
    from spine.work.onboarding.init import init_workspace

    managed, preserved = init_workspace(path, list(tech_stack), force=force)

    config_path = f"{path.rstrip('/')}/.spine/config.yaml"
    config = SpineConfig.load(path=config_path)
    config.ensure_dirs()

    table = Table(title="Scaffolded files", show_header=True, header_style="bold")
    table.add_column("Path")
    table.add_column("Status")
    for rel in managed:
        status = "[yellow]preserved[/yellow]" if rel in preserved else "[green]written[/green]"
        table.add_row(rel, status)
    console.print(table)

    if preserved:
        console.print(
            f"[yellow]Preserved {len(preserved)} existing file(s). "
            f"Re-run with --force to overwrite.[/yellow]"
        )

    next_steps = (
        f"[bold]Next steps[/bold]\n\n"
        f"  1. Edit [cyan]{config_path}[/cyan] — uncomment and fill in one entry under\n"
        f"     [cyan]providers.llm[/cyan] and [cyan]providers.embedding[/cyan].\n"
        f"  2. Run [cyan]spine index[/cyan] to build the RAG vector index.\n"
        f"  3. Run [cyan]spine run \"your task description\"[/cyan] to start a work item.\n"
    )
    console.print(Panel(next_steps, title="SPINE initialized", border_style="green"))


@main.group()
def project() -> None:
    """Manage project/milestone envelopes spanning many work items."""


@project.command(name="create")
@click.argument("project_id")
@click.option("--title", default=None, help="Project title (defaults to the id).")
@click.option(
    "--from-json",
    "from_json",
    default=None,
    help="Path to a JSON file with ProjectSpec fields (title, requirements, roadmap, ...).",
)
@click.option("--config", "config_path", default=".spine/config.yaml", help="Path to config file.")
def project_create(
    project_id: str, title: str | None, from_json: str | None, config_path: str
) -> None:
    """Create a new project. PROJECT_ID is the unique project slug."""
    from datetime import datetime
    from pathlib import Path

    from spine.models.types import ProjectSpec
    from spine.persistence.project_store import ProjectStore

    config = SpineConfig.load(path=config_path)
    store = ProjectStore(base_path=config.project_path)

    if store.load_project(project_id) is not None:
        console.print(f"[bold red]Project '{project_id}' already exists.[/bold red]")
        sys.exit(1)

    now = datetime.now().isoformat()
    if from_json:
        import json

        data = json.loads(Path(from_json).read_text(encoding="utf-8"))
        data["id"] = project_id
        data.setdefault("title", title or project_id)
        data.setdefault("created_at", now)
        data["updated_at"] = now
        spec = ProjectSpec.model_validate(data)
    else:
        spec = ProjectSpec(
            id=project_id,
            title=title or project_id,
            created_at=now,
            updated_at=now,
        )

    store.save_project(spec)
    console.print(
        Panel(
            f"Project: {spec.id}\nTitle: {spec.title}\n"
            f"Requirements: {len(spec.requirements)}\nMembers: {len(spec.member_work_ids)}",
            title="Project Created",
        )
    )


@project.command(name="add")
@click.argument("project_id")
@click.argument("work_ids", nargs=-1, required=True)
@click.option("--config", "config_path", default=".spine/config.yaml", help="Path to config file.")
def project_add(project_id: str, work_ids: tuple[str, ...], config_path: str) -> None:
    """Add one or more WORK_IDS to PROJECT_ID's membership."""
    from spine.persistence.project_store import ProjectStore

    config = SpineConfig.load(path=config_path)
    store = ProjectStore(base_path=config.project_path)
    try:
        spec = store.add_members(project_id, list(work_ids))
    except KeyError:
        console.print(f"[bold red]Project '{project_id}' not found.[/bold red]")
        sys.exit(1)

    console.print(
        f"[green]Project '{project_id}' now has {len(spec.member_work_ids)} member(s).[/green]"
    )


@project.command(name="show")
@click.argument("project_id")
@click.option("--config", "config_path", default=".spine/config.yaml", help="Path to config file.")
def project_show(project_id: str, config_path: str) -> None:
    """Show a project and its deterministic requirement-coverage rollup."""
    from spine.persistence.project_store import ProjectStore
    from spine.project.aggregator import aggregate_project_coverage

    config = SpineConfig.load(path=config_path)
    store = ProjectStore(base_path=config.project_path)
    spec = store.load_project(project_id)
    if spec is None:
        console.print(f"[bold red]Project '{project_id}' not found.[/bold red]")
        sys.exit(1)

    coverage = asyncio.run(aggregate_project_coverage(spec, config))
    summary = coverage["summary"]

    console.print(
        Panel(
            f"Title: {spec.title}\n"
            f"Members: {coverage['total_members']} "
            f"({coverage['members_with_state']} with state, "
            f"{coverage['verified_members']} verified)\n"
            f"Requirements: [green]{summary['satisfied']} satisfied[/green], "
            f"[yellow]{summary['partial']} partial[/yellow], "
            f"[red]{summary['unsatisfied']} unsatisfied[/red]",
            title=f"Project: {project_id}",
        )
    )

    if coverage["requirements"]:
        status_colors = {"satisfied": "green", "partial": "yellow", "unsatisfied": "red"}
        table = Table(title="Requirement Coverage")
        table.add_column("ID", style="bold")
        table.add_column("Requirement")
        table.add_column("Status")
        table.add_column("Verified / Covering")
        for r in coverage["requirements"]:
            color = status_colors.get(r["status"], "white")
            table.add_row(
                r["id"],
                r["text"][:60],
                f"[{color}]{r['status']}[/{color}]",
                f"{len(r['verified'])}/{len(r['covering'])}",
            )
        console.print(table)

    if coverage["phases"]:
        phase_colors = {"complete": "green", "in_progress": "yellow", "pending": "white"}
        lines = [
            f"[{phase_colors.get(p['status'], 'white')}]{p['status']}[/] — "
            f"{p['id']}: {p['title']}"
            for p in coverage["phases"]
        ]
        console.print(Panel("\n".join(lines), title="Roadmap Phases"))


@project.command(name="list")
@click.option("--config", "config_path", default=".spine/config.yaml", help="Path to config file.")
def project_list(config_path: str) -> None:
    """List all projects."""
    from spine.persistence.project_store import ProjectStore

    config = SpineConfig.load(path=config_path)
    store = ProjectStore(base_path=config.project_path)
    ids = store.list_projects()
    if not ids:
        console.print("[dim]No projects found.[/dim]")
        return

    table = Table(title="Projects")
    table.add_column("ID", style="bold")
    table.add_column("Title")
    table.add_column("Members")
    for pid in ids:
        spec = store.load_project(pid)
        if spec is None:
            continue
        table.add_row(spec.id, spec.title, str(len(spec.member_work_ids)))
    console.print(table)
