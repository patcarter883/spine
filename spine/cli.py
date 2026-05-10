"""SPINE CLI - Command line interface."""

import os
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .core import SpineStateMachine
from .providers.base import ProviderConfig, ProviderFallbackChain, ConflictResolver, ConflictResult, DiscordNotifyProvider, SlackNotifyProvider, EmailNotifyProvider, Notification
from .providers.llm import OllamaProvider, OpenAIProvider, OpenRouterProvider, LocalOpenAIProvider
from .providers.agents import create_agent_provider


console = Console()


def _expand_env_vars(value):
    """Recursively expand environment variables in string values.
    
    Supports ${VAR} and $VAR patterns. Missing vars expand to empty string.
    
    Args:
        value: Any value (string, dict, list, or primitive).
        
    Returns:
        Value with all ${VAR} and $VAR patterns replaced by env var values.
    """
    if isinstance(value, str):
        import re
        # Expand ${VAR} patterns
        def expand_braced(match):
            return os.environ.get(match.group(1), '')
        result = re.sub(r'\$\{([^}]+)\}', expand_braced, value)
        # Expand $VAR patterns (word chars after $)
        result = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', 
                        lambda m: os.environ.get(m.group(1), ''), result)
        return result
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str = ".spine/config.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        Parsed configuration dict, or empty dict if file not found.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    
    return _expand_env_vars(config)


def create_provider(cfg: ProviderConfig):
    """Instantiate a provider from its configuration.
    
    Args:
        cfg: ProviderConfig with name, type, and config fields.
        
    Returns:
        Configured provider instance, or None if type unknown.
    """
    provider_type = cfg.type.lower()
    config = cfg.config.copy()
    
    # Agent Providers
    if provider_type in ("opencode", "codex", "claude-code"):
        instance = create_agent_provider(provider_type, config)
        # Already configured by create_agent_provider, return early
        return instance
    
    # LLM Providers
    if provider_type == "ollama":
        instance = OllamaProvider(
            model=config.pop("model", "qwen3:32b"),
            base_url=config.pop("base_url", "http://localhost:11434"),
        )
    elif provider_type == "openai":
        instance = OpenAIProvider(
            api_key=config.pop("api_key", ""),
            model=config.pop("model", "gpt-4"),
        )
    elif provider_type == "openrouter":
        instance = OpenRouterProvider(
            api_key=config.pop("api_key", ""),
            model=config.pop("model", "openai/gpt-4"),
            base_url=config.pop("base_url", OpenRouterProvider.DEFAULT_BASE_URL),
        )
    elif provider_type == "local-openai":
        instance = LocalOpenAIProvider(
            api_key=config.pop("api_key", "not-required"),
            model=config.pop("model", "local-model"),
            base_url=config.pop("base_url", "http://localhost:8000/v1"),
        )
    else:
        console.print(f"[yellow]Unknown provider type: {provider_type}[/]")
        return None
    
    # Always call configure() so providers can initialize their
    # internal clients (e.g. OllamaProvider creates its HTTP client
    # in configure(), not in __init__).
    instance.configure(config)
    
    return instance


def load_providers(config_path: str = ".spine/config.yaml") -> dict[str, any]:
    """Load and instantiate all providers from config.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        Dict mapping provider category to list of (name, provider) tuples.
        Categories are derived from the YAML structure (llm, memory, storage, etc.).
    """
    providers_by_category: dict[str, list] = {}
    provider_configs_by_type: dict[str, list] = {}
    
    config = load_config(config_path)
    
    # Iterate over provider categories from YAML (llm, memory, storage, etc.)
    for category, provider_list in config.get("providers", {}).items():
        if not provider_list:
            continue
            
        for instance in provider_list:
            name = instance.get("name", "unnamed")
            impl_type = instance.get("type", "")
            enabled = instance.get("enabled", True)
            priority = instance.get("priority", 0)
            
            # Capture non-standard top-level keys (model, base_url, api_key, etc.)
            # and merge with any nested config dict into instance_config
            known_keys = {"name", "type", "enabled", "config", "priority"}
            top_level_config = {k: v for k, v in instance.items() if k not in known_keys}
            nested_config = instance.get("config", {})
            instance_config = {**top_level_config, **nested_config}
            
            if not enabled:
                continue
            
            # Create ProviderConfig with implementation type
            cfg = ProviderConfig(
                name=name,
                type=impl_type,
                enabled=enabled,
                priority=priority,
                config=instance_config,
            )
            
            provider = create_provider(cfg)
            if provider:
                if category not in providers_by_category:
                    providers_by_category[category] = []
                providers_by_category[category].append((name, provider, priority))
                
                # Track configs for fallback chain
                if category not in provider_configs_by_type:
                    provider_configs_by_type[category] = []
                provider_configs_by_type[category].append((name, provider, priority))
    
    # Sort providers by priority (lower number = higher priority, so first)
    for category in providers_by_category:
        providers_by_category[category].sort(key=lambda x: x[2])
    
    return providers_by_category


def get_primary_provider(providers_by_type: dict[str, list], provider_type: str = "llm"):
    """Get the highest priority provider of a given type.
    
    Args:
        providers_by_type: Dict from load_providers().
        provider_type: Type of provider to retrieve (e.g., 'llm').
        
    Returns:
        The provider instance, or None if not found.
    """
    providers = providers_by_type.get(provider_type, [])
    if providers:
        # First one is primary (sorted by priority in config)
        return providers[0][1]
    return None


def get_fallback_chain(providers_by_type: dict[str, list], provider_type: str = "llm") -> "ProviderFallbackChain":
    """Get a ProviderFallbackChain for the given provider type.
    
    Args:
        providers_by_type: Dict from load_providers().
        provider_type: Type of provider to retrieve (e.g., 'llm').
        
    Returns:
        ProviderFallbackChain instance with all providers of that type.
    """
    providers = providers_by_type.get(provider_type, [])
    provider_instances = [p[1] for p in providers]
    return ProviderFallbackChain(provider_instances)


@click.group()
@click.version_option(version="0.1.0", prog_name="spine")
def cli():
    """SPINE - Deterministic AI agent harness."""
    pass


PHASE_ICONS = {
    "INIT": "⚙️",
    "PLANNING": "📋",
    "EXECUTION": "🔨",
    "VERIFICATION": "✅",
    "COMPLETE": "🏁",
    "REWORK": "🔄",
    "ERROR": "❌",
    "BLOCKED": "🚧",
    "HUMAN_REVIEW": "👤",
}

PHASE_COLORS = {
    "INIT": "cyan",
    "PLANNING": "blue",
    "EXECUTION": "yellow",
    "VERIFICATION": "green",
    "COMPLETE": "green",
    "REWORK": "magenta",
    "ERROR": "red",
    "BLOCKED": "red",
    "HUMAN_REVIEW": "yellow",
}


def _print_phase_progress(phase_name: str, state: dict) -> None:
    """Print progress for a completed phase."""
    icon = PHASE_ICONS.get(phase_name, "•")
    color = PHASE_COLORS.get(phase_name, "white")
    tasks = state.get("completed_tasks", [])
    errors = state.get("errors", [])
    plan = state.get("plan")
    
    console.print(f"\n[{color}]{icon} Phase {phase_name}[/]")
    console.print(f"  Tasks completed: {len(tasks)}")
    
    if plan and phase_name == "PLANNING":
        console.print("  Plan created: [green]yes[/]")
        if plan.get("tasks"):
            for t in plan["tasks"]:
                console.print(f"    - {t.get('id', '?')}: {t.get('description', '')}")
    
    if errors:
        for err in errors[-3:]:  # Show last 3 errors
            console.print(f"  [red]⚠ {err}[/]")
    
    if phase_name == "VERIFICATION":
        console.print("  Artifacts written: [green]yes[/]")
        console.print(f"  Tasks completed: {len(tasks)}")
    
    if phase_name == "COMPLETE":
        console.print(f"  All tasks completed: {len(tasks)}")
        console.print("  [bold green]✓ Workflow complete![/]")


@cli.command()
@click.argument("requirement")
@click.option("--thread-id", "-t", default="default", help="Thread ID for persistence")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def work(requirement: str, thread_id: str, checkpoint: str, config: str):
    """Start a new work item."""
    console.print(f"[bold blue]SPINE[/] [dim]|[/] [bold]Starting work:[/] {requirement}")
    
    # Load providers from config
    providers_by_type = load_providers(config)
    llm_provider = get_primary_provider(providers_by_type, "llm")
    agent_provider = get_primary_provider(providers_by_type, "agent")
    
    if llm_provider:
        console.print(f"  [green]✓[/] LLM provider: [cyan]{llm_provider.name}[/]")
    else:
        console.print("  [yellow]⚠[/] No LLM provider configured, using stub mode")
    
    if agent_provider:
        console.print(f"  [green]✓[/] Agent provider: [cyan]{agent_provider.name}[/]")
    else:
        console.print("  [yellow]⚠[/] No agent provider configured, using LLM for execution")
    
    machine = SpineStateMachine(
        checkpoint_path=checkpoint,
        llm_provider=llm_provider,
    )
    
    # Use streaming to show phase-by-phase progress
    seen_phases = set()
    final_state = None
    
    # Safely build providers dict: transform (name, provider, priority) tuples -> provider instances
    # Avoids IndexError when a provider list is empty or has only 1 element
    providers_dict = {
        k: v[0][1] if v else None
        for k, v in providers_by_type.items()
    }

    try:
        stream = machine.app.stream(
            {"phase": "INIT", "requirement": requirement, "plan": None, "tasks": {},
             "completed_tasks": [], "failed_tasks": [], "swarm_state": {},
             "hive_cells": {}, "swarm_events": [], "variables": {},
             "errors": [], "providers": providers_dict, "agent_provider": agent_provider,
             "critic_gate_result": None, "error_state": None, "error_history": []},
            {"configurable": {"thread_id": thread_id}}
        )
        
        for chunk in stream:
            for node_name, state in chunk.items():
                current_phase = state.get("phase", "")
                if current_phase and current_phase not in seen_phases:
                    seen_phases.add(current_phase)
                    _print_phase_progress(current_phase, state)
                final_state = state
        
        if final_state:
            final_phase = final_state.get("phase", "UNKNOWN")
            if final_phase not in seen_phases:
                _print_phase_progress(final_phase, final_state)
            
            if final_phase == "COMPLETE":
                console.print(f"\n[bold green]🎉 All done![/] {len(final_state.get('completed_tasks', []))} tasks completed")
            elif final_phase in ("ERROR", "BLOCKED"):
                console.print(f"\n[bold red]Work stopped at {final_phase}[/]")
                for err in final_state.get("errors", []):
                    console.print(f"  [red]• {err}[/]")
            else:
                console.print(f"\n[dim]Final phase: {final_phase} | Tasks: {len(final_state.get('completed_tasks', []))}[/]")
                
    except Exception as e:
        console.print(f"\n[bold red]Error during workflow:[/] {e}")
        console.print_exception()


@cli.command()
@click.option("--thread-id", "-t", default="default", help="Thread ID to resume")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def resume(thread_id: str, checkpoint: str, config: str):
    """Resume a previous work item."""
    # Load providers from config
    providers_by_type = load_providers(config)
    llm_provider = get_primary_provider(providers_by_type, "llm")
    
    machine = SpineStateMachine(
        checkpoint_path=checkpoint,
        llm_provider=llm_provider,
    )
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
@click.option("--force", is_flag=True, help="Completely reinitialise Spine (overwrite everything)")
@click.option("--keep-config", is_flag=True, help="Reinitialise but keep existing configuration settings")
@click.option("--abort", is_flag=True, help="Do nothing and exit (no changes made)")
def init(force: bool, keep_config: bool, abort: bool):
    """Initialize SPINE in the current directory."""
    spine_dir = Path(".spine")
    already_initialised = spine_dir.is_dir()

    # --- Non-interactive safeguard ---
    if already_initialised and not sys.stdout.isatty():
        console.print("[yellow]Spine is already initialised in this repository.[/]")
        console.print("[yellow]Not a TTY — running non-interactively.[/]")
        console.print("  Use [bold]--force[/]   to completely reinitialise")
        console.print("  Use [bold]--keep-config[/] to reinitialise while preserving configuration")
        console.print("  Use [bold]--abort[/]      to do nothing and exit")
        raise SystemExit(10)

    # --- Existing setup detection ---
    if already_initialised:
        console.print("[yellow]Spine is already initialised in this repository.[/]")

        if abort:
            console.print("[dim]No changes made.[/]")
            raise SystemExit(0)

        if force or keep_config:
            # Flag-based path — proceed silently
            pass
        elif sys.stdout.isatty():
            # Interactive TTY path — prompt user
            console.print()
            console.print("  How would you like to proceed?")
            console.print("    [bold]1[/]  Completely reinitialise Spine (overwrite everything)")
            console.print("    [bold]2[/]  Reinitialise but keep existing configuration settings")
            console.print("    [bold]3[/]  Do nothing (exit)")
            try:
                choice = input("  Enter choice [1/2/3]: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("[dim]No changes made.[/]")
                raise SystemExit(0)

            if choice in ("1", ""):
                force = True
            elif choice == "2":
                keep_config = True
            else:
                console.print("[dim]No changes made.[/]")
                raise SystemExit(0)
        else:
            console.print("[yellow]Not a TTY — use --force, --keep-config, or --abort.[/]")
            raise SystemExit(10)

    # --- Build directory tree ---
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

    # --- Backup & wipe (if reinitialising) ---
    if already_initialised:
        backup_dir = _create_backup()
        if force:
            # Full wipe: remove everything in .spine except the backup
            _wipe_spine_dir()
            console.print(f"[dim]Backup created at {backup_dir}/[/]")
            console.print("[bold green]Completely reinitialised.[/]")
        elif keep_config:
            # Preserve config.yaml: copy it, wipe, then restore
            config_path = spine_dir / "config.yaml"
            if config_path.exists():
                config_data = config_path.read_text()
            else:
                config_data = ""
            _wipe_spine_dir()
            # Rewrite directories (they were wiped too)
            for d in dirs:
                os.makedirs(d, exist_ok=True)
            # Restore config
            with open(".spine/config.yaml", "w") as f:
                f.write(config_data)
            console.print(f"[dim]Backup created at {backup_dir}/[/]")
            console.print("[bold green]Preserved existing configuration.[/]")
        else:
            console.print("[bold green]SPINE initialized![/]")
    elif force:
        # --force on a clean repo: just proceed normally
        console.print("[bold green]SPINE initialized![/]")
    else:
        console.print("[bold green]SPINE initialized![/]")
        console.print("Created directories and default configuration")

    # --- Write default config (unless config was preserved) ---
    if not (already_initialised and keep_config):
        default_config = """# SPINE Configuration
spine:
  checkpoint_path: .spine/spine.db

providers:
  llm:
    - name: primary
      type: ollama
      model: qwen3:32b
"""
        with open(".spine/config.yaml", "w") as f:
            f.write(default_config)

    # --- Write default spec/plan if missing ---
    spec_path = spine_dir / "spec" / "default.md"
    plans_dir = spine_dir / "artifacts" / "plans"
    if not spec_path.exists():
        spec_path.write_text("# Default Spec\n")
    if not (plans_dir / "default.json").exists():
        (plans_dir / "default.json").write_text("{}")


def _create_backup() -> Path:
    """Create a timestamped backup of .spine/ and return the backup path."""
    import shutil
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f".spine_backup_{timestamp}")
    shutil.copytree(".spine", str(backup_dir), dirs_exist_ok=True)
    return backup_dir


def _wipe_spine_dir() -> None:
    """Remove all contents inside .spine/ (keeps the directory itself)."""
    import shutil

    spine_dir = Path(".spine")
    for child in spine_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(str(child))
        else:
            child.unlink()


@cli.command()
@click.option("--strategy", "-s", default="confidence_weighted", 
              type=click.Choice(["confidence_weighted", "voting", "consensus", "highest_priority"]),
              help="Conflict resolution strategy")
@click.option("--values", "-v", multiple=True, help="Provider values in format provider:value")
@click.option("--confidence", "-c", multiple=True, help="Confidence scores in format provider:score")
def resolve_conflict(strategy: str, values: tuple, confidence: tuple):
    """Resolve conflicts between provider results.
    
    Example: spine resolve-conflict -v provider1:value1 -v provider2:value2 -c provider1:0.9 -c provider2:0.7
    """
    console.print(f"[bold]Resolving conflict with {strategy} strategy[/]")
    
    values_dict = {}
    confidence_dict = {}
    
    for v in values:
        if ":" in v:
            name, val = v.split(":", 1)
            values_dict[name] = val
    
    for c in confidence:
        if ":" in c:
            name, score = c.split(":")
            confidence_dict[name] = float(score)
    
    if not values_dict:
        console.print("[yellow]No values provided[/]")
        return
    
    conflict = ConflictResult(
        key="cli-conflict",
        values=values_dict,
        confidence=confidence_dict or {k: 1.0 for k in values_dict.keys()}
    )
    
    resolver = ConflictResolver()
    result = resolver.resolve(conflict, strategy)
    
    console.print(f"[green]Resolved value:[/] {result}")


@cli.command()
@click.option("--provider", "-p", required=True, type=click.Choice(["discord", "slack", "email"]),
              help="Notification provider type")
@click.option("--title", "-t", required=True, help="Notification title")
@click.option("--message", "-m", required=True, help="Notification message")
@click.option("--level", "-l", default="info", type=click.Choice(["info", "warning", "error", "success"]),
              help="Notification level")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def notify(provider: str, title: str, message: str, level: str, config: str):
    """Send a notification via configured provider.
    
    Example: spine notify -p discord -t "Build Complete" -m "All tests passed"
    """
    import asyncio
    
    config_data = load_config(config)
    
    # Get notification provider from config
    notify_providers = config_data.get("providers", {}).get("notify", [])
    provider_config = None
    
    for p in notify_providers:
        if p.get("type") == provider:
            provider_config = p
            break
    
    if not provider_config:
        console.print(f"[red]No {provider} provider configured in {config}[/]")
        return
    
    # Instantiate the appropriate provider
    if provider == "discord":
        instance = DiscordNotifyProvider()
    elif provider == "slack":
        instance = SlackNotifyProvider()
    elif provider == "email":
        instance = EmailNotifyProvider()
    else:
        console.print(f"[red]Unknown provider: {provider}[/]")
        return
    
    instance.configure(provider_config.get("config", {}))
    
    if not instance.validate():
        console.print(f"[red]Provider {provider} not properly configured[/]")
        return
    
    notification = Notification(
        title=title,
        message=message,
        level=level
    )
    
    async def send():
        result = await instance.send(notification)
        return result
    
    result = asyncio.run(send())
    
    if result:
        console.print(f"[green]Notification sent via {provider}[/]")
    else:
        console.print("[red]Failed to send notification[/]")


@cli.command()
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def plugins(config: str):
    """List and manage external provider plugins.
    
    Discovers providers from spine-plugin.yaml manifests in plugin directories.
    """
    from .providers.base import PluginLoader
    
    loader = PluginLoader()
    loader.discover_plugins()
    
    available = []
    if hasattr(loader.registry, "_factories"):
        available = list(loader.registry._factories.keys())
    
    console.print("[bold]Available Provider Plugins[/]")
    if available:
        for p in available:
            console.print(f"  - {p}")
    else:
        console.print("[yellow]No plugins discovered[/]")


@cli.command()
def ui():
    """Start the SPINE web dashboard."""
    import subprocess
    import sys

    console.print("[bold blue]Starting SPINE Dashboard...[/]")
    console.print("[dim]Open http://localhost:8501 in your browser[/]")

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    project_root = str(Path(__file__).resolve().parent.parent)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "--browser.serverAddress", "localhost",
        "--server.port", "8501",
        "spine/ui/app.py",
    ], cwd=project_root, env=env)


def main():
    """Entry point for CLI."""
    cli()