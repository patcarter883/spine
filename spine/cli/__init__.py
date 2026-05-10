"""SPINE CLI - Command line interface."""

import os
import sys
import uuid
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from spine.core import SpineStateMachine
from spine.providers.base import ProviderConfig, ProviderFallbackChain, ConflictResolver, ConflictResult, DiscordNotifyProvider, SlackNotifyProvider, EmailNotifyProvider, Notification
from spine.providers.llm import OllamaProvider, OpenAIProvider, OpenRouterProvider, LocalOpenAIProvider
from spine import __version__
from spine.providers.agents import create_agent_provider


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
@click.option("--thread-id", "-t", default=None, help="Thread ID for persistence (auto-generated UUID if omitted)")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
@click.option("--debug-prompts", "-d", is_flag=True, help="Print prompts sent to agents to console")
def work(requirement: str, thread_id: str | None, checkpoint: str, config: str, debug_prompts: bool):
    """Start a new work item."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
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
    
    if debug_prompts:
        console.print("  [bold magenta]📝[bold] Debug prompts enabled — prompts will be printed below")
    
    machine = SpineStateMachine(
        checkpoint_path=checkpoint,
        llm_provider=llm_provider,
    )
    
    # Stash the debug flag so the executor can use it
    machine._debug_prompts = debug_prompts
    
    # Use streaming to show phase-by-phase progress
    seen_phases = set()
    final_state = None
    
    # Safely build providers dict: transform (name, provider, priority) tuples -> provider instances
    # Avoids IndexError when a provider list is empty or has only 1 element
    providers_dict = {
        k: v[0][1] if v else None
        for k, v in providers_by_type.items()
    }

    # Generate a work item ID from the requirement
    import re
    work_slug = re.sub(r'[^a-z0-9]+', '-', requirement.lower().strip())[:50].strip('-') or 'work'
    
    # Detect project context
    project_root = os.getcwd()
    project_name = Path(project_root).name
    
    # Try to load project context from config
    config_data = load_config(config)
    project_config = config_data.get("project", {})
    
    project_context = {
        "name": project_config.get("name", project_name),
        "root": project_config.get("root", project_root),
        "description": project_config.get("description", ""),
        "tech_stack": project_config.get("tech_stack", []),
    }
    
    try:
        stream = machine.app.stream(
            {"phase": "INIT", "requirement": requirement, "plan": None, "tasks": {},
             "completed_tasks": [], "failed_tasks": [], "swarm_state": {},
             "hive_cells": {}, "swarm_events": [],
             "variables": {"work_item_id": work_slug, "thread_id": thread_id, "debug_prompts": debug_prompts},
             "errors": [], "providers": providers_dict, "agent_provider": agent_provider,
             "critic_gate_result": None, "error_state": None, "error_history": [],
             "project_context": project_context},
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
@click.option("--thread-id", "-t", default=None, help="Thread ID to resume")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def resume(thread_id: str | None, checkpoint: str, config: str):
    """Resume a previous work item."""
    if thread_id is None:
        from spine.ui.utils import get_active_work_items
        items = get_active_work_items(checkpoint_path=checkpoint)
        if not items:
            console.print("[yellow]No active workflows found to resume.[/]")
            return
        thread_id = items[0]["thread_id"]
        console.print(f"[dim]Resuming most recent workflow: {thread_id}[/]")
    
    # Load providers from config
    providers_by_type = load_providers(config)
    llm_provider = get_primary_provider(providers_by_type, "llm")
    
    machine = SpineStateMachine(
        checkpoint_path=checkpoint,
        llm_provider=llm_provider,
    )
    state = machine.resume(thread_id)
    
    if state:
        console.print(f"[bold green]Resumed phase:[/] {state.get('phase', 'UNKNOWN')}")
        console.print(f"[bold green]Completed tasks:[/] {len(state.get('completed_tasks', []))}")
    else:
        console.print("[yellow]No state found to resume.[/]")


@cli.command()
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
def status(checkpoint: str):
    """Show current workflow status."""
    console.print("[bold]SPINE Status[/]")
    
    from .commands.status import get_threads
    from .renderers.status_renderer import render_threads
    
    threads = get_threads(checkpoint)
    render_threads(threads, console)


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Reinitialise completely without prompting.")
@click.option("--keep-config", "-k", is_flag=True, help="Reinitialise keeping existing config.yaml.")
def init(force: bool, keep_config: bool):
    """Initialize SPINE in the current directory."""
    spine_dir = Path(".spine")
    
    if spine_dir.exists():
        if force:
            console.print("[yellow].spine/ directory exists. Reinitialising completely (--force).[/]")
            import shutil
            shutil.rmtree(".spine")
        elif keep_config:
            console.print("[yellow].spine/ directory exists. Reinitialising keeping config (--keep-config).[/]")
            config_backup = Path(".spine/config.yaml")
            if config_backup.exists():
                config_data = config_backup.read_text()
            else:
                config_data = None
            import shutil
            shutil.rmtree(".spine")
        else:
            console.print("[red].spine/ directory already exists.[/]")
            choice = Prompt.ask(
                "Choose an option",
                choices=["reinit", "keep-config", "abort"],
                default="abort",
            )
            if choice == "abort":
                console.print("[dim]Aborted.[/]")
                return
            elif choice == "keep-config":
                if config_backup.exists():
                    config_data = config_backup.read_text()
                else:
                    config_data = None
                import shutil
                shutil.rmtree(".spine")
            else:
                import shutil
                shutil.rmtree(".spine")
    
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
    
    # Restore config if keep_config was used
    if 'config_data' in locals() and config_data is not None:
        with open(".spine/config.yaml", "w") as f:
            f.write(config_data)
        console.print("[bold green]SPINE reinitialised (config kept)![/]")
        console.print("Directories recreated, existing config preserved")
        return
    
    # Create default config
    config = """# SPINE Configuration
spine:
  checkpoint_path: .spine/spine.db

# Project context (optional - auto-detected if not specified)
project:
  name: spine  # Override project name (default: directory name)
  # description: "A deterministic AI-agent execution harness"  # Optional
  # tech_stack:  # Optional - helps agents understand the codebase
  #   - Python
  #   - LangGraph
  
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
    from spine.providers.base import PluginLoader
    
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
def hello():
    """Print a greeting with the current SPINE version."""
    from spine import __version__
    console.print(f"[bold green]Hello! Welcome to Spine v{__version__}[/]")


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