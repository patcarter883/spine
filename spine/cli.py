"""SPINE CLI - Command line interface."""

import os
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .core import SpineStateMachine, SpineState
from .swarm import Supervisor
from .providers.base import ProviderConfig, ProviderFallbackChain, ConflictResolver, ConflictResult, DiscordNotifyProvider, SlackNotifyProvider, EmailNotifyProvider, Notification
from .providers.llm import OllamaProvider, OpenAIProvider, OpenRouterProvider, LocalOpenAIProvider


console = Console()


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
    
    return config


def create_provider(cfg: ProviderConfig):
    """Instantiate a provider from its configuration.
    
    Args:
        cfg: ProviderConfig with name, type, and config fields.
        
    Returns:
        Configured provider instance, or None if type unknown.
    """
    provider_type = cfg.type.lower()
    config = cfg.config.copy()
    
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
    
    # Apply remaining config
    if config:
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
            instance_config = instance.get("config", {})
            priority = instance.get("priority", 0)
            
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


@cli.command()
@click.argument("requirement")
@click.option("--thread-id", "-t", default="default", help="Thread ID for persistence")
@click.option("--checkpoint", "-c", default=".spine/spine.db", help="Checkpoint database path")
@click.option("--config", "-f", default=".spine/config.yaml", help="Config file path")
def work(requirement: str, thread_id: str, checkpoint: str, config: str):
    """Start a new work item."""
    console.print(f"[bold blue]Starting work:[/] {requirement}")
    
    # Load providers from config
    providers_by_type = load_providers(config)
    llm_provider = get_primary_provider(providers_by_type, "llm")
    
    if llm_provider:
        console.print(f"[green]LLM provider:[/] {llm_provider.name}")
    else:
        console.print("[yellow]No LLM provider configured, using stub mode[/]")
    
    machine = SpineStateMachine(
        checkpoint_path=checkpoint,
        llm_provider=llm_provider,
    )
    result = machine.run(requirement, thread_id=thread_id)
    
    console.print(f"\n[bold green]Phase:[/] {result['phase']}")
    console.print(f"[bold green]Completed tasks:[/] {len(result['completed_tasks'])}")
    console.print(f"[bold green]Plan created:[/] {result['plan'] is not None}")


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
def init():
    """Initialize SPINE in the current directory."""
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
        console.print(f"[red]Failed to send notification[/]")


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


def main():
    """Entry point for CLI."""
    cli()