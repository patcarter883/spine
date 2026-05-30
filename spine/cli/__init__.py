"""CLI entrypoint for SPINE."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import spine
from spine.agents.debug_callback import DebugCallback
from spine.config import Config
from spine.persistence.vector_store import VectorStore
from spine.work.dispatcher import Dispatcher
from spine.work.ralph_worker import RalphWorker
from spine.workflow.export import export_work
from spine.workflow.workers.vector_indexer import VectorIndexer

__all__ = ["main"]

log = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure logging level based on verbose flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click
def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="SPINE CLI")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    
    configure_logging(verbose=args.verbose)
    
    log.debug("Starting SPINE CLI with verbose=%s", args.verbose)
    
    # Import and run the click group
    from spine.cli import cli
    cli()
    
    log.debug("Starting SPINE CLI with verbose=%s", args.verbose)
    
    log.debug("Starting SPINE CLI with verbose=%s", args.verbose)
    
    # Import and run the click group
    from spine.cli import cli
    cli()


if __name__ == "__main__":
    main()


@click.group()
def cli() -> None:
    """SPINE CLI commands."""
    pass


@cli.command()
def run(description: str, work_type: str, config_path: pathlib.Path = None, debug_llm: bool = False) -> None:
    """Run a work item."""
    log.debug("Running work item: %s", description)
    # ... rest of implementation