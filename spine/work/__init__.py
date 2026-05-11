"""Work submission and execution module."""

from .dispatcher import (
    record_work_item,
    run_workflow,
    submit_work,
    submit_work_from_config,
)

__all__ = [
    "record_work_item",
    "run_workflow",
    "submit_work",
    "submit_work_from_config",
]
