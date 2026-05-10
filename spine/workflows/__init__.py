"""SPINE workflow lifecycle engines.

Greenfields Workflow Engine provides orchestrated lifecycles
for new projects using the Ralph Loop hierarchy pattern:
- SDD (Spec-Driven Development): Full 6-phase lifecycle
- Quick Work: Streamlined 3-phase lifecycle
"""

from .engine import WorkflowEngine, WorkflowPhase, WorkflowContext, WorkflowResult
from .sdd import SDDWorkflow
from .quick_work import QuickWorkflow

__all__ = [
    "WorkflowEngine",
    "WorkflowPhase",
    "WorkflowContext",
    "WorkflowResult",
    "SDDWorkflow",
    "QuickWorkflow",
]