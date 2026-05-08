"""SPINE Brownfields Discovery & Analysis Engine.

Provides tools to analyze existing ("brownfields") codebases and infer
their structure, architecture, requirements, and specifications.

Phases (feed into Ralph Loop hierarchy):
1. Mapper (mapper.py) — scans codebase for file tree, imports, dependencies
2. Analyzer (analyzer.py) — identifies architecture patterns, components
3. ReverseEngineer (reverse_engineer.py) — infers project specifications

All phases integrate with:
- spine/providers/llm.py (for AI-powered analysis)
- spine/core/persistence.py (for saving discovery results)
- spine/core/hierarchy.py (RalphLoopEngine for structured output)
"""

from .mapper import CodebaseMapper, CodeMap, CodebaseFileInfo, ImportInfo
from .analyzer import (
    CodebaseAnalyzer,
    ArchitecturePattern,
    Component,
    AnalysisResult,
)
from .reverse_engineer import (
    ReverseEngineer,
    InferredSpec,
    ApiEndpoint,
    DataSchema,
)

__all__ = [
    # Mapper
    "CodebaseMapper",
    "CodeMap",
    "CodebaseFileInfo",
    "ImportInfo",
    # Analyzer
    "CodebaseAnalyzer",
    "ArchitecturePattern",
    "Component",
    "AnalysisResult",
    # Reverse Engineer
    "ReverseEngineer",
    "InferredSpec",
    "ApiEndpoint",
    "DataSchema",
]
