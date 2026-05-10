"""Codebase Analyzer — identifies architecture patterns, catalogs components,
and assesses code quality from a CodeMap.

This is the second phase of the Brownfields Discovery & Analysis Engine.
Takes a CodeMap from the Mapper and produces an AnalysisResult.
Integrates with LLM provider for advanced pattern recognition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .mapper import CodeMap, CodebaseFileInfo, ImportInfo

if TYPE_CHECKING:
    from ..providers.llm import LLMProvider
    from ..models.types import PhaseNode, ProjectNode
    from ..core.hierarchy import RalphLoopEngine


# ═══════════════════════════════════════════════════════════════════
# Architecture Patterns
# ═══════════════════════════════════════════════════════════════════


class ArchitecturePattern(str, Enum):
    """Recognized architecture patterns."""

    LAYERED = "layered"  # e.g., presentation/service/data layers
    MVC = "mvc"  # Model-View-Controller
    MVVM = "mvvm"  # Model-View-ViewModel
    MICROSERVICES = "microservices"
    MONOLITHIC = "monolithic"
    EVENT_DRIVEN = "event_driven"
    HEXAGONAL = "hexagonal"  # Ports & Adapters
    CQRS = "cqrs"  # Command Query Responsibility Segregation
    PIPELINE = "pipeline"  # ETL / data pipeline
    PLUGIN = "plugin_based"
    SERVERLESS = "serverless"
    PACKAGE = "package"  # Library / package
    SCRIPT = "script"  # Simple script(s) — no discernible architecture
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════
# Component
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Component:
    """A logical component identified in the codebase."""

    name: str
    path: str
    component_type: str  # "service", "model", "view", "controller", "handler", "util", etc.
    dependencies: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    file_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": self.component_type,
            "dependencies": self.dependencies,
            "responsibilities": self.responsibilities,
            "file_count": self.file_count,
            "metadata": self.metadata,
        }


# Component type detection by directory name
_DIRECTORY_HINTS: Dict[str, str] = {
    "models": "model",
    "model": "model",
    "views": "view",
    "view": "view",
    "templates": "view",
    "controllers": "controller",
    "controller": "controller",
    "handlers": "handler",
    "handler": "handler",
    "services": "service",
    "service": "service",
    "repositories": "repository",
    "repository": "repository",
    "dao": "repository",
    "routers": "router",
    "router": "router",
    "routes": "router",
    "api": "api",
    "middleware": "middleware",
    "utils": "util",
    "util": "util",
    "helpers": "util",
    "helper": "util",
    "config": "config",
    "settings": "config",
    "tests": "test",
    "test": "test",
    "schemas": "schema",
    "schema": "schema",
    "serializers": "serializer",
    "migrations": "migration",
    "migration": "migration",
    "core": "core",
    "providers": "provider",
    "provider": "provider",
    "swarm": "orchestrator",
    "hive": "orchestrator",
    "cli": "cli",
    "scripts": "script",
    "script": "script",
}

# Pattern keywords for architecture detection
_ARCH_HINTS: Dict[ArchitecturePattern, List[str]] = {
    ArchitecturePattern.MVC: [
        r"\bmodels?\b", r"\bviews?\b", r"\bcontrollers?\b",
    ],
    ArchitecturePattern.MICROSERVICES: [
        r"\bmicroservice", r"\bservice discovery", r"\bgrpc\b",
        r"\bconsul\b", r"\bkubernetes\b", r"\bdocker\b.*compose",
    ],
    ArchitecturePattern.EVENT_DRIVEN: [
        r"\bevent\b", r"\bmessage\b.*queue", r"\bkafka\b",
        r"\brabbitmq\b", r"\bpublisher\b", r"\bsubscriber\b",
        r"\bevent\b.*handler", r"\bevent\b.*bus",
    ],
    ArchitecturePattern.HEXAGONAL: [
        r"\bports?\b", r"\badapter", r"\bhexagonal\b",
        r"\bdomain\b.*service", r"\bapplication\b.*service",
    ],
    ArchitecturePattern.CQRS: [
        r"\bcommand\b", r"\bquery\b.*handler", r"\bcqrs\b",
        r"\bcommand\b.*handler", r"\bquery\b.*bus",
    ],
    ArchitecturePattern.PIPELINE: [
        r"\bpipeline\b", r"\betl\b", r"\btransform\b.*data",
        r"\bextract\b", r"\bload\b",
    ],
    ArchitecturePattern.SERVERLESS: [
        r"\blambda\b", r"\bserverless\b", r"\bcloud\s*function",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# AnalysisResult
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AnalysisResult:
    """Result of codebase analysis phase.

    Attributes:
        architecture_pattern: Detected architecture pattern.
        confidence: Confidence in the architecture detection (0.0-1.0).
        component_count: Number of cataloged components.
        components: Detailed component list.
        code_quality_score: Aggregate code quality score (0.0-1.0).
        patterns: Recognized code patterns (design patterns, idioms).
        metrics: Additional metrics (complexity estimates, etc.).
        recommendations: Suggested improvements.
        raw_notes: Human-readable analysis notes.
    """

    architecture_pattern: str = "unknown"
    confidence: float = 0.0
    component_count: int = 0
    components: List[Component] = field(default_factory=list)
    code_quality_score: float = 0.5
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    raw_notes: str = ""

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Architecture: {self.architecture_pattern} (confidence: {self.confidence:.0%})",
            f"Components: {self.component_count}",
            f"Code Quality: {self.code_quality_score:.0%}",
            f"Patterns found: {len(self.patterns)}",
        ]
        if self.recommendations:
            lines.append(f"Recommendations: {len(self.recommendations)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_pattern": self.architecture_pattern,
            "confidence": self.confidence,
            "component_count": self.component_count,
            "components": [c.to_dict() for c in self.components],
            "code_quality_score": self.code_quality_score,
            "patterns": self.patterns,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "raw_notes": self.raw_notes,
        }


# ═══════════════════════════════════════════════════════════════════
# CodebaseAnalyzer
# ═══════════════════════════════════════════════════════════════════


class CodebaseAnalyzer:
    """Analyzes architecture patterns, components, and code quality.

    Takes a CodeMap from the Mapper phase and produces an AnalysisResult.
    Optionally uses an LLM provider for advanced analysis.

    Usage:
        analyzer = CodebaseAnalyzer(llm=my_llm_provider)
        result = analyzer.full_analysis(code_map)
        print(result.summary())
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        """Initialize analyzer.

        Args:
            llm: Optional LLM provider for advanced analysis.
        """
        self.llm = llm

    # ── Architecture Detection ────────────────────────────────────

    def detect_architecture(self, code_map: CodeMap) -> str:
        """Detect the architecture pattern of the codebase.

        Uses directory structure and import patterns to guess architecture.
        With LLM available, uses it for higher-confidence detection.

        Args:
            code_map: The code map from Mapper.

        Returns:
            Architecture pattern name.
        """
        # Check directory structure hints
        dir_names: Set[str] = set()
        for f in code_map.files:
            dir_path = f.path.parent
            for part in dir_path.parts:
                dir_names.add(part.lower())

        # Check for MVC
        mvc_hits = dir_names & {"models", "views", "controllers", "templates"}
        if len(mvc_hits) >= 2:
            return ArchitecturePattern.MVC.value

        # Check for layered architecture
        layered_hits = dir_names & {
            "services", "repositories", "handlers", "domain",
            "api", "routes", "presentation", "application",
            "infrastructure", "data",
        }
        if len(layered_hits) >= 2:
            return ArchitecturePattern.LAYERED.value

        # Check file/import content for hints
        all_source = ""
        file_samples = 0
        for f in code_map.files:
            if f.is_python and file_samples < 20:
                try:
                    content = f.path.read_text(errors="ignore")
                    all_source += content[:2000] + "\n"
                    file_samples += 1
                except OSError:
                    pass

        # Check regex hints
        matches: Dict[str, int] = {}
        for pattern, keywords in _ARCH_HINTS.items():
            count = 0
            for kw in keywords:
                count += len(re.findall(kw, all_source, re.IGNORECASE))
            if count > 0:
                matches[pattern.value] = count

        if matches:
            best = max(matches, key=lambda k: matches[k])
            return best

        # Has subpackages → likely layered or package
        python_dirs: Set[str] = set()
        for f in code_map.files:
            if f.is_python:
                python_dirs.add(str(f.path.parent))
        if len(python_dirs) > 3:
            return ArchitecturePattern.LAYERED.value

        # Few files → script
        py_count = code_map.languages.get("python", 0)
        if py_count <= 3:
            return ArchitecturePattern.SCRIPT.value

        return ArchitecturePattern.UNKNOWN.value

    # ── Component Cataloging ─────────────────────────────────────

    def catalog_components(self, code_map: CodeMap) -> List[Component]:
        """Catalog logical components from directory structure.

        Groups files by directory to identify components.

        Args:
            code_map: The code map.

        Returns:
            List of Component objects.
        """
        components: List[Component] = []
        seen_dirs: Set[str] = set()

        # Group by immediate parent directory
        dir_groups: Dict[str, List[CodebaseFileInfo]] = {}
        for f in code_map.files:
            if f.is_python:
                parent = str(f.path.parent)
                if parent not in dir_groups:
                    dir_groups[parent] = []
                dir_groups[parent].append(f)

        for dir_path, files in dir_groups.items():
            if dir_path in seen_dirs:
                continue
            seen_dirs.add(dir_path)

            dir_name = Path(dir_path).name.lower()
            comp_type = _DIRECTORY_HINTS.get(dir_name, "module")

            # Determine dependencies from graph
            deps: List[str] = []
            for f in files:
                rel = f.relative_path
                f_deps = code_map.dependency_graph.get(rel, [])
                for dep in f_deps:
                    if dep not in deps:
                        deps.append(dep)

            components.append(Component(
                name=Path(dir_path).name.replace("_", " ").title(),
                path=dir_path,
                component_type=comp_type,
                dependencies=deps,
                file_count=len(files),
            ))

        return components

    # ── Code Quality Assessment ──────────────────────────────────

    def assess_code_quality(self, code_map: CodeMap) -> float:
        """Assess code quality score from structural indicators.

        Considers: test coverage ratio, documentation, module organization,
        dependency complexity, file size distribution.

        Args:
            code_map: The code map.

        Returns:
            Score from 0.0 to 1.0.
        """
        py_files = [f for f in code_map.files if f.is_python]
        if not py_files:
            return 0.5

        score = 0.5  # Start at 0.5 (neutral)

        # Test coverage proxy: ratio of test to non-test files
        test_files = [f for f in py_files if f.is_test]
        non_test = [f for f in py_files if not f.is_test]
        if non_test:
            test_ratio = len(test_files) / len(non_test)
            score += min(test_ratio, 0.3) * 0.5  # Up to +0.15 for good test coverage

        # Documentation: files with docstrings
        doc_count = 0
        for f in py_files[:50]:  # Sample first 50
            try:
                content = f.path.read_text(errors="ignore")
                # Simple heuristic: triple-quote strings
                if '"""' in content or "'''" in content:
                    doc_count += 1
            except OSError:
                pass
        if py_files:
            doc_ratio = doc_count / min(len(py_files), 50)
            score += min(doc_ratio, 1.0) * 0.1  # Up to +0.1 for docs

        # File size consistency (penalize extreme sizes)
        sizes = [f.size_bytes for f in py_files if f.size_bytes > 0]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            very_large = sum(1 for s in sizes if s > avg_size * 5)
            if very_large > 0:
                score -= min(very_large / len(sizes), 0.5) * 0.2  # Up to -0.1

        # Dependency complexity penalty
        dep_counts = [len(deps) for deps in code_map.dependency_graph.values()]
        if dep_counts:
            high_deps = sum(1 for d in dep_counts if d > 10)
            if high_deps > 0:
                score -= min(high_deps / len(dep_counts), 1.0) * 0.1  # Up to -0.1

        return max(0.0, min(1.0, score))

    # ── Pattern Detection ────────────────────────────────────────

    def detect_code_patterns(self, code_map: CodeMap) -> List[Dict[str, Any]]:
        """Detect code patterns and design idioms.

        Looks for: dataclasses, decorators, async patterns, factory methods,
        singletons, dependency injection patterns.

        Args:
            code_map: The code map.

        Returns:
            List of pattern info dicts with 'name', 'count', 'files'.
        """
        patterns: Dict[str, Dict[str, Any]] = {}

        # Scan Python files for patterns
        sample_files = [f for f in code_map.files if f.is_python][:30]
        for f in sample_files:
            try:
                content = f.path.read_text(errors="ignore")
            except OSError:
                continue

            # Dataclass pattern
            if "@dataclass" in content or "from dataclasses import" in content:
                pat = patterns.setdefault("dataclass", {"name": "dataclass", "count": 0, "files": []})
                pat["count"] += 1
                pat["files"].append(f.name)

            # Async patterns
            if "async def" in content or "await " in content:
                pat = patterns.setdefault("async", {"name": "async/await", "count": 0, "files": []})
                pat["count"] += 1
                pat["files"].append(f.name)

            # Decorator usage
            decorator_count = len(re.findall(r"@\w+", content))
            if decorator_count > 2:
                pat = patterns.setdefault("decorators", {"name": "decorators", "count": 0, "files": []})
                pat["count"] += 1
                pat["files"].append(f.name)

            # Type hints
            type_hint_count = len(re.findall(r":\s*(int|str|bool|float|list|dict|tuple|Optional|Union)\b", content))
            if type_hint_count > 2:
                pat = patterns.setdefault("type_hints", {"name": "type_hints", "count": 0, "files": []})
                pat["count"] += 1
                pat["files"].append(f.name)

            # Base classes / inheritance
            if "class" in content and "(" in content:
                pat = patterns.setdefault("inheritance", {"name": "inheritance", "count": 0, "files": []})
                pat["count"] += len(re.findall(r"class\s+\w+\s*\(\s*\w+", content))
                pat["files"].append(f.name)

            # ABC / abstract
            if "ABC" in content or "abstractmethod" in content:
                pat = patterns.setdefault("abstract", {"name": "abstract_base", "count": 0, "files": []})
                pat["count"] += 1
                pat["files"].append(f.name)

            # Factory pattern (function that returns an instance)
            factory_matches = re.findall(r"def\s+(create_\w+|make_\w+|build_\w+|new_\w+)\(", content)
            if factory_matches:
                pat = patterns.setdefault("factory", {"name": "factory_method", "count": 0, "files": []})
                pat["count"] += len(factory_matches)
                pat["files"].append(f.name)

        return list(patterns.values())

    # ── Analysis Pipeline ────────────────────────────────────────

    def analyze(self, code_map: CodeMap) -> AnalysisResult:
        """Quick analysis: architecture + quality + patterns.

        Args:
            code_map: The code map.

        Returns:
            AnalysisResult.
        """
        arch = self.detect_architecture(code_map)
        components = self.catalog_components(code_map)
        quality = self.assess_code_quality(code_map)
        patterns = self.detect_code_patterns(code_map)

        # Map architecture to enum to get confidence
        try:
            arch_enum = ArchitecturePattern(arch)
            confidence = 0.7 if arch_enum != ArchitecturePattern.UNKNOWN else 0.3
        except ValueError:
            confidence = 0.3

        return AnalysisResult(
            architecture_pattern=arch,
            confidence=confidence,
            component_count=len(components),
            components=components,
            code_quality_score=quality,
            patterns=patterns,
            metrics={
                "total_files": code_map.total_files,
                "python_files": code_map.languages.get("python", 0),
                "imports": len(code_map.imports),
                "third_party_deps": len(
                    [i for i in code_map.imports if i.import_type == "third_party"]
                ),
            },
        )

    def full_analysis(self, code_map: CodeMap) -> AnalysisResult:
        """Run complete analysis including LLM-powered insights if available.

        Args:
            code_map: The code map.

        Returns:
            Comprehensive AnalysisResult.
        """
        result = self.analyze(code_map)

        # LLM-enhanced analysis
        if self.llm and self.llm.enabled:
            result = self._llm_enhance(result, code_map)

        return result

    def _llm_enhance(
        self, result: AnalysisResult, code_map: CodeMap
    ) -> AnalysisResult:
        """Use LLM to provide deeper architectural analysis.

        Args:
            result: Current analysis result.
            code_map: The code map.

        Returns:
            Enhanced AnalysisResult.
        """
        # Build a concise prompt describing the codebase
        file_summary = "\n".join(
            f"  - {f.name} ({f.language}, {f.size_bytes}B)"
            for f in code_map.files[:30]
        )
        imports_summary = ", ".join(
            i.imported_name for i in code_map.imports
            if i.import_type == "third_party"
        )[:500]

        prompt = f"""Analyze this Python codebase structure:

Files ({code_map.total_files} total, {code_map.total_dirs} directories):
{file_summary}
...
Third-party imports: {imports_summary}
...
Initial architecture guess: {result.architecture_pattern}

Provide a concise codebase analysis with:
1. Architecture pattern identification
2. Key components and their roles
3. Code quality observations
4. One improvement recommendation

Format: plain text, 3-5 sentences."""

        try:
            llm_result = self.llm.generate_sync(prompt, max_tokens=300)
            result.raw_notes = llm_result
            # Boost confidence if LLM agrees
            if result.architecture_pattern in llm_result.lower():
                result.confidence = max(result.confidence, 0.85)
        except Exception:
            pass  # LLM is advisory only

        return result

    # ── Ralph Loop Integration ────────────────────────────────────

    def to_hierarchy_phase(
        self,
        result: AnalysisResult,
        engine: Optional[RalphLoopEngine] = None,
        parent_project: Optional[ProjectNode] = None,
    ) -> PhaseNode:
        """Create a Ralph Loop PhaseNode populated with analysis results.

        Args:
            result: The AnalysisResult.
            engine: Optional RalphLoopEngine instance.
            parent_project: Optional parent project.

        Returns:
            PhaseNode with analysis metadata.
        """
        from ..models.types import PhaseNode, NodeStatus

        node = PhaseNode(
            id="discovery-analyze",
            name="Codebase Analysis",
            status=NodeStatus.SUCCESS,
            progress=100.0,
            metadata={
                "architecture_pattern": result.architecture_pattern,
                "confidence": result.confidence,
                "component_count": result.component_count,
                "code_quality_score": result.code_quality_score,
                "pattern_count": len(result.patterns),
                "recommendations": result.recommendations,
                "analysis_notes": result.raw_notes,
            },
        )

        if engine and parent_project:
            parent_project.phases.append(node)

        return node


__all__ = [
    "ArchitecturePattern",
    "Component",
    "AnalysisResult",
    "CodebaseAnalyzer",
]