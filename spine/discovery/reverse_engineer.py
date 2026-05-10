"""Reverse Engineer — infers project specifications, APIs, schemas, and
architecture from an existing codebase.

This is the third phase of the Brownfields Discovery & Analysis Engine.
Consumes output from Mapper (CodeMap) and Analyzer (AnalysisResult).
Produces InferredSpec with detected APIs, schemas, requirements, and technologies.
Integrates with LLM provider for intelligent spec inference.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .mapper import CodeMap, ImportInfo

if TYPE_CHECKING:
    from ..providers.llm import LLMProvider
    from ..models.types import PhaseNode, ProjectNode
    from ..core.hierarchy import RalphLoopEngine


# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ApiEndpoint:
    """A detected API endpoint in the codebase."""

    method: str  # GET, POST, PUT, DELETE, PATCH, etc.
    path: str  # URL path, e.g., "/users/{id}"
    handler: str  # Function name
    source_file: str  # File containing the endpoint
    line: int = 0
    description: str = ""
    parameters: List[str] = field(default_factory=list)
    response_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "handler": self.handler,
            "source_file": self.source_file,
            "line": self.line,
            "description": self.description,
            "parameters": self.parameters,
            "response_type": self.response_type,
        }


@dataclass
class DataSchema:
    """A detected data schema / model in the codebase."""

    name: str
    source_file: str
    fields: Dict[str, str] = field(default_factory=dict)  # field_name → type
    schema_type: str = "unknown"  # dataclass, pydantic, sqlalchemy, etc.
    description: str = ""
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self.source_file,
            "fields": self.fields,
            "type": self.schema_type,
            "description": self.description,
            "line": self.line,
        }


@dataclass
class InferredSpec:
    """Inferred project specification from reverse engineering.

    Represents the complete understanding of what the codebase does,
    including its purpose, requirements, APIs, schemas, and technologies.
    """

    project_name: str = ""
    description: str = ""
    confidence: float = 0.0
    inferred_requirements: List[str] = field(default_factory=list)
    entry_points: List[Dict[str, str]] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    apis: List[ApiEndpoint] = field(default_factory=list)
    schemas: List[DataSchema] = field(default_factory=list)
    architecture: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)  # name → version/type
    raw_notes: str = ""

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Project: {self.project_name}",
            f"Description: {self.description}",
            f"Confidence: {self.confidence:.0%}",
            f"Technologies: {', '.join(self.technologies[:10])}",
        ]
        if self.apis:
            lines.append(f"APIs: {len(self.apis)} endpoints")
        if self.schemas:
            lines.append(f"Schemas: {len(self.schemas)} data models")
        if self.inferred_requirements:
            lines.append(f"Requirements: {len(self.inferred_requirements)} inferred")
        if self.entry_points:
            lines.append(f"Entry points: {len(self.entry_points)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "description": self.description,
            "confidence": self.confidence,
            "inferred_requirements": self.inferred_requirements,
            "entry_points": self.entry_points,
            "technologies": self.technologies,
            "apis": [a.to_dict() for a in self.apis],
            "schemas": [s.to_dict() for s in self.schemas],
            "architecture": self.architecture,
            "dependencies": self.dependencies,
            "raw_notes": self.raw_notes,
        }


# ═══════════════════════════════════════════════════════════════════
# Technology detection
# ═══════════════════════════════════════════════════════════════════

# Framework / library detection via import patterns
_TECH_IMPORTS: Dict[str, str] = {
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "starlette": "Starlette",
    "sanik": "Sanic",
    "tornado": "Tornado",
    "aiohttp": "aiohttp",
    "sqlalchemy": "SQLAlchemy",
    "sqlmodel": "SQLModel",
    "tortoise": "Tortoise ORM",
    "pony": "PonyORM",
    "peewee": "Peewee",
    "pydantic": "Pydantic",
    "marshmallow": "Marshmallow",
    "attrs": "attrs",
    "litestar": "Litestar",
    "httpx": "HTTPX",
    "requests": "Requests",
    "redis": "Redis",
    "pymongo": "MongoDB",
    "motor": "MongoDB (async)",
    "aioredis": "Redis (async)",
    "celery": "Celery",
    "dramatiq": "Dramatiq",
    "arq": "ARQ",
    "pytest": "pytest",
    "unittest": "unittest",
    "click": "Click",
    "typer": "Typer",
    "rich": "Rich",
    "loguru": "Loguru",
    "structlog": "Structlog",
    "pandas": "pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "tensorflow": "TensorFlow",
    "torch": "PyTorch",
    "jax": "JAX",
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "openai": "OpenAI SDK",
    "anthropic": "Anthropic SDK",
    "transformers": "HuggingFace Transformers",
    "grpc": "gRPC",
    "protobuf": "Protocol Buffers",
    "graphql": "GraphQL",
    "strawberry": "Strawberry GraphQL",
    "graphene": "Graphene",
    "websockets": "WebSockets",
    "sse": "Server-Sent Events",
    "jinja2": "Jinja2",
    "mako": "Mako",
    "psycopg2": "PostgreSQL",
    "psycopg": "PostgreSQL",
    "asyncpg": "PostgreSQL (async)",
    "aiosqlite": "SQLite (async)",
}

# Config / build file patterns
_BUILD_FILE_HINTS: Dict[str, str] = {
    "pyproject.toml": "Python",
    "setup.py": "Python",
    "setup.cfg": "Python",
    "requirements.txt": "Python",
    "Pipfile": "Python/pipenv",
    "poetry.lock": "Python/poetry",
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "build.gradle": "Java/Kotlin/Gradle",
    "pom.xml": "Java/Maven",
    "Makefile": "Make",
    "CMakeLists.txt": "CMake",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    ".github/workflows/": "GitHub Actions",
    ".gitlab-ci.yml": "GitLab CI",
}


# ═══════════════════════════════════════════════════════════════════
# ReverseEngineer
# ═══════════════════════════════════════════════════════════════════


class ReverseEngineer:
    """Infers project specifications from existing code.

    Consumes output from Mapper and Analyzer. Produces InferredSpec.
    Optionally uses an LLM provider for intelligent spec inference.

    Usage:
        re = ReverseEngineer(llm=my_llm_provider)
        spec = re.reverse_engineer(Path("/path/to/project"))
        print(spec.to_summary())
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        """Initialize reverse engineer.

        Args:
            llm: Optional LLM provider for intelligent inference.
        """
        self.llm = llm

    # ── Entry Points ─────────────────────────────────────────────

    def extract_entry_points(self, root: Path) -> List[Dict[str, str]]:
        """Find likely entry points: main files, CLI scripts, apps.

        Args:
            root: Root directory.

        Returns:
            List of entry point dicts with 'file', 'type', 'description'.
        """
        entries: List[Dict[str, str]] = []

        # Check common patterns
        candidates = [
            "main.py", "app.py", "cli.py", "run.py",
            "server.py", "manage.py", "wsgi.py", "asgi.py",
            "index.py", "bot.py", "worker.py",
        ]
        for fname in candidates:
            for found in root.rglob(fname):
                if found.is_file():
                    entries.append({
                        "file": str(found.relative_to(root)),
                        "type": "executable",
                        "description": f"Entry point: {fname}",
                    })

        # Check for __main__.py
        for found in root.rglob("__main__.py"):
            if found.is_file():
                entries.append({
                    "file": str(found.relative_to(root)),
                    "type": "package",
                    "description": "Package entry via __main__",
                })

        # Check pyproject.toml scripts
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                scripts = re.findall(r'(\w+)\s*=\s*["\']([^"\']+:main)', content)
                if scripts:
                    for name, path in scripts[:5]:
                        entries.append({
                            "file": path.split(":")[0].replace(".", "/") + ".py",
                            "type": "cli",
                            "description": f"CLI command: {name} = {path}",
                        })
            except OSError:
                pass

        # Check config files
        dockerfile = root / "Dockerfile"
        if dockerfile.exists():
            cmd = dockerfile.read_text(errors="ignore")
            entries.append({
                "file": "Dockerfile",
                "type": "container",
                "description": "Docker container entry",
            })

        return entries

    # ── Technology Detection ─────────────────────────────────────

    def detect_technologies(self, root: Path) -> List[str]:
        """Detect technologies/frameworks used in the codebase.

        Scans imports and config files for technology signatures.

        Args:
            root: Root directory.

        Returns:
            List of technology names.
        """
        techs: Set[str] = set()

        # Always Python for Python projects
        techs.add("Python")

        # Scan Python files for import-based tech detection
        for py_file in root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for imp, tech_name in _TECH_IMPORTS.items():
                if re.search(rf"\bimport\s+{imp}\b|from\s+{imp}\s+import", content):
                    techs.add(tech_name)

        # Scan config files
        for fname, tech in _BUILD_FILE_HINTS.items():
            fpath = root / fname
            if fpath.exists():
                parts = tech.split("/")
                for p in parts:
                    techs.add(p)

        return sorted(techs)

    # ── Requirement Inference ────────────────────────────────────

    def infer_requirements(self, root: Path) -> List[str]:
        """Infer high-level requirements from code structure.

        Scans for docstrings, README, pyproject.toml, etc.

        Args:
            root: Root directory.

        Returns:
            List of inferred requirements.
        """
        requirements: List[str] = []

        # Read README
        readme = root / "README.md"
        if readme.exists():
            try:
                content = readme.read_text(errors="ignore")
                # Extract bullet points as potential requirements
                bullets = re.findall(r'[-*]\s+(.*)', content)
                for bullet in bullets[:5]:
                    requirements.append(f"README: {bullet.strip()[:200]}")
            except OSError:
                pass

        # Parse pyproject.toml
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(errors="ignore")
                name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
                if name_match:
                    requirements.append(f"Package name: {name_match.group(1)}")
                desc_match = re.search(r'description\s*=\s*"([^"]+)"', content)
                if desc_match:
                    requirements.append(f"Purpose: {desc_match.group(1)}")
            except OSError:
                pass

        # Scan for top-level docstrings
        for py_file in list(root.rglob("*.py"))[:10]:
            if "__pycache__" in py_file.parts:
                continue
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source)
                doc = ast.get_docstring(tree)
                if doc and len(doc) > 20:
                    requirements.append(
                        f"Module {py_file.relative_to(root)}: {doc[:200]}"
                    )
            except (SyntaxError, OSError, ast.ASTError):
                pass

        return requirements

    # ── Data Schema Extraction ──────────────────────────────────

    def extract_data_schemas(self, root: Path) -> List[DataSchema]:
        """Extract data schemas/models from Python dataclasses, Pydantic models, etc.

        Args:
            root: Root directory.

        Returns:
            List of DataSchema objects.
        """
        schemas: List[DataSchema] = []

        for py_file in root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            rel = str(py_file.relative_to(root))
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            # Find class definitions
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                # Check if it's a dataclass or model
                has_decorator = any(
                    isinstance(d, ast.Name) and d.id == "dataclass"
                    for d in node.decorator_list
                )
                has_pydantic = any(
                    hasattr(base, 'id') and base.id == "BaseModel"
                    for base in node.bases
                ) if node.bases else False

                if not has_decorator and not has_pydantic:
                    # Check for field annotations as a heuristic
                    ann_assigns = [
                        n for n in ast.walk(node)
                        if isinstance(n, ast.AnnAssign)
                    ]
                    if len(ann_assigns) < 2:
                        continue

                # Determine schema type
                schema_type = "class"
                if has_decorator:
                    schema_type = "dataclass"
                if has_pydantic:
                    schema_type = "pydantic"

                # Extract fields
                fields: Dict[str, str] = {}
                for child in node.body:
                    if isinstance(child, ast.AnnAssign):
                        field_name = (
                            child.target.id
                            if isinstance(child.target, ast.Name)
                            else str(type(child.target))
                        )
                        field_type = ast.unparse(child.annotation) if child.annotation else "Any"
                        fields[field_name] = field_type

                if fields:
                    schemas.append(DataSchema(
                        name=node.name,
                        source_file=rel,
                        fields=fields,
                        schema_type=schema_type,
                        line=node.lineno,
                        description=(
                            ast.get_docstring(node) or ""
                        ),
                    ))

        return schemas

    # ── API Inference ────────────────────────────────────────────

    def infer_apis(self, root: Path) -> List[ApiEndpoint]:
        """Detect API endpoints from route decorators.

        Supports: FastAPI, Flask, Django REST, Starlette patterns.

        Args:
            root: Root directory.

        Returns:
            List of ApiEndpoint objects.
        """
        endpoints: List[ApiEndpoint] = []

        for py_file in root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            rel = str(py_file.relative_to(root))
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Check decorators for route patterns
                for decorator in node.decorator_list:
                    method = ""
                    path = ""

                    if isinstance(decorator, ast.Call):
                        # e.g., @app.get("/"), @blueprint.route("/")
                        if isinstance(decorator.func, ast.Attribute):
                            attr = decorator.func.attr.lower()
                            if attr in ("get", "post", "put", "delete", "patch", "options", "head"):
                                method = attr.upper()
                        elif isinstance(decorator.func, ast.Name):
                            # Simple decorator
                            pass

                        # Get path from first string argument
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            path = str(decorator.args[0].value)

                    # Flask-style @app.route("/path", methods=["GET"])
                    if not method and isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == "route":
                                # Check for methods kwarg
                                for kw in decorator.keywords:
                                    if kw.arg == "methods":
                                        if isinstance(kw.value, ast.List):
                                            methods = [
                                                e.value
                                                for e in kw.value.elts
                                                if isinstance(e, ast.Constant)
                                            ]
                                            if methods:
                                                method = methods[0].upper()
                                if not method:
                                    method = "GET"  # Default
                                if (
                                    decorator.args
                                    and isinstance(decorator.args[0], ast.Constant)
                                ):
                                    path = str(decorator.args[0].value)

                    if method and path:
                        endpoints.append(ApiEndpoint(
                            method=method,
                            path=path,
                            handler=node.name,
                            source_file=rel,
                            line=node.lineno,
                            parameters=[
                                p.arg for p in node.args.args[1:]  # Skip self/cls
                            ],
                            description=(
                                ast.get_docstring(node) or ""
                            ),
                        ))

        return endpoints

    # ── Docstring Parsing ────────────────────────────────────────

    def parse_docstrings(self, root: Path) -> List[Dict[str, Any]]:
        """Parse docstrings from top-level functions and classes.

        Args:
            root: Root directory.

        Returns:
            List of dicts with 'name', 'file', 'docstring', 'type'.
        """
        docs: List[Dict[str, Any]] = []

        for py_file in root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            rel = str(py_file.relative_to(root))
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc = ast.get_docstring(node)
                    if doc:
                        docs.append({
                            "name": node.name,
                            "type": "function",
                            "file": rel,
                            "line": node.lineno,
                            "docstring": doc,
                        })
                elif isinstance(node, ast.AsyncFunctionDef):
                    doc = ast.get_docstring(node)
                    if doc:
                        docs.append({
                            "name": node.name,
                            "type": "async_function",
                            "file": rel,
                            "line": node.lineno,
                            "docstring": doc,
                        })

        return docs

    # ── Full Reverse Engineering Pipeline ────────────────────────

    def reverse_engineer(self, root: Path) -> InferredSpec:
        """Run the full reverse engineering pipeline.

        Args:
            root: Root directory of the codebase.

        Returns:
            InferredSpec with complete project understanding.
        """
        root = root.resolve()

        spec = InferredSpec(
            project_name=root.name,
            description="",
            confidence=0.5,
        )

        # Extract entry points
        spec.entry_points = self.extract_entry_points(root)

        # Detect technologies
        spec.technologies = self.detect_technologies(root)

        # Infer requirements
        spec.inferred_requirements = list(dict.fromkeys(
            self.infer_requirements(root)
        ))  # deduplicate

        # Extract data schemas
        spec.schemas = self.extract_data_schemas(root)

        # Infer APIs
        spec.apis = self.infer_apis(root)

        # Build dependencies from imports (run in mapper context)
        spec.dependencies = self._extract_dependency_versions(root)

        # Build description from requirements
        if spec.inferred_requirements:
            spec.description = "; ".join(spec.inferred_requirements[:3])[:500]

        # Adjust confidence based on findings
        if spec.apis:
            spec.confidence += 0.15
        if spec.schemas:
            spec.confidence += 0.1
        if spec.entry_points:
            spec.confidence += 0.1
        if spec.technologies:
            spec.confidence += 0.1
        spec.confidence = min(1.0, spec.confidence)

        # LLM enhancement
        if self.llm and self.llm.enabled:
            spec = self._llm_enhance(spec, root)

        return spec

    def _extract_dependency_versions(self, root: Path) -> Dict[str, str]:
        """Extract dependency information from project files."""
        deps: Dict[str, str] = {}

        # pyproject.toml
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(errors="ignore")
                deps["pyproject.toml"] = "found"
                # Try to find dependency section
                dep_matches = re.findall(
                    r'"([^"]+)"\s*#', content
                )
                for dep in dep_matches[:10]:
                    deps.setdefault("dependencies", "")
                    deps["dependencies"] += dep + ", "
            except OSError:
                pass

        return deps

    def _llm_enhance(self, spec: InferredSpec, root: Path) -> InferredSpec:
        """Use LLM for intelligent spec enhancement.

        Args:
            spec: Current specification.
            root: Root directory.

        Returns:
            Enhanced spec.
        """
        context = f"""Project: {spec.project_name}
Technologies: {', '.join(spec.technologies[:10])}
APIs: {len(spec.apis)} endpoints
{', '.join(f'{e.method} {e.path}' for e in spec.apis[:5])}
Schemas: {len(spec.schemas)} models
{', '.join(s.name for s in spec.schemas[:10])}
Requirements: {'; '.join(spec.inferred_requirements[:3])}

Based on this codebase analysis, provide a concise project purpose
description (1-2 sentences) and identify the primary architectural
pattern. Format: plain text."""

        try:
            llm_result = self.llm.generate_sync(context, max_tokens=200)
            spec.raw_notes = llm_result
            spec.description = llm_result[:200]
            spec.confidence = min(1.0, spec.confidence + 0.15)
        except Exception:
            pass  # LLM is advisory

        return spec

    # ── Ralph Loop Integration ────────────────────────────────────

    def to_hierarchy_phase(
        self,
        spec: InferredSpec,
        engine: Optional[RalphLoopEngine] = None,
        parent_project: Optional[ProjectNode] = None,
    ) -> PhaseNode:
        """Create a Ralph Loop PhaseNode populated with reverse-engineered spec.

        Args:
            spec: The InferredSpec.
            engine: Optional RalphLoopEngine instance.
            parent_project: Optional parent project.

        Returns:
            PhaseNode with spec metadata.
        """
        from ..models.types import PhaseNode, NodeStatus

        node = PhaseNode(
            id="discovery-reverse-engineer",
            name="Reverse Engineering",
            status=NodeStatus.SUCCESS,
            progress=100.0,
            metadata={
                "project_name": spec.project_name,
                "description": spec.description,
                "confidence": spec.confidence,
                "technology_count": len(spec.technologies),
                "api_count": len(spec.apis),
                "schema_count": len(spec.schemas),
                "requirement_count": len(spec.inferred_requirements),
                "raw_notes": spec.raw_notes,
                "technologies": spec.technologies,
            },
        )

        if engine and parent_project:
            parent_project.phases.append(node)

        return node


__all__ = [
    "ApiEndpoint",
    "DataSchema",
    "InferredSpec",
    "ReverseEngineer",
]