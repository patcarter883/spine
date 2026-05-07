"""Tests for Brownfields Discovery & Analysis Engine.

Tests cover:
- CodebaseMapper: file tree scanning, import detection, dependency mapping
- CodebaseAnalyzer: pattern recognition, architecture detection
- ReverseEngineer: spec inference, interface extraction
- Ralph Loop integration: ProjectNode/PhaseNode output
- Integration with providers and persistence
"""

import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataclasses import asdict

from spine.models.types import (
    ProjectNode,
    PhaseNode,
    SubPhaseNode,
    TaskNode,
    NodeStatus,
    HierarchyLevel,
)
from spine.core.hierarchy import RalphLoopEngine


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase with various structures for testing."""
    root = tmp_path / "sample_project"
    root.mkdir()

    # Python files with imports
    (root / "main.py").write_text(
        textwrap.dedent("""\
        import os
        import sys
        from .utils import helper
        from .models.user import User
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/")
        def root():
            return {"message": "hello"}
        """)
    )

    # Subpackage with models
    models_dir = root / "models"
    models_dir.mkdir()
    (models_dir / "__init__.py").write_text("")
    (models_dir / "user.py").write_text(
        textwrap.dedent("""\
        from dataclasses import dataclass
        from typing import Optional, List

        @dataclass
        class User:
            id: int
            name: str
            email: Optional[str] = None
        """)
    )

    # Utils
    (root / "utils.py").write_text(
        textwrap.dedent("""\
        def helper(x: int) -> int:
            \"\"\"A helper function.\"\"\"
            return x * 2
        """)
    )

    # Tests
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_user.py").write_text(
        textwrap.dedent("""\
        import pytest
        from ..models.user import User

        def test_user_creation():
            user = User(id=1, name="Alice")
            assert user.name == "Alice"
        """)
    )

    # Config files
    (root / "README.md").write_text("# Sample Project\nA test project.")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    )

    # Nesting
    nested = root / "deeply" / "nested" / "package"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("")
    (nested / "module.py").write_text('"""Deep module."""\n')

    return root


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.name = "mock:test"
    provider.generate_sync.return_value = "Mock analysis result"
    provider.enabled = True
    return provider


# ═══════════════════════════════════════════════════════════════════
# CodebaseMapper Tests
# ═══════════════════════════════════════════════════════════════════


class TestCodebaseFileInfo:
    """Tests for CodebaseFileInfo dataclass."""

    def test_create_file_info(self):
        from spine.discovery.mapper import CodebaseFileInfo
        info = CodebaseFileInfo(
            path=Path("src/main.py"),
            name="main.py",
            extension=".py",
            language="python",
            size_bytes=1024,
        )
        assert info.path == Path("src/main.py")
        assert info.name == "main.py"
        assert info.extension == ".py"
        assert info.language == "python"
        assert info.size_bytes == 1024
        assert info.is_test is False

    def test_file_info_is_test_detection(self):
        from spine.discovery.mapper import CodebaseFileInfo
        test_info = CodebaseFileInfo(
            path=Path("tests/test_main.py"),
            name="test_main.py",
            extension=".py",
            language="python",
            size_bytes=512,
        )
        assert test_info.is_test is True

    def test_file_info_to_dict(self):
        from spine.discovery.mapper import CodebaseFileInfo
        info = CodebaseFileInfo(
            path=Path("src/main.py"),
            name="main.py",
            extension=".py",
            language="python",
            size_bytes=100,
        )
        d = info.to_dict()
        assert d["name"] == "main.py"
        assert d["extension"] == ".py"
        assert d["language"] == "python"


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_create_import_info(self):
        from spine.discovery.mapper import ImportInfo
        imp = ImportInfo(
            source_file="main.py",
            imported_name="os",
            module_path="os",
            import_type="stdlib",
        )
        assert imp.source_file == "main.py"
        assert imp.imported_name == "os"
        assert imp.module_path == "os"
        assert imp.import_type == "stdlib"
        assert imp.is_relative is False
        assert imp.is_conditional is False

    def test_import_info_types(self):
        from spine.discovery.mapper import ImportInfo
        # Relative import
        rel = ImportInfo(
            source_file="main.py",
            imported_name="utils",
            module_path=".utils",
            import_type="relative",
        )
        assert rel.is_relative is True
        assert rel.import_type == "relative"

        # Third-party
        third = ImportInfo(
            source_file="app.py",
            imported_name="FastAPI",
            module_path="fastapi",
            import_type="third_party",
        )
        assert third.import_type == "third_party"
        assert third.is_relative is False


class TestCodeMap:
    """Tests for CodeMap dataclass."""

    def test_create_code_map(self):
        from spine.discovery.mapper import CodeMap, CodebaseFileInfo
        cm = CodeMap(
            root_path=Path("/test"),
            total_files=5,
            total_dirs=3,
            languages={"python": 4, "markdown": 1},
        )
        assert cm.root_path == Path("/test")
        assert cm.total_files == 5
        assert cm.total_dirs == 3
        assert cm.languages == {"python": 4, "markdown": 1}
        assert cm.imports == []
        assert cm.dependency_graph == {}

    def test_code_map_add_file(self):
        from spine.discovery.mapper import CodeMap, CodebaseFileInfo
        cm = CodeMap(root_path=Path("/test"))
        info = CodebaseFileInfo(
            path=Path("/test/main.py"),
            name="main.py",
            extension=".py",
            language="python",
            size_bytes=100,
        )
        cm.files.append(info)
        cm._recalc_counts()
        assert cm.total_files == 1
        assert cm.languages == {"python": 1}


class TestCodebaseMapper:
    """Tests for the file tree scanner."""

    def test_scan_file_tree_basic(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        cm = mapper.scan(sample_codebase)

        assert isinstance(cm.root_path, Path)
        assert cm.total_files > 0
        assert cm.total_dirs > 0
        assert "python" in cm.languages
        assert cm.languages.get("python", 0) > 0

    def test_scan_finds_python_files(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        cm = mapper.scan(sample_codebase)

        py_files = [f for f in cm.files if f.extension == ".py"]
        assert len(py_files) >= 3  # main, user, utils, etc.
        assert any(f.name == "main.py" for f in py_files)
        assert any(f.name == "user.py" for f in py_files)

    def test_scan_respects_ignore_patterns(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper(ignore_patterns=["tests/**"])
        cm = mapper.scan(sample_codebase)

        test_files = [f for f in cm.files if f.is_test]
        assert len(test_files) == 0

    def test_scan_respects_max_depth(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper(max_depth=1)
        cm = mapper.scan(sample_codebase)

        # Deeply nested should not appear at depth=1
        deep_files = [f for f in cm.files if "deeply" in str(f.path)]
        assert len(deep_files) == 0

    def test_scan_with_include_only(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper(include_extensions=[".py"])
        cm = mapper.scan(sample_codebase)

        for f in cm.files:
            assert f.extension == ".py"

    def test_detect_imports(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        mapper.scan(sample_codebase)
        imports = mapper.detect_imports(sample_codebase)

        assert len(imports) > 0
        # main.py imports
        main_imports = [i for i in imports if "main.py" in i.source_file]
        assert len(main_imports) > 0
        imported_names = {i.imported_name for i in main_imports}
        assert "FastAPI" in imported_names or "os" in imported_names

    def test_detect_imports_classifies_types(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        mapper.scan(sample_codebase)
        imports = mapper.detect_imports(sample_codebase)

        types_found = {i.import_type for i in imports}
        # Should detect relative, third_party, and stdlib
        assert len(types_found) >= 2

    def test_build_dependency_graph(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        cm = mapper.scan(sample_codebase)
        imports = mapper.detect_imports(sample_codebase)
        graph = mapper.build_dependency_graph(cm, imports)

        assert isinstance(graph, dict)
        # main.py should have dependencies
        if graph:
            any_has_deps = any(len(deps) > 0 for deps in graph.values())
            assert any_has_deps

    def test_full_map_pipeline(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        assert cm.total_files > 0
        assert len(cm.imports) > 0
        assert len(cm.files) > 0
        # Dependency graph should be populated
        assert isinstance(cm.dependency_graph, dict)
        assert len(cm.dependency_graph) > 0

    def test_file_tree_to_dict(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        mapper = CodebaseMapper()
        mapper.scan(sample_codebase)
        tree = mapper.file_tree(sample_codebase)

        assert isinstance(tree, dict)
        # Should have the root name as key
        assert sample_codebase.name in tree or len(tree) > 0

    def test_mapper_empty_directory(self, tmp_path: Path):
        from spine.discovery.mapper import CodebaseMapper
        empty = tmp_path / "empty_project"
        empty.mkdir()
        mapper = CodebaseMapper()
        cm = mapper.scan(empty)

        assert cm.total_files == 0
        assert cm.total_dirs == 1  # just the root
        assert cm.languages == {}


# ═══════════════════════════════════════════════════════════════════
# CodebaseAnalyzer Tests
# ═══════════════════════════════════════════════════════════════════


class TestArchitecturePattern:
    """Tests for ArchitecturePattern detection."""

    def test_pattern_types(self):
        from spine.discovery.analyzer import ArchitecturePattern
        patterns = list(ArchitecturePattern)
        assert len(patterns) > 0
        assert ArchitecturePattern.LAYERED in patterns
        assert ArchitecturePattern.MVC in patterns
        assert ArchitecturePattern.MICROSERVICES in patterns


class TestComponent:
    """Tests for Component dataclass."""

    def test_create_component(self):
        from spine.discovery.analyzer import Component
        comp = Component(
            name="UserService",
            path="src/services/user_service.py",
            component_type="service",
            dependencies=["Database", "Cache"],
        )
        assert comp.name == "UserService"
        assert comp.component_type == "service"
        assert comp.dependencies == ["Database", "Cache"]
        assert comp.responsibilities == []

    def test_component_add_responsibility(self):
        from spine.discovery.analyzer import Component
        comp = Component(
            name="AuthHandler",
            path="src/handlers/auth.py",
            component_type="handler",
        )
        comp.responsibilities.append("Handle user authentication")
        assert "Handle user authentication" in comp.responsibilities


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_create_analysis_result(self):
        from spine.discovery.analyzer import AnalysisResult
        result = AnalysisResult(
            architecture_pattern="layered",
            confidence=0.85,
            component_count=12,
        )
        assert result.architecture_pattern == "layered"
        assert result.confidence == 0.85
        assert result.component_count == 12
        assert result.components == []
        assert result.code_quality_score is not None
        assert 0.0 <= result.code_quality_score <= 1.0

    def test_analysis_summary(self):
        from spine.discovery.analyzer import AnalysisResult
        result = AnalysisResult(
            architecture_pattern="microservices",
            confidence=0.92,
            component_count=5,
            code_quality_score=0.78,
        )
        summary = result.summary()
        assert "microservices" in summary
        assert "5" in summary
        assert "78" in summary


class TestCodebaseAnalyzer:
    """Tests for the architecture analyzer."""

    def test_analyze_from_code_map(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        result = analyzer.analyze(cm)

        assert result.confidence >= 0.0
        assert result.component_count >= 0

    def test_detect_architecture(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        arch = analyzer.detect_architecture(cm)

        assert isinstance(arch, str)
        assert len(arch) > 0

    def test_catalog_components(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper
        from spine.discovery.analyzer import Component

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        components = analyzer.catalog_components(cm)

        assert isinstance(components, list)
        for comp in components:
            assert isinstance(comp, Component)
            assert comp.name

    def test_assess_code_quality(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        score = analyzer.assess_code_quality(cm)

        assert 0.0 <= score <= 1.0

    def test_detect_code_patterns(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        patterns = analyzer.detect_code_patterns(cm)

        assert isinstance(patterns, list)
        # Should detect dataclass pattern in user.py
        assert any("dataclass" in p["name"].lower() for p in patterns)

    def test_analyze_with_llm(self, sample_codebase: Path, mock_llm_provider):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer(llm=mock_llm_provider)
        result = analyzer.analyze(cm)

        assert result.confidence >= 0.0
        # LLM integration - may or may not be called depending on confidence

    def test_full_analysis_pipeline(self, sample_codebase: Path):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        result = analyzer.full_analysis(cm)

        assert result.architecture_pattern
        assert result.component_count >= 0
        assert len(result.components) > 0
        assert result.code_quality_score >= 0.0


# ═══════════════════════════════════════════════════════════════════
# ReverseEngineer Tests
# ═══════════════════════════════════════════════════════════════════


class TestInferredSpec:
    """Tests for InferredSpec."""

    def test_create_inferred_spec(self):
        from spine.discovery.reverse_engineer import InferredSpec
        spec = InferredSpec(
            project_name="TestProject",
            description="A test project",
            confidence=0.75,
        )
        assert spec.project_name == "TestProject"
        assert spec.description == "A test project"
        assert spec.confidence == 0.75
        assert spec.inferred_requirements == []
        assert spec.entry_points == []
        assert spec.technologies == []

    def test_inferred_spec_summary(self):
        from spine.discovery.reverse_engineer import InferredSpec
        spec = InferredSpec(
            project_name="MyApp",
            description="Web application",
            confidence=0.88,
            inferred_requirements=["req1", "req2"],
            entry_points=["main:app"],
            technologies=["python", "fastapi"],
        )
        summary = spec.to_summary()
        assert "MyApp" in summary
        assert "python" in summary or "Python" in summary


class TestApiEndpoint:
    """Tests for API endpoint detection."""

    def test_create_api_endpoint(self):
        from spine.discovery.reverse_engineer import ApiEndpoint
        ep = ApiEndpoint(
            method="GET",
            path="/users/{id}",
            handler="get_user",
            source_file="routers/users.py",
            line=42,
        )
        assert ep.method == "GET"
        assert ep.path == "/users/{id}"
        assert ep.handler == "get_user"
        assert ep.source_file == "routers/users.py"
        assert ep.line == 42


class TestDataSchema:
    """Tests for schema detection."""

    def test_create_schema(self):
        from spine.discovery.reverse_engineer import DataSchema
        schema = DataSchema(
            name="User",
            source_file="models/user.py",
            fields={"id": "int", "name": "str", "email": "Optional[str]"},
        )
        assert schema.name == "User"
        assert schema.fields["id"] == "int"
        assert len(schema.fields) == 3


class TestReverseEngineer:
    """Tests for the reverse engineer."""

    def test_extract_entry_points(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer()
        entry_points = re.extract_entry_points(sample_codebase)

        assert isinstance(entry_points, list)
        # main.py should be detected as entry point
        assert any("main.py" in ep.get("file", "") for ep in entry_points)

    def test_detect_technologies(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer()
        techs = re.detect_technologies(sample_codebase)

        assert isinstance(techs, list)
        assert "Python" in techs or "python" in techs
        # Should detect FastAPI
        assert any("fastapi" in t.lower() or "fastapi" in t for t in techs)

    def test_infer_requirements(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer()
        reqs = re.infer_requirements(sample_codebase)

        assert isinstance(reqs, list)
        assert len(reqs) > 0

    def test_extract_data_schemas(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        from spine.discovery.reverse_engineer import DataSchema

        re = ReverseEngineer()
        schemas = re.extract_data_schemas(sample_codebase)

        assert isinstance(schemas, list)
        # Should detect User dataclass
        assert any(s.name == "User" for s in schemas)
        user = next(s for s in schemas if s.name == "User")
        assert "id" in user.fields
        assert "name" in user.fields

    def test_infer_apis(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        from spine.discovery.reverse_engineer import ApiEndpoint

        re = ReverseEngineer()
        apis = re.infer_apis(sample_codebase)

        assert isinstance(apis, list)
        # Should detect the FastAPI route in main.py
        assert any("root" in ep.handler.lower() or "/" in ep.path for ep in apis)

    def test_reverse_engineer_full(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer()
        spec = re.reverse_engineer(sample_codebase)

        assert isinstance(spec.project_name, str)
        assert spec.confidence >= 0.0
        assert len(spec.technologies) > 0
        assert len(spec.inferred_requirements) > 0

    def test_reverse_engineer_with_llm(self, sample_codebase: Path, mock_llm_provider):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer(llm=mock_llm_provider)
        spec = re.reverse_engineer(sample_codebase)

        assert spec.confidence >= 0.0
        assert spec.project_name

    def test_parse_docstrings(self, sample_codebase: Path):
        from spine.discovery.reverse_engineer import ReverseEngineer
        re = ReverseEngineer()
        docs = re.parse_docstrings(sample_codebase)

        assert isinstance(docs, list)
        # Helper has a docstring
        if docs:
            assert any("helper" in d.get("name", "").lower() for d in docs)


# ═══════════════════════════════════════════════════════════════════
# Discovery Engine Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestDiscoveryEngine:
    """Integration tests for the full discovery pipeline."""

    def test_create_engine(self):
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper
        from spine.discovery.reverse_engineer import ReverseEngineer

        mapper = CodebaseMapper()
        analyzer = CodebaseAnalyzer()
        re = ReverseEngineer()

        assert mapper is not None
        assert analyzer is not None
        assert re is not None

    def test_discovery_produces_hierarchy(self, sample_codebase: Path):
        """Discovery results should feed into Ralph Loop hierarchy."""
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.mapper import CodebaseMapper
        from spine.discovery.reverse_engineer import ReverseEngineer
        from spine.core.hierarchy import RalphLoopEngine

        engine = RalphLoopEngine()
        project = engine.create_project(
            id="discovery-sample",
            name=f"Discovery: {sample_codebase.name}"
        )

        # Create discovery phases
        map_phase = engine.create_phase(
            id="discovery-map",
            name="Codebase Mapping",
            parent_project=project,
        )
        analyze_phase = engine.create_phase(
            id="discovery-analyze",
            name="Codebase Analysis",
            parent_project=project,
        )
        re_phase = engine.create_phase(
            id="discovery-re",
            name="Reverse Engineering",
            parent_project=project,
        )

        assert len(project.phases) == 3
        assert project.phases[0].name == "Codebase Mapping"

        # Validate the hierarchy
        from spine.core.hierarchy import HierarchyValidator
        validator = HierarchyValidator()
        result = validator.validate(project)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_discovery_populates_phase_metadata(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        from spine.core.hierarchy import RalphLoopEngine
        from spine.models.types import NodeStatus

        mapper = CodebaseMapper()
        engine = RalphLoopEngine()

        project = engine.create_project("meta-test", "Meta Test")
        map_phase = engine.create_phase("map", "Mapping", parent_project=project)

        cm = mapper.map_codebase(sample_codebase)

        # Store discovery data in phase metadata
        map_phase.metadata["total_files"] = cm.total_files
        map_phase.metadata["total_dirs"] = cm.total_dirs
        map_phase.metadata["languages"] = cm.languages
        map_phase.status = NodeStatus.SUCCESS

        assert map_phase.metadata["total_files"] > 0
        assert map_phase.metadata["languages"]
        assert map_phase.status == NodeStatus.SUCCESS

    def test_end_to_end_discovery_flow(self, sample_codebase: Path):
        """Full discovery flow: map → analyze → reverse engineer → hierarchy."""
        from spine.discovery.mapper import CodebaseMapper
        from spine.discovery.analyzer import CodebaseAnalyzer, AnalysisResult
        from spine.discovery.reverse_engineer import ReverseEngineer, InferredSpec
        from spine.core.hierarchy import RalphLoopEngine
        from spine.models.types import NodeStatus

        # Step 1: Map
        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)
        assert cm.total_files > 0

        # Step 2: Analyze
        analyzer = CodebaseAnalyzer()
        analysis = analyzer.full_analysis(cm)
        assert analysis.architecture_pattern

        # Step 3: Reverse Engineer
        re = ReverseEngineer()
        spec = re.reverse_engineer(sample_codebase)
        assert spec.project_name

        # Step 4: Build Ralph Loop Hierarchy
        engine = RalphLoopEngine()
        project = engine.create_project(
            id=f"discovery-{sample_codebase.name}",
            name=f"Discovery: {sample_codebase.name}"
        )

        # Fill with discovery results
        project.metadata.update({
            "total_files": cm.total_files,
            "total_dirs": cm.total_dirs,
            "languages": cm.languages,
            "architecture": analysis.architecture_pattern,
            "component_count": analysis.component_count,
            "spec_confidence": spec.confidence,
        })
        engine.transition_node(project, NodeStatus.SUCCESS)

        assert project.status == NodeStatus.SUCCESS
        assert project.metadata["architecture"]
        assert project.metadata["total_files"] > 0


# ═══════════════════════════════════════════════════════════════════
# Persistence Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestDiscoveryPersistence:
    """Tests for persistence integration with discovery results."""

    def test_discovery_results_persistable(self, sample_codebase: Path):
        from spine.discovery.mapper import CodebaseMapper
        from spine.discovery.analyzer import CodebaseAnalyzer
        from spine.discovery.reverse_engineer import ReverseEngineer
        from spine.core.hierarchy import RalphLoopEngine
        from spine.models.types import NodeStatus
        import json

        # Run discovery
        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        analyzer = CodebaseAnalyzer()
        analysis = analyzer.full_analysis(cm)

        re = ReverseEngineer()
        spec = re.reverse_engineer(sample_codebase)

        # Build hierarchy
        engine = RalphLoopEngine()
        project = engine.create_project("persist-test", "Persist Test")
        project.status = NodeStatus.SUCCESS

        # Verify serializable to JSON
        project_dict = asdict(project)
        json_str = json.dumps(project_dict, default=str)
        assert len(json_str) > 0
        reloaded = json.loads(json_str)
        assert reloaded["id"] == "persist-test"

    def test_persistence_integration_creates_checkpoint(self, sample_codebase: Path):
        from spine.core.persistence import ContinuityManager
        from spine.discovery.mapper import CodebaseMapper

        mapper = CodebaseMapper()
        cm = mapper.map_codebase(sample_codebase)

        continuity = ContinuityManager()
        checkpoint = continuity.create_checkpoint(
            work_item_id="discovery-test",
            phase_name="discovery",
            phase_progress=0.5,
            state={},
            dag={},
            context_vars={},
            swarm_state={
                "active_subphases": ["map", "analyze"],
                "pending_gates": [],
                "file_reservations": {},
            },
        )

        assert checkpoint.work_item_id == "discovery-test"
        assert checkpoint.phase_name == "discovery"

        saved_path = continuity.save_checkpoint(checkpoint)
        assert saved_path
        assert Path(saved_path).exists()
