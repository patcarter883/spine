"""Tests for cross-slice API contract validation in PLAN synthesis.

The plan can decompose work so a PRODUCER slice creates one API while CONSUMER
slices reference a different, non-existent name — verify then never converges
(trace 019f2040: producer builds `add_provider`, consumers call
`add_embedding_provider` that no slice creates). `repair_and_validate_contracts`
auto-injects producer→consumer dependency edges and flags references no slice
provides, so the manager can reconcile the contract before the workers run.

These exercise the deterministic validator; the manager/worker LLM calls are not
invoked here.
"""

from __future__ import annotations

import pytest

from spine.agents.plan_synthesis import (
    PlanSkeleton,
    SliceStub,
    _contract_block,
    _leaf,
    _root,
    repair_and_validate_contracts,
)


def _stub(sid: str, **kw) -> SliceStub:
    kw.setdefault("title", sid)
    kw.setdefault("summary", f"do {sid}")
    return SliceStub(id=sid, **kw)


@pytest.fixture()
def no_index(monkeypatch: pytest.MonkeyPatch):
    """Make the index report a controllable set of existing symbols (by leaf)."""

    def _install(existing_leaves: set[str]) -> None:
        monkeypatch.setattr(
            "spine.agents.plan_synthesis._symbol_exists_in_index",
            lambda db_path, sym: _leaf(sym) in existing_leaves,
        )

    return _install


def test_leaf_and_root() -> None:
    assert _leaf("api.add_provider") == "add_provider"
    assert _leaf("UIApi.add_provider()") == "add_provider"
    assert _leaf("bare") == "bare"
    assert _root("st.form") == "st"
    assert _root("add_provider") == "add_provider"


def test_injects_missing_producer_dependency(no_index) -> None:
    no_index(set())  # nothing exists yet
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub("ui", reference_symbols=["api.add_provider"], dependencies=[]),
    ])
    violations = repair_and_validate_contracts(skel, "w")
    assert violations == []
    ui = next(s for s in skel.slices if s.id == "ui")
    assert ui.dependencies == ["api"]  # edge injected so producer runs first


def test_flags_name_mismatch_as_violation(no_index) -> None:
    no_index(set())
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub("ui", reference_symbols=["api.add_embedding_provider"]),
    ])
    violations = repair_and_validate_contracts(skel, "w")
    assert len(violations) == 1
    assert "add_embedding_provider" in violations[0]
    # No spurious dependency injected for the unmatched reference.
    assert next(s for s in skel.slices if s.id == "ui").dependencies == []


def test_external_library_symbols_never_flagged(no_index) -> None:
    no_index(set())
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub("ui", reference_symbols=["st.form", "st.session_state", "yaml.safe_load"]),
    ])
    assert repair_and_validate_contracts(skel, "w") == []


def test_existing_codebase_symbol_skipped(no_index) -> None:
    no_index({"add_llm_provider"})  # already in the codebase
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub("ui", reference_symbols=["UIApi.add_llm_provider"]),
    ])
    assert repair_and_validate_contracts(skel, "w") == []


def test_degrades_when_no_slice_declares_provides(no_index) -> None:
    no_index(set())
    skel = PlanSkeleton(slices=[
        _stub("api"),
        _stub("ui", reference_symbols=["api.add_embedding_provider"], dependencies=[]),
    ])
    # No `provides` anywhere → legacy behaviour: no violations, no mutation.
    assert repair_and_validate_contracts(skel, "w") == []
    assert next(s for s in skel.slices if s.id == "ui").dependencies == []


def test_cycle_guard_reports_instead_of_injecting(no_index) -> None:
    no_index(set())
    # Producer already depends on the consumer → injecting the reverse edge would
    # form a cycle; the validator must report it, not create the cycle.
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"], dependencies=["ui"]),
        _stub("ui", reference_symbols=["api.add_provider"], dependencies=[]),
    ])
    violations = repair_and_validate_contracts(skel, "w")
    assert len(violations) == 1
    assert "cycle" in violations[0].lower()
    assert next(s for s in skel.slices if s.id == "ui").dependencies == []


def test_contract_block_lists_dependency_provides() -> None:
    api = _stub("api", provides=["UIApi.add_provider", "UIApi.remove_provider"])
    ui = _stub("ui", dependencies=["api"])
    block = _contract_block(ui, [api, ui])
    assert "api creates" in block
    assert "UIApi.add_provider" in block and "UIApi.remove_provider" in block
    # A slice with no producer dependencies gets no block.
    assert _contract_block(api, [api, ui]) == ""


class TestInferMissingProvides:
    """A forgotten `provides` declaration is inferred, not parked on.

    Regression (run ad28d82e): the impl slice created
    ArtifactStore.artifact_exists with provides=[], the test slice referenced
    it with a full module-qualified name and depended on the impl slice —
    and the reference gate still rejected the plan two rounds running,
    escalating a spec_amendment park. The dependency edge plus the owner
    class's file make the producer unambiguous.
    """

    def _install_index(self, monkeypatch, *, existing_leaves, owner_files):
        monkeypatch.setattr(
            "spine.agents.plan_synthesis._symbol_exists_in_index",
            lambda db_path, sym: _leaf(sym) in existing_leaves,
        )
        monkeypatch.setattr(
            "spine.agents.plan_synthesis._symbol_files",
            lambda db_path, name: owner_files.get(name, []),
        )

    def test_provides_inferred_from_dependency_and_owner_file(self, monkeypatch):
        self._install_index(
            monkeypatch,
            existing_leaves={"ArtifactStore", "save_artifact"},
            owner_files={"ArtifactStore": ["spine/persistence/artifacts.py"]},
        )
        skel = PlanSkeleton(slices=[
            _stub("impl", target_files=["spine/persistence/artifacts.py"], provides=[]),
            _stub(
                "tests",
                target_files=["tests/unit/test_artifact_store_exists.py"],
                reference_symbols=[
                    "spine.persistence.artifacts.ArtifactStore.artifact_exists",
                    "spine.persistence.artifacts.ArtifactStore.save_artifact",
                ],
                dependencies=["impl"],
            ),
        ])
        violations = repair_and_validate_contracts(skel, "w")
        impl = next(s for s in skel.slices if s.id == "impl")
        assert impl.provides == ["ArtifactStore.artifact_exists"]
        assert violations == []

    def test_no_inference_without_dependency_edge(self, monkeypatch):
        self._install_index(
            monkeypatch,
            existing_leaves={"ArtifactStore"},
            owner_files={"ArtifactStore": ["spine/persistence/artifacts.py"]},
        )
        skel = PlanSkeleton(slices=[
            _stub("impl", target_files=["spine/persistence/artifacts.py"], provides=[]),
            _stub(
                "tests",
                reference_symbols=["ArtifactStore.artifact_exists"],
                dependencies=[],  # no declared edge — intent is ambiguous
            ),
        ])
        repair_and_validate_contracts(skel, "w")
        impl = next(s for s in skel.slices if s.id == "impl")
        assert impl.provides == []

    def test_no_inference_when_owner_unknown(self, monkeypatch):
        self._install_index(monkeypatch, existing_leaves=set(), owner_files={})
        skel = PlanSkeleton(slices=[
            _stub("impl", target_files=["spine/persistence/artifacts.py"], provides=[]),
            _stub(
                "tests",
                reference_symbols=["Ghost.method"],
                dependencies=["impl"],
            ),
        ])
        repair_and_validate_contracts(skel, "w")
        impl = next(s for s in skel.slices if s.id == "impl")
        assert impl.provides == []


class TestOwnerDeclaresAttribute:
    """Instance/class attributes are invisible to the symbol index; a
    reference like 'ArtifactStore._base' (assigned in __init__) must resolve
    against the owner's source instead of being flagged dangling (run
    1ed302ca: the false dangling paired with a scope exclusion and escalated
    a fabricated spec_amendment park)."""

    _OWNER_SRC = (
        "class Store:\n"
        "    LIMIT = 10\n"
        "    def __init__(self, base):\n"
        "        self._base = base\n"
        "    def load(self):\n"
        "        return self._base\n"
    )

    def _install(self, monkeypatch):
        monkeypatch.setattr(
            "spine.agents.tools.codebase_query.get_symbol_source",
            lambda db, root, name: self._OWNER_SRC if name == "Store" else None,
        )

    def test_instance_attribute_resolves(self, monkeypatch):
        from spine.agents.plan_synthesis import _owner_declares_attribute

        self._install(monkeypatch)
        assert _owner_declares_attribute("db", "Store._base")
        assert _owner_declares_attribute("db", "pkg.mod.Store._base")

    def test_class_attribute_resolves(self, monkeypatch):
        from spine.agents.plan_synthesis import _owner_declares_attribute

        self._install(monkeypatch)
        assert _owner_declares_attribute("db", "Store.LIMIT")

    def test_phantom_attribute_stays_dangling(self, monkeypatch):
        from spine.agents.plan_synthesis import _owner_declares_attribute

        self._install(monkeypatch)
        assert not _owner_declares_attribute("db", "Store.ghost")
        assert not _owner_declares_attribute("db", "Unknown._base")

    def test_usage_without_assignment_does_not_resolve(self, monkeypatch):
        # `return self._base` alone must not count — only assignments declare.
        from spine.agents.plan_synthesis import _owner_declares_attribute

        monkeypatch.setattr(
            "spine.agents.tools.codebase_query.get_symbol_source",
            lambda db, root, name: "class S:\n    def g(self):\n        return self.x\n",
        )
        assert not _owner_declares_attribute("db", "S.x")


# ── PHP symbol forms (run b15cee51: 21 framework false-violations × 3 rounds) ──


@pytest.fixture()
def php_classmap(tmp_path, monkeypatch):
    """A fake composer classmap in CWD with Laravel facade classes."""
    vend = tmp_path / "vendor" / "composer"
    vend.mkdir(parents=True)
    (vend / "autoload_classmap.php").write_text(
        "<?php return array(\n"
        "  'Illuminate\\\\Support\\\\Facades\\\\Schema' => '',\n"
        "  'Illuminate\\\\Support\\\\Facades\\\\DB' => '',\n"
        "  'Illuminate\\\\Support\\\\Facades\\\\Gate' => '',\n"
        "  'Illuminate\\\\Support\\\\Facades\\\\Storage' => '',\n"
        ");\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # The classmap-basenames cache keys on the resolved path, so a fresh
    # tmp_path never collides — but clear defensively.
    from spine.agents.plan_synthesis import _composer_class_basenames
    _composer_class_basenames.cache_clear()


def test_facade_static_calls_never_flagged(no_index, php_classmap) -> None:
    no_index(set())
    sk = PlanSkeleton(
        architecture_overview="a", testing_strategy="t",
        slices=[
            _stub("mig", target_files=["database/migrations/x.php"],
                  provides=["files table"],
                  reference_symbols=["Schema::create", "DB::raw", "$table->uuid"]),
        ],
    )
    assert repair_and_validate_contracts(sk, "w") == []


def test_bare_unprovided_identifiers_never_flagged(no_index) -> None:
    no_index(set())
    sk = PlanSkeleton(
        architecture_overview="a", testing_strategy="t",
        slices=[
            _stub("mig", target_files=["x.php"], provides=["files.table"],
                  reference_symbols=["constrained", "nullable", "config"]),
        ],
    )
    assert repair_and_validate_contracts(sk, "w") == []


def test_provided_bare_symbol_still_gets_dependency_edge(no_index) -> None:
    no_index(set())
    sk = PlanSkeleton(
        architecture_overview="a", testing_strategy="t",
        slices=[
            _stub("api", target_files=["api.py"], provides=["format_bytes"]),
            _stub("ui", target_files=["ui.py"], provides=["render"],
                  reference_symbols=["format_bytes"]),
        ],
    )
    assert repair_and_validate_contracts(sk, "w") == []
    ui = next(s for s in sk.slices if s.id == "ui")
    assert "api" in (ui.dependencies or [])


def test_framework_provides_do_not_create_phantom_producers(
    no_index, php_classmap
) -> None:
    """A slice listing 'Gate::authorize' in provides must not become a
    producer other slices are forced to depend on (run b15cee51: phantom
    cycle violations against test slices)."""
    no_index(set())
    sk = PlanSkeleton(
        architecture_overview="a", testing_strategy="t",
        slices=[
            _stub("test-a", target_files=["tests/a.php"],
                  provides=["Gate::authorize"],
                  reference_symbols=[]),
            _stub("ctrl", target_files=["app/Ctrl.php"], provides=["Ctrl.store"],
                  reference_symbols=["Gate::authorize"]),
        ],
    )
    assert repair_and_validate_contracts(sk, "w") == []
    ctrl = next(s for s in sk.slices if s.id == "ctrl")
    assert not ctrl.dependencies  # no phantom edge onto the test slice


def test_unprovable_references_never_flagged(no_index) -> None:
    # Run 5646d24c: 'console.print' (module-level variable root, invisible
    # to the index) and 'spine.cli' (importable internal module path) burned
    # five plan-critic rounds as gate false positives.
    no_index(set())
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub(
            "cli",
            reference_symbols=["console.print", "spine.cli"],
        ),
    ])
    assert repair_and_validate_contracts(skel, "w") == []


def test_plan_internal_roots_still_flag(no_index) -> None:
    # A root the plan itself defines (slice id) stays contract territory.
    no_index(set())
    skel = PlanSkeleton(slices=[
        _stub("api", provides=["UIApi.add_provider"]),
        _stub("ui", reference_symbols=["api.add_embedding_provider"]),
    ])
    violations = repair_and_validate_contracts(skel, "w")
    assert len(violations) == 1 and "add_embedding_provider" in violations[0]
