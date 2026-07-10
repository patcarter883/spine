"""AC-author grounding with sibling conventions for created files.

Probe 20 (run 8eaa5887): the editor's sibling exemplar made the generated
factory correctly ``extend BaseFactory`` (the repo convention), while the
AC author — who never saw a sibling — demanded the framework-default
``Illuminate\\Database\\Factories\\Factory``. A correct implementation
failed verification three cycles running. Author and editor must see the
same ground truth.
"""

from spine.agents.plan_synthesis import SliceStub, _new_file_convention_block


def _stub(**kw):
    kw.setdefault("id", "create-unit-of-measure-factory")
    kw.setdefault("title", "Create factory")
    kw.setdefault("summary", "Create the UnitOfMeasure model factory.")
    return SliceStub(**kw)


def _repo(tmp_path):
    d = tmp_path / "database" / "factories"
    d.mkdir(parents=True)
    (d / "FarmFactory.php").write_text(
        "<?php\n\nnamespace Database\\Factories;\n\n"
        "class FarmFactory extends BaseFactory\n{\n"
        "    protected $model = Farm::class;\n}\n",
        encoding="utf-8",
    )
    return tmp_path


def test_new_file_gets_sibling_convention(tmp_path):
    ws = _repo(tmp_path)
    stub = _stub(target_files=["database/factories/UnitOfMeasureFactory.php"])
    block = _new_file_convention_block(stub, str(ws))
    assert "extends BaseFactory" in block
    assert "CONSISTENT with these existing siblings" in block
    assert "do NOT prescribe framework defaults" in block


def test_existing_target_file_gets_no_block(tmp_path):
    ws = _repo(tmp_path)
    stub = _stub(target_files=["database/factories/FarmFactory.php"])
    assert _new_file_convention_block(stub, str(ws)) == ""


def test_no_sibling_no_block(tmp_path):
    (tmp_path / "app").mkdir()
    stub = _stub(target_files=["app/Lonely.php"])
    assert _new_file_convention_block(stub, str(tmp_path)) == ""


def test_no_workspace_root_no_block():
    stub = _stub(target_files=["database/factories/UnitOfMeasureFactory.php"])
    assert _new_file_convention_block(stub, "") == ""
