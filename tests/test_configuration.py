"""Resolve one scenario, propagate its values and reject stale run identities."""
from dataclasses import replace
import json

import pytest

from coupling.cases import load_configuration
from coupling.execution import prepare_case, run_case
from test_case_interfaces import slab_case


def write_study(tmp_path, *, scenarios=None):
    """Write a small study fixture with shared references and overrides.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    scenarios : mapping or None, optional
        Optional scenario definitions replacing the default two-scenario fixture.

    Returns
    -------
    pathlib.Path
        Written study JSON path.
    """
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"height_cm": 3., "limits": [1., 8.]}))
    record = {
        "schema_version": 1,
        "data": {"model": "model.json"},
        "defaults": {
            "cycle": {"nested": {"left": 2., "right": 4.}, "bounds": {
                "$ref": "#/settings/optimization/bounds"}},
            "optimization": {"bounds": {"$ref": "#/data/model/limits"}},
        },
        "scenarios": scenarios or {"1": {}, "2": {"cycle": {"nested": {"right": 7.}},
                                                  "optimization": {"bounds": [2., 9.]}}},
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(record))
    return path


def test_shared_inputs_resolve_after_scenario_overrides_and_return_isolated_values(tmp_path):
    """Verify shared inputs resolve after scenario overrides and return isolated values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    configuration = load_configuration(write_study(tmp_path), 2)
    assert configuration.settings("cycle") == {"nested": {"left": 2., "right": 7.}, "bounds": [2., 9.]}
    assert configuration.settings("optimization")["bounds"] == [2., 9.]
    assert configuration.data("model")["height_cm"] == 3.
    settings = configuration.settings("cycle")
    settings["nested"]["left"] = -100.
    resolved = configuration.resolved()
    resolved["data"]["model"]["height_cm"] = 99.
    assert configuration.settings("cycle")["nested"]["left"] == 2.
    assert configuration.data("model")["height_cm"] == 3.
    assert configuration.resolved()["scenario"] == "2"


def test_commented_toml_resolves_shared_goals_and_json_source_data(tmp_path):
    """Verify commented toml resolves shared goals and JSON source data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    (tmp_path / "model.json").write_text('{"height_cm": 3.0}')
    path = tmp_path / "study.toml"
    path.write_text('''# Editable goals, with explicit units in their names.
schema_version = 1
[data]
model = "model.json"
[defaults.cycle]
power_W = 35_000_000
[defaults.cycle.objective]
minimum_keff = 1.002
[defaults.optimization]
target = { "$ref" = "#/settings/cycle/objective/minimum_keff" }
[scenarios."2".cycle.objective]
minimum_keff = 1.01
''')
    configuration = load_configuration(path, 2)
    assert configuration.settings("optimization")["target"] == 1.01
    assert configuration.settings("cycle")["power_W"] == 35000000
    assert configuration.data("model")["height_cm"] == 3.


def test_unknown_input_format_is_rejected(tmp_path):
    """Verify unknown input format is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    path = tmp_path / "study.txt"
    path.write_text("{}")
    with pytest.raises(ValueError, match="must be TOML or JSON"):
        load_configuration(path, 1)


def test_reference_replaces_an_entire_default_object(tmp_path):
    """Verify reference replaces an entire default object.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    path = write_study(tmp_path, scenarios={"1": {"cycle": {
        "nested": {"$ref": "#/data/model"}}}})
    configuration = load_configuration(path, 1)
    assert configuration.settings("cycle")["nested"] == {"height_cm": 3., "limits": [1., 8.]}


@pytest.mark.parametrize("reference, error, message", [
    ({"$ref": "#/missing"}, KeyError, "missing"),
    ({"$ref": "#/settings/cycle/nested"}, ValueError, "Cyclic configuration reference"),
    ({"$ref": "model.json"}, ValueError, "must start with"),
    ({"$ref": "#/data/model", "extra": 1}, ValueError, "cannot contain other fields"),
])
def test_invalid_references_fail_before_building_a_model(tmp_path, reference, error, message):
    """Verify invalid references fail before building a model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    reference : dict
        Malformed configuration reference used to exercise fail-fast loading.
    error : type[Exception]
        Native lookup error or explicit reference-contract error.
    message : str
        Expected diagnostic identifying the invalid input.
    """
    path = write_study(tmp_path, scenarios={"1": {"cycle": {"nested": reference}}})
    with pytest.raises(error, match=message):
        load_configuration(path, 1)


def test_unknown_scenario_and_modified_source_are_rejected(tmp_path):
    """Verify unknown scenario and modified source are rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    path = write_study(tmp_path)
    with pytest.raises(KeyError, match="3"):
        load_configuration(path, 3)
    configuration = load_configuration(path, 1)
    configuration.verify_sources()
    (tmp_path / "model.json").write_text('{"height_cm": 4.0, "limits": [1.0, 8.0]}')
    with pytest.raises(ValueError, match="changed after loading"):
        configuration.verify_sources()


def configured_slab(tmp_path):
    """Bind the slab fixture through the resolved configuration interface.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.

    Returns
    -------
    CycleCase
        Slab case using one canonical study definition instead of separate input files.
    """
    case = slab_case(tmp_path)
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"schema_version": 1,
        "defaults": {kind: case.settings(kind) for kind in case.inputs},
        "scenarios": {"1": {}, "2": {}}}))
    return replace(case, inputs=None, configuration=load_configuration(path, 1))


def test_preparation_records_the_resolved_inputs_and_refuses_stale_configuration(tmp_path):
    """Verify preparation records the resolved inputs and refuses stale configuration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = configured_slab(tmp_path)
    directory = prepare_case(case)
    recorded = json.loads((directory / "resolved_inputs.json").read_text())
    assert recorded == case.definition()
    assert recorded["settings"]["cycle"]["design"] == case.make_case()["design"]
    assert (directory / "slab.txt").read_text() == "20.0"
    case.configuration.path.write_text(case.configuration.path.read_text() + "\n")
    prepare_case(case)
    changed = json.loads(case.configuration.path.read_text())
    changed["defaults"]["cycle"]["design"]["width_cm"] = [3.]
    case.configuration.path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed after loading"):
        prepare_case(case)


def test_run_identity_includes_the_scenario_even_when_values_match(tmp_path):
    """Verify run identity includes the scenario even when values match.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = configured_slab(tmp_path)
    second = replace(case, configuration=load_configuration(case.configuration.path, 2))
    assert case.settings() == second.settings()
    assert case.definition() != second.definition()
    case.prepare_outputs()
    case.result("optimization").write_text(json.dumps({
        "case_definition": case.definition()}))
    with pytest.raises(ValueError, match="contains a different case"):
        run_case(second)


def test_case_cannot_supply_competing_input_paths(tmp_path):
    """Verify case cannot supply competing input paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = configured_slab(tmp_path)
    with pytest.raises(ValueError, match="conflict with the resolved configuration"):
        replace(case, inputs={"cycle": tmp_path / "different.json"})


def test_wr1_model_uses_the_declared_geometry_materials_and_goal(tmp_path):
    """Verify WR-1 model uses the declared geometry materials and goal.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    import shutil
    from designs.wr_1.model.case import study
    from designs.wr_1 import WR1_ROOT

    shutil.copytree(WR1_ROOT / "inputs", tmp_path / "inputs")
    source = tmp_path / "inputs/study.toml"
    source.write_text(source.read_text().replace("assembly_height_mm = 990.0", "assembly_height_mm = 1000.0")
                      .replace("minimum_keff = 1.002", "minimum_keff = 1.01"))
    model_path = tmp_path / "inputs/model.toml"
    model_path.write_text(model_path.read_text().replace(
        "[materials.moderator]\ntemperature_K = 327.15", "[materials.moderator]\ntemperature_K = 340.0"))
    case = study(1, root=tmp_path, work=tmp_path / "work")
    native = case.make_case()
    assert native["geometry"]["upper_cm"][2] == 300.
    assert {region["temperature_K"] for region in native["regions"]
            if region["kind"] == "wr1_background"} == {340.}
    assert case.settings("optimization")["native_minimum_keff"] == 1.01
    assert case.settings("optimization")["surrogate_keff_target"] == 1.01


@pytest.mark.parametrize("step", [4, 5, 6])
def test_graphite_mass_density_controls_the_native_material(tmp_path, step):
    """Verify graphite mass density controls the native material.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    step : int
        WR-1 graphite scenario number under test.
    """
    import shutil
    from designs.wr_1.model.case import study
    from designs.wr_1 import WR1_ROOT

    shutil.copytree(WR1_ROOT / "inputs", tmp_path / "inputs")
    source = tmp_path / "inputs/study.toml"
    source.write_text(source.read_text().replace("density_g_cm3 = 1.7", "density_g_cm3 = 1.8"))
    original = study(step).make_case()
    modified = study(step, root=tmp_path, work=tmp_path / "work").make_case()
    assert modified["geometry"] == original["geometry"]
    for before, after in zip(original["regions"], modified["regions"]):
        if before.get("equivalence_material") == "graphite":
            assert after["atom_density"]["C0"] == pytest.approx(before["atom_density"]["C0"] * 1.8 / 1.7)
