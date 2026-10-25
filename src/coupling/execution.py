"""Prepare and run a case through the same public interfaces used by designs."""

import json

import numpy as np
import torch

from optimization.design import parameters
from runtime.io import save_json
from runtime.io import sha


def prepare_case(case):
    """Validate the declared design and export native inputs.

    Parameters
    ----------
    case : CycleCase
        Resolved case providing model factories, settings and result paths.

    Returns
    -------
    pathlib.Path
        Prepared-input directory containing resolved settings, case identity and any native
        XML inputs.

    Raises
    ------
    ValueError
        Parameters, bounds, input sources, constraints or cycle definitions disagree.
    ArithmeticError
        Native and surrogate region volumes disagree.
    """
    if case.configuration is not None:
        case.configuration.verify_sources()
    settings = case.settings()
    options = case.settings("optimization")
    design = case.design()
    values = parameters(design)
    if set(values) != set(settings["design"]) or set(values) != set(options["bounds"]):
        raise ValueError("Design parameters, scenario values and optimization bounds differ")
    for name, value in values.items():
        if not np.array_equal(value, settings["design"][name]):
            raise ValueError(f"The design factory does not use the declared initial parameter {name}")
        lower, upper = options["bounds"][name]
        if not np.isfinite(value).all() or np.any(np.asarray(value) < lower) or np.any(np.asarray(value) > upper):
            raise ValueError(f"Initial parameter {name} is outside its declared bounds")
    if options["bounds"]["cycle_days"] != settings["refueling"]["cycle_days_bounds"]:
        raise ValueError("Native and optimizer cycle-duration bounds differ")
    if settings["differentiated_cycles"] != list(range(1, settings["cycles"] + 1)):
        raise ValueError("Every simulated cycle must participate in the residence derivative")
    if settings["objective_cycle"] != settings["cycles"]:
        raise ValueError("This residence objective must select the final simulated cycle")
    native = case.make_case(design_values=values)
    with torch.no_grad():
        mesh = case.model.make_mesh(design, native)
        volumes = None if mesh is None else (mesh.fractions().T @ mesh.grid.volumes).numpy()
    if volumes is not None:
        expected = np.asarray([region["volume_cm3"] for region in native["regions"]])
        if not np.allclose(volumes, expected, rtol=mesh.volume_rtol, atol=1e-8):
            raise ArithmeticError(f"Native and surrogate region volumes differ beyond volume_rtol={mesh.volume_rtol:g}")
    if "fresh_fuel_mass" in options:
        from optimization.fuel_constraints import native_fuel_mass_check
        if not native_fuel_mass_check(native, options["fresh_fuel_mass"])["passed"]:
            raise ValueError("The initial loading exceeds the fresh fuel mass ceiling")
    directory = case.work / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    if case.model.export_transport is not None:
        case.model.export_transport(native, directory)
    save_json(directory / "resolved_inputs.json", case.definition())
    save_json(directory / "case.json", native)
    return directory


def run_case(case):
    """Prepare evidence, calibration and a bounded optimization search.

    Parameters
    ----------
    case : CycleCase
        Resolved case providing model factories, settings and result paths.

    Notes
    -----
    Completed evidence is reused only when its settings and case definitions match.
    This function can launch native transport and the configured optimization budget.
    """
    from coupling.calibration import calibration_identity
    from coupling.history import prepare_baseline, check_replay, check_sampling
    from coupling.optimization import run_search

    prepare_case(case)
    options = case.settings("optimization")
    record_path = case.result("optimization")
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    if record is not None:
        if record.get("case_definition") != case.definition() or record.get("search_id") != options.get("search_id"):
            raise ValueError("The run directory contains a different case; choose a new output directory")
        return run_search(case)
    if not case.result("history").is_file():
        prepare_baseline(case, native_only=True)
    else:
        manifest = json.loads((case.result("states", "") / "manifest.json").read_text())
        if manifest.get("case_definition") != case.definition():
            raise ValueError("The native baseline uses different inputs; choose a new output directory")
    for kind, check in (("sampling_check", check_sampling), ("replay_checks", check_replay)):
        path = case.result(kind)
        previous = json.loads(path.read_text()) if path.is_file() else {}
        if (previous.get("status") != "checked" or previous.get("history_sha256") != sha(case.result("history"))
                or previous.get("settings") != case.settings()):
            check(case)
    fit_path = case.result("fit")
    fit = json.loads(fit_path.read_text()) if fit_path.is_file() else {}
    checks_path = case.result("gradient_checks")
    checks = json.loads(checks_path.read_text()) if checks_path.is_file() else {}
    if (any(fit.get(key) != value for key, value in calibration_identity(case).items())
            or checks.get("calibration_sha256") != (sha(fit_path) if fit_path.is_file() else None)
            or checks.get("optimization_settings") != case.settings("optimization")
            or not checks.get("gradient_graph_passed")):
        prepare_baseline(case, reuse_baseline=True)
    run_search(case)
