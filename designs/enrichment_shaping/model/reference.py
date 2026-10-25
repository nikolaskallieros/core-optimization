"""Use material calibration, geometry derivatives and native verification to size the lattice."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch

from coupling.openmc import SteadyStateTransport
from optimization.constrained import constrained_minimize
from optimization.design import Snapshot, optimize
from optimization.fitting import Target
from runtime.io import save_json, save_state
from runtime.paths import RUNS

from .case import DESIGN_ROOT, INPUTS, prepare, settings
from .calibration import fit_is_acceptable, fitting_reference, make_size_surrogate
from .design import LatticeSize
from .optimization import reuse_reference


def prepare_reference(reference_directory=None, *, initial_scale=1.):
    """Find critical dimensions through the shared calibrated design workflow.

    Parameters
    ----------
    reference_directory : str or pathlib.Path or None, optional
        Explicit native reference at initial_scale, reused only after comparing
        its complete physical and sampling inputs.
    initial_scale : float, optional
        Common transverse scale relative to the current TOML dimensions.

    Returns
    -------
    pathlib.Path
        Selected native calculation containing baseline.json. Its physical
        pitches and fuel radius also replace the corresponding TOML values.
    """
    config = settings()
    options = config["critical_reference"]
    target_options = config["optimization"]
    tolerance = target_options["keff_absolute_tolerance"] - target_options["prediction_margin"]
    torch.set_num_threads(config["openmc"]["threads"])
    directory = RUNS / DESIGN_ROOT.name / ("critical_surrogate_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    (directory / "initial_model.toml").write_text(INPUTS.read_text())
    (directory / "goals.md").write_text((DESIGN_ROOT / "goals.md").read_text())
    print(f"Critical geometry workflow: {directory}", flush=True)
    design = LatticeSize(config, initial_scale)
    evaluations = []
    step_limit = options["scale_step_limit"]

    def detailed(current, previous):
        """Evaluate the shared CAD at one proposed physical size.

        Parameters
        ----------
        current : LatticeSize
            Live physical size; enrichment remains uniform at 4 wt%.
        previous : Snapshot or None
            Previously accepted native reference.

        Returns
        -------
        Snapshot
            Actual native criticality, spatial powers and material observations.
        """
        resolved = current.configuration(native=True)
        work = directory / f"size_{len(evaluations):02d}"
        work.mkdir()
        save_json(work / "config.json", resolved)
        cad, _, _, _, case = prepare(resolved)
        if not evaluations and reference_directory is not None:
            retained = Path(reference_directory)
            state = reuse_reference(case, retained / "baseline.json")
        else:
            retained = work
            native = SteadyStateTransport(case, cad, work / "geometry")
            state = native([region["atom_density"] for region in case["regions"]], work / "native_00")
        save_state(work / "baseline.json", state)
        value, error = state["result"]["combined_keff"]
        metadata = {"index": len(evaluations), "scale": float(current.radial_scale.detach()),
                    "keff": value, "standard_error": error, "geometry": resolved["geometry"],
                    "directory": str(retained), "record_directory": str(work)}
        evaluations.append(metadata)
        save_json(directory / "evaluations.json", evaluations)
        print(f"Critical geometry native {metadata['index']}: scale={metadata['scale']:.6f}, k={value:.6f} +/- {error:.6f}", flush=True)
        return Snapshot({"keff": torch.tensor(value, dtype=torch.float64)},
                        fitting_reference(config, state), state=state, metadata=metadata)

    def critical(snapshot):
        """Apply the native criticality band including statistical uncertainty.

        Parameters
        ----------
        snapshot : Snapshot
            Independently sampled native observation.

        Returns
        -------
        bool
            Whether the observation satisfies the reference preparation band.
        """
        point = snapshot.metadata
        return abs(point["keff"] - 1.) + target_options["native_sigma_multiplier"] * point["standard_error"] <= tolerance

    def search(current, model, objective):
        """Use the surrogate's geometry gradient to propose a bounded critical size.

        Parameters
        ----------
        current : LatticeSize
            One live geometry parameter.
        model : CalibratedDiffusion
            Material-calibrated diffusion with a moving physical mesh.
        objective : callable
            Squared departure of k-effective from unity.

        Returns
        -------
        dict
            Shared constrained optimizer report, including the geometry gradient.
        """
        origin = float(current.radial_scale.detach())
        low, high = options["radial_scale_bounds"]
        return constrained_minimize(current, model, objective,
            bounds=(max(low, origin - step_limit), min(high, origin + step_limit)),
            constraints=lambda output: current.radial_scale.new_empty(0),
            constraint_bounds=([], []), steps=target_options["iterations"], ftol=1e-12)

    def accept(current, candidate, proposal):
        """Accept native progress toward criticality and shrink a rejected size step.

        Parameters
        ----------
        current, candidate : Snapshot
            Actual native observations at the two physical sizes.
        proposal : dict
            Shared optimizer's saved surrogate prediction.

        Returns
        -------
        dict
            Acceptance decision and the next permitted numerical step.
        """
        nonlocal step_limit
        accepted = bool(critical(candidate) or
                        abs(candidate.metadata["keff"] - 1.) < abs(current.metadata["keff"] - 1.))
        if not accepted:
            step_limit /= 2
        return {"accepted": accepted, "next_step_limit": step_limit}

    def progress(event, step, payload):
        """Retain each completed calibration and proposed geometry.

        Parameters
        ----------
        event : str
            Shared workflow event name.
        step : int
            Zero-based fit or proposal index.
        payload : dict
            Report or proposal supplied by the shared design loop.
        """
        if event == "fit":
            save_json(directory / f"fit_{step:02d}.json", payload["report"])
            print(f"Critical geometry: material fit {step} completed", flush=True)
        else:
            record = payload if event == "proposal" else payload["proposal"]
            save_json(directory / f"proposal_{step:02d}.json", record)

    answer = optimize(design, detailed,
        lambda current, reference: make_size_surrogate(config, current, reference.state),
        [Target("keff", transform="identity"), Target("calibration_power_W")],
        lambda output: (output["keff"] - 1.).square(),
        steps=options["max_updates"], accept=accept, search=search, stop=critical,
        callback=progress, fit_accept=lambda report: fit_is_acceptable(config, report),
        fit_options={"steps": config["calibration"]["iterations"], "jacobian_mode": "reverse"})
    save_json(directory / "optimization.json", {key: answer[key] for key in
              ("fits", "proposals", "selected_fit", "design_parameters")})
    selected = answer["snapshot"].metadata
    if not critical(answer["snapshot"]):
        raise RuntimeError("The native-verified critical geometry search exhausted its update budget")

    model = answer["model"]
    output = model()
    derivative, = torch.autograd.grad(output["keff"], design.radial_scale)
    origin = design.radial_scale.detach().clone()
    increment = options["gradient_step"]
    with torch.no_grad():
        design.radial_scale.copy_(origin + increment)
        upper = float(model()["keff"])
        design.radial_scale.copy_(origin - increment)
        lower = float(model()["keff"])
        design.radial_scale.copy_(origin)
    finite_difference = (upper - lower) / (2 * increment)
    adjoint = float(derivative)
    np.testing.assert_allclose(adjoint, finite_difference,
        rtol=config["verification"]["gradient_relative_tolerance"], atol=1e-8)
    verification = {"parameter": "radial_scale", "implicit_derivative": adjoint,
                    "finite_difference": finite_difference, "step": increment,
                    "relative_error": abs(adjoint - finite_difference) / max(abs(adjoint), abs(finite_difference))}
    save_json(directory / "gradient_verification.json", verification)
    text = INPUTS.read_text()
    for key in ("pitch_cm", "fuel_radius_cm"):
        text = re.sub(rf"(?m)^{key} = .+$", f"{key} = {json.dumps(selected['geometry'][key])}", text, count=1)
    INPUTS.write_text(text)
    (directory / "selected_model.toml").write_text(text)
    save_json(directory / "summary.json", {"status": "critical_reference_verified",
        "selected": selected, "native": evaluations, "target_keff": 1.,
        "allowed_distance": tolerance, "gradient_verification": verification})
    return Path(selected["directory"])

