"""Power-shaping goals and native acceptance policies for this physical design."""

from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np
import torch

from coupling.openmc import SteadyStateTransport
from optimization.constrained import constrained_minimize
from optimization.design import Snapshot, optimize, set_parameters
from optimization.fitting import Target, selected_fit
from optimization.objectives import relative_power_density, smooth_maximum
from runtime.io import load_state, save_json, save_state, serializable
from runtime.paths import RUNS

from .case import DESIGN_ROOT, INPUTS, compositions, prepare, settings
from .design import Enrichment
from .calibration import fitting_reference, make_surrogate, fit_is_acceptable


def reuse_reference(case, path):
    """Read a native reference only when its physical and sampling inputs match.

    Parameters
    ----------
    case : dict
        Current native inputs derived from the single physical configuration.
    path : pathlib.Path
        Saved native state, with its associated numerical array file.

    Returns
    -------
    dict
        Matching native case and results; mesh-integration roundoff in volume
        accounting is allowed while the complete CAD definition must agree.
    """
    state = load_state(path)
    expected = deepcopy(case)
    for current, retained in zip(expected["regions"], state["case"]["regions"], strict=True):
        np.testing.assert_allclose(current["volume_cm3"], retained["volume_cm3"], rtol=1e-12)
        current["volume_cm3"] = retained["volume_cm3"]
    if serializable(expected) != serializable(state["case"]):
        raise ValueError("The retained native reference has different physical or sampling inputs")
    return state


def run(reference_directory=None, *, prepare_only=False, initial_fit=None,
        surrogate_checkpoint=None, continuation_directory=None, callback=None):
    """Prepare or optimize the declared steady-state case in a fresh run folder.

    Parameters
    ----------
    reference_directory : str or pathlib.Path or None, optional
        Explicit prior run containing baseline.json to reuse after comparing
        its actual inputs. None performs a new uniform native calculation.
    prepare_only : bool, optional
        Prepare and retain the uniform native reference without changing any
        design parameters.
    initial_fit : dict or None, optional
        Material coefficient record for warming the first calibration. The
        shared fitter still checks its accuracy against the current native state.
    surrogate_checkpoint : str or pathlib.Path or None, optional
        Saved first-proposal iterate with matching native reference inputs.
        Resume its remaining iteration budget before native verification.
    continuation_directory : str or pathlib.Path or None, optional
        Prior summary whose accepted parameters and native state should continue.
        Original targets, inventory, evaluations and decisions are preserved.
    callback : callable or None, optional
        Receive the current summary after every native decision and at completion,
        for plotting directly from saved observations.

    Returns
    -------
    dict
        Effective configuration, native evaluations, selected design, fitting
        and optimization histories, gradients and measured elapsed time.
    """
    started = time.perf_counter()
    previous = None
    if continuation_directory is not None:
        continuation_directory = Path(continuation_directory)
        previous = json.loads((continuation_directory / "summary.json").read_text())
        reference_directory = continuation_directory
    config = settings()
    options = config["optimization"]
    if not prepare_only:
        reference_mode = options["keff_reference"]
    torch.set_num_threads(config["openmc"]["threads"])
    directory = RUNS / DESIGN_ROOT.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    (directory / "model.toml").write_text(INPUTS.read_text())
    (directory / "goals.md").write_text((DESIGN_ROOT / "goals.md").read_text())
    print(f"Run: {directory}", flush=True)
    cad, mesh, fractions, initial, case = prepare(config)
    volumes = (fractions.T @ mesh.grid.volumes)[:-1]
    design = Enrichment(config, initial, volumes)
    initial_mass = float(design.u235_mass_kg().detach())
    points = []
    initial_state = (None if reference_directory is None else
                     reuse_reference(case, Path(reference_directory) / "baseline.json"))
    pending_checkpoint = None
    if surrogate_checkpoint is not None:
        surrogate_checkpoint = Path(surrogate_checkpoint)
        reuse_reference(case, surrogate_checkpoint.parent / "baseline.json")
        pending_checkpoint = json.loads(surrogate_checkpoint.read_text())
    native = SteadyStateTransport(case, cad, directory / "geometry")

    def detailed(current, previous):
        """Evaluate one new composition with independent continuous-energy transport.

        Parameters
        ----------
        current : Enrichment
            Current physical enrichment values.
        previous : Snapshot or None
            Prior accepted native reference.

        Returns
        -------
        Snapshot
            Actual native powers, criticality, composition and calibration targets.
        """
        index = len(points)
        work = directory / f"native_{index:02d}"
        if index == 0 and initial_state is not None:
            state = initial_state
            work = Path(reference_directory) / "native_00"
        else:
            sampling = dict(config["openmc"], seed=config["openmc"]["seed"] + index)
            state = native(compositions(config, current.enrichment_wt_percent), work, sampling=sampling)
        save_state(directory / f"native_{index:02d}.json", state)
        if index == 0:
            save_state(directory / "baseline.json", state)
        power = torch.as_tensor(state["result"]["observables"]["region_power_W"]["mean"][:-1], dtype=torch.float64)
        relative = relative_power_density(power, volumes)
        keff, error = state["result"]["combined_keff"]
        outputs = {"keff": torch.tensor(keff, dtype=torch.float64), "power_W": power,
                   "relative_power_density": relative, "u235_mass_kg": current.u235_mass_kg().detach()}
        point = {"index": index, "directory": str(work), "keff": keff, "standard_error": error,
                 "power_W": power.tolist(), "relative_power_density": relative.tolist(),
                 "peaking_factor": float(relative.max()), "u235_mass_kg": float(outputs["u235_mass_kg"]),
                 "enrichment_wt_percent": current.enrichment_wt_percent.detach().tolist()}
        points.append(point)
        save_json(directory / "native_history.json", points)
        print(f"Native {index}: k={keff:.6f} +/- {error:.6f}; peak/average={point['peaking_factor']:.4f}", flush=True)
        return Snapshot(outputs, fitting_reference(config, state), state=state, metadata=point)

    with torch.no_grad():
        baseline = detailed(design, None)
    current_reference = baseline
    prior_proposals, prior_fits = [], []
    selected_index = 0
    if previous is not None:
        points[:] = deepcopy(previous["native"])
        baseline.metadata = points[0]
        selected_index = previous["selected_index"]
        point = points[selected_index]
        set_parameters(design, {"enrichment_wt_percent": point["enrichment_wt_percent"]})
        expected = deepcopy(case)
        state_path = continuation_directory / f"native_{selected_index:02d}.json"
        retained = load_state(state_path)
        expected["sampling"] = retained["case"]["sampling"]
        for region, atoms in zip(expected["regions"], compositions(config, design.enrichment_wt_percent), strict=True):
            region["atom_density"] = atoms
            if region["depletable"]:
                region["fresh_fuel_atom_density"] = atoms
        retained = reuse_reference(expected, state_path)
        current_reference = Snapshot(
            {"keff": torch.tensor(point["keff"], dtype=torch.float64),
             "power_W": torch.tensor(point["power_W"], dtype=torch.float64),
             "relative_power_density": torch.tensor(point["relative_power_density"], dtype=torch.float64),
             "u235_mass_kg": design.u235_mass_kg().detach()},
            fitting_reference(config, retained), state=retained, metadata=point)
        prior_proposals = deepcopy(previous.get("proposals", []))
        prior_fits = deepcopy(previous.get("fits", [])[:len(prior_proposals)])
        if initial_fit is None and previous.get("fits"):
            initial_fit = selected_fit(previous["fits"][-1])
        for item in points:
            save_state(directory / f"native_{item['index']:02d}.json",
                       load_state(continuation_directory / f"native_{item['index']:02d}.json"))
        for index, proposal in enumerate(prior_proposals):
            save_json(directory / f"proposal_{index:02d}.json", proposal)
        for index, report in enumerate(prior_fits):
            save_json(directory / f"fit_{index:02d}.json", report)
        save_json(directory / "native_history.json", points)
        print(f"Continuing accepted native {selected_index}: peak/average={point['peaking_factor']:.4f}", flush=True)
    result = {"directory": str(directory), "config": config, "native": points,
              "selected_index": selected_index, "status": "baseline_completed",
              "proposals": list(prior_proposals), "fits": list(prior_fits),
              "continuation_directory": None if continuation_directory is None else str(continuation_directory),
              "surrogate_checkpoint": None if surrogate_checkpoint is None else str(surrogate_checkpoint)}
    if not prepare_only:
        target = {"baseline": baseline.metadata["keff"], "critical": 1.}[reference_mode]
        reference_error = baseline.metadata["standard_error"] if reference_mode == "baseline" else 0.
        allowed = options["keff_absolute_tolerance"]
        step_limit = (prior_proposals[-1]["next_step_limit_wt_percent"] if prior_proposals
                      else options["enrichment_step_limit_wt_percent"])
        result.update(status="running", target_keff=target)
        offset = len(prior_proposals)
        if reference_mode == "critical" and (abs(baseline.metadata["keff"] - target)
                + options["native_sigma_multiplier"] * baseline.metadata["standard_error"] > allowed):
            raise RuntimeError("Establish a native critical reference before enrichment optimization")

        def search(current, model, objective):
            """Keep physical inventory and criticality bounded during every proposal.

            Parameters
            ----------
            current : Enrichment
                Independent enrichment variables.
            model : FreshFuelSurrogate
                Current fitted reference, frozen during the design search.
            objective : callable
                Scalar smooth power-density maximum.

            Returns
            -------
            dict
                Shared constrained-search result and its complete gradient vector.
            """
            nonlocal pending_checkpoint
            origin = current.enrichment_wt_percent.detach().numpy().ravel().copy()
            low, high = config["materials"]["enrichment_bounds_wt_percent"]
            previous_iterations = 0 if pending_checkpoint is None else pending_checkpoint["iteration"]
            iteration = -1
            trajectory = []

            def checkpoint(record):
                """Retain the latest surrogate iterate before native verification.

                Parameters
                ----------
                record : dict
                    Physical enrichment vector, objective and constraint values.
                """
                nonlocal iteration
                if record["phase"] == "optimization":
                    iteration += 1
                trajectory.append(record)
                save_json(directory / f"surrogate_{len(points):02d}_progress.json",
                          {"iteration": previous_iterations + iteration,
                           "previous_iterations": previous_iterations, **record, "trace": trajectory})

            answer = constrained_minimize(current, model, objective,
                bounds=(np.maximum(low, origin - step_limit), np.minimum(high, origin + step_limit)),
                constraints=lambda output: torch.stack((output["u235_mass_kg"] / initial_mass - 1., output["keff"])),
                constraint_bounds=([0., target], [0., target]),
                steps=max(1, options["iterations"] - previous_iterations), ftol=options["ftol"],
                callback=checkpoint, allow_iteration_limit=True,
                starting_point=None if pending_checkpoint is None else pending_checkpoint["parameters"])
            pending_checkpoint = None
            answer["previous_iterations"] = previous_iterations
            answer["proposed"] = answer["improved"]
            if not answer["proposed"]:
                answer["stop_reason"] = "no_improving_feasible_diffusion_candidate"
            return answer

        def accept(current, candidate, proposal):
            """Accept native power improvement only when physical constraints also pass.

            Parameters
            ----------
            current, candidate : Snapshot
                Actual independently sampled native observations.
            proposal : dict
                Shared optimizer's recorded proposal.

            Returns
            -------
            dict
                Native acceptance and explicit inventory/criticality checks. Rejection
                halves the next numerical enrichment step without changing the goals.
            """
            nonlocal step_limit
            point = candidate.metadata
            mass_error = abs(point["u235_mass_kg"] / initial_mass - 1)
            interval = abs(point["keff"] - target) + options["native_sigma_multiplier"] * np.hypot(point["standard_error"], reference_error)
            accepted = bool(point["peaking_factor"] < current.metadata["peaking_factor"]
                            and mass_error <= config["verification"]["inventory_relative_tolerance"]
                            and interval <= allowed)
            if not accepted:
                step_limit /= 2
            return {"accepted": accepted, "inventory_relative_error": mass_error,
                    "native_keff_interval_distance": float(interval), "next_step_limit_wt_percent": step_limit,
                    "native_index": point["index"], "starting_native_index": current.metadata["index"],
                    "power_improved": bool(point["peaking_factor"] < current.metadata["peaking_factor"])}

        def progress(event, step, payload):
            """Retain completed fits and native decisions throughout the study.

            Parameters
            ----------
            event : str
                Shared optimization event.
            step : int
                Zero-based fitting or proposal index.
            payload : dict
                Current fitting report or proposal and native decision.
            """
            step += offset
            if event == "fit":
                payload["report"]["native_index"] = payload["snapshot"].metadata["index"]
                result["fits"].append(payload["report"])
                save_json(directory / f"fit_{step:02d}.json", payload["report"])
                save_json(directory / "summary.json", result)
                print(f"Enrichment: material fit {step} completed", flush=True)
            else:
                record = payload if event == "proposal" else payload["proposal"]
                record["step"] = step + 1
                save_json(directory / f"proposal_{step:02d}.json", record)
                if event == "decision":
                    result["proposals"].append(record)
                    result["selected_index"] = payload["snapshot"].metadata["index"]
                    save_json(directory / "summary.json", result)
                    if callback is not None:
                        callback(result)
                    print(f"Enrichment proposal {step + 1}: accepted={record['accepted']}", flush=True)

        answer = optimize(design, detailed,
            lambda current, reference: make_surrogate(config, current, reference.state, mesh, fractions),
            [Target("keff", transform="identity"), Target("calibration_power_W")],
            lambda output: smooth_maximum(output["relative_power_density"], options["smooth_max_sharpness"]),
            steps=None, accept=accept, search=search,
            stop=lambda reference: step_limit < options["minimum_enrichment_step_wt_percent"],
            initial_snapshot=current_reference, initial_fit=initial_fit, callback=progress,
            fit_accept=lambda report: fit_is_acceptable(config, report),
            fit_options={"steps": config["calibration"]["iterations"], "jacobian_mode": "reverse"})
        answer["fits"] = prior_fits + answer["fits"]
        answer["proposals"] = prior_proposals + answer["proposals"]
        stop_reason = ("minimum_enrichment_step_after_native_rejections" if answer["stop_reason"] == "caller_stop"
                       else answer["stop_reason"])
        answer["stop_reason"] = stop_reason
        save_json(directory / "optimization.json", {key: answer[key] for key in
                  ("fits", "proposals", "selected_fit", "design_parameters", "stop_reason", "stopped_search")})
        result.update(status="local_search_stopped", selected_index=answer["snapshot"].metadata["index"],
                      proposals=answer["proposals"], fits=answer["fits"], target_keff=target,
                      stop_reason=stop_reason, stopped_search=answer["stopped_search"])
    result["wall_s"] = time.perf_counter() - started
    save_json(directory / "summary.json", result)
    if callback is not None:
        callback(result)
    return result
