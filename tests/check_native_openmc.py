"""Verify native OpenMC and its C API on a one-group known-answer problem."""
import json
from pathlib import Path
import tempfile

import numpy as np
import openmc

from diffusion_surrogate.reference import native_batches
from runtime.processes import native_library_path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation-output"
EXPECTED_KEFF = 1.25  # Infinite medium: nu-Sigma_f / Sigma_a = 0.25 / 0.20.


def check(directory):
    """Verify the native OpenMC runtime with a manufactured MG problem.

    Parameters
    ----------
    directory : pathlib.Path
        Scratch directory for one-group data, native inputs and outputs.

    Returns
    -------
    dict
        Sampled keff, uncertainty and batch-array checks against the infinite-medium
        analytic reference.
    """
    groups = openmc.mgxs.EnergyGroups([0., 20.e6])
    xs = openmc.XSdata("verification", groups)
    xs.order = 0
    xs.set_total([1.])
    xs.set_absorption([.2])
    xs.set_scatter_matrix(np.array([[[.8]]]))
    xs.set_fission([.1])
    xs.set_nu_fission([.25])
    xs.set_kappa_fission([2.e7])
    xs.set_chi([1.])
    library = openmc.MGXSLibrary(groups)
    library.add_xsdata(xs)
    library.export_to_hdf5(directory / "mgxs.h5")
    material = openmc.Material()
    material.set_density("macro", 1.)
    material.add_macroscopic("verification")
    materials = openmc.Materials([material])
    materials.cross_sections = directory / "mgxs.h5"
    planes = [plane(-1., boundary_type="reflective") for plane in (openmc.XPlane, openmc.YPlane, openmc.ZPlane)]
    upper = [plane(1., boundary_type="reflective") for plane in (openmc.XPlane, openmc.YPlane, openmc.ZPlane)]
    cell = openmc.Cell(fill=material, region=+planes[0] & -upper[0] & +planes[1] & -upper[1] & +planes[2] & -upper[2])
    model = openmc.Model(geometry=openmc.Geometry([cell]), materials=materials)
    model.settings.energy_mode = "multi-group"
    model.settings.particles = 2000
    model.settings.batches = 30
    model.settings.inactive = 5
    model.settings.seed = 7481
    model.settings.verbosity = 1
    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point())
    tally = openmc.Tally(tally_id=1)
    tally.filters = [openmc.CellFilter([cell])]
    tally.scores = ["kappa-fission", "nu-fission"]
    model.tallies = openmc.Tallies([tally])
    model.export_to_xml(directory)
    samples = native_batches(directory, model.settings.inactive, threads=1)
    with openmc.StatePoint(directory / "statepoint.30.h5") as state:
        generations = state.k_generation[model.settings.inactive:]
    # The combined estimator has singular covariance in this exact homogeneous
    # case. Check the sampled generation estimates directly instead.
    mean = float(generations.mean())
    error = float(generations.std(ddof=1) / np.sqrt(len(generations)))
    if samples.shape != (25, 1, 2) or not np.isfinite(samples).all() or not (samples > 0).all():
        raise AssertionError("Native C API did not return all finite positive batch scores")
    if not np.isfinite([mean, error]).all() or abs(mean - EXPECTED_KEFF) > max(6 * error, .01):
        raise AssertionError(f"Native OpenMC differs from the analytical reference: {mean} +/- {error}")
    return {"passed": True, "openmc": openmc.__version__, "expected_keff": EXPECTED_KEFF,
            "keff": mean, "standard_error": error, "active_histories": 50000,
            "estimator": "Mean active-generation k; standard error of the generation mean",
            "batch_shape": list(samples.shape), "library": native_library_path().name,
            "scope": "One-group infinite-medium runtime verification; no reactor validation."}


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openmc-check-") as temporary:
        result = check(Path(temporary))
    (OUTPUT / "native_openmc.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Native OpenMC passed: k-effective {result['keff']:.6f} (reference {EXPECTED_KEFF}).")
