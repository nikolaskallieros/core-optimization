"""Check the public tutorial against the homogeneous slab's critical width."""
import math
import tomllib

from examples.slab.run import INPUTS, run


def test_tutorial_recovers_analytical_critical_width():
    """Recover the continuum critical slab width within finite-volume mesh error."""
    with INPUTS.open("rb") as stream:
        settings = tomllib.load(stream)
    material = settings["material"]
    diffusion = 1 / (3 * (material["absorption_per_cm"] + material["scatter_per_cm"]))
    buckling = math.sqrt((material["neutrons_per_fission"] * material["fission_per_cm"]
                          / settings["optimization"]["target_keff"]
                          - material["absorption_per_cm"]) / diffusion)
    critical_width = 2 * math.atan(1 / (2 * diffusion * buckling)) / buckling
    result = run()
    assert abs(result["final_width_cm"] / critical_width - 1) < 0.002
    assert abs(result["final_keff"] - settings["optimization"]["target_keff"]) < 1e-7
    assert result["initial_dkeff_dwidth_per_cm"] > 0
