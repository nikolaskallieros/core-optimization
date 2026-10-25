"""Verify the WR-1 design geometry and its differentiable volume mapping."""
from copy import deepcopy
import json
import sys
from runtime.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from designs.wr_1 import WR1_ROOT as ROOT
RADIUS_CHECK_STEP_MM = 1.0


def design_check():
    """Verify WR-1 parameter mapping, exact volumes and their gradients.

    Returns
    -------
    dict
        Geometry, identity, derivative and parameter-independence checks; no transport solve
        is performed.
    """
    import torch
    from diffusion_surrogate import Grid3D, cylinder_regions

    sys.path.insert(0, str(PROJECT_ROOT))
    from designs.wr_1.model.case import study

    case = study(1, root=ROOT)
    design = case.design()
    initial = deepcopy(design.state_dict())
    output = design()
    sites = case.configuration.data("model")["lattice"]["sites"]
    lattice = torch.tensor([[site["q"], site["r"]] for site in sites], dtype=torch.float64)
    basis = torch.tensor([[234.95, 0.], [117.475, 234.95 * 3**.5 / 2]], dtype=torch.float64)
    expected = lattice @ basis
    center_error = float((output["channel_xy_mm"].detach() - expected).abs().max())
    assert center_error < 1e-10
    assert len(set(design.assembly_ids)) == len(design.assembly_ids) == 111
    assert torch.equal(output["assembly_z_bounds_cm"][:3, 0], torch.tensor(
        [198., 99., 0.], dtype=torch.float64))
    assert torch.allclose(output["assembly_radius_cm"], torch.full((111,), 4.1275, dtype=torch.float64),
                          rtol=1e-14, atol=0.)

    # This rectangular grid tests exact volume mapping only. Its boundary
    # is not a replacement for the requested WR-1 hexagon and 60-degree sector.
    extent = float((output["assembly_xy_cm"].detach().abs().max() + 6).ceil())
    grid = Grid3D((torch.linspace(-extent, extent, 9, dtype=torch.float64),
                   torch.linspace(-extent, extent, 9, dtype=torch.float64),
                   design.z_edges_mm / 10))

    def volumes(state):
        """Integrate mapped assembly fractions over the fixture grid.

        Parameters
        ----------
        state : mapping
            Design output with assembly centers, radii and axial bounds in cm.

        Returns
        -------
        torch.Tensor
            Mapped physical assembly volumes in cm³.
        """
        fractions = cylinder_regions(grid, state["assembly_xy_cm"], state["assembly_radius_cm"],
                                     state["assembly_z_bounds_cm"])
        return (grid.volumes[:, None] * fractions[:, :len(design.assembly_ids)]).sum(0)

    mapped = volumes(output)
    volume_error = float(((mapped - output["assembly_volume_cm3"]) / output["assembly_volume_cm3"])
                         .detach().abs().max())
    assert volume_error < 1e-11
    dr, dp = torch.autograd.grad(mapped.sum(), (design.element_radius_mm, design.pitch_mm), retain_graph=True)
    expected_dr = torch.autograd.grad(output["assembly_volume_cm3"].sum(), design.element_radius_mm)[0]
    gradient_error = float(((dr - expected_dr) / expected_dr).abs().max())
    assert gradient_error < 1e-10 and float(dp.abs().max()) < 1e-8
    finite = []
    for ring in range(4):
        values = []
        for direction in (-1, 1):
            design.load_state_dict(initial)
            with torch.no_grad():
                design.element_radius_mm[ring] += direction * RADIUS_CHECK_STEP_MM
                values.append(float(volumes(design()).sum()))
        finite.append((values[1] - values[0]) / (2 * RADIUS_CHECK_STEP_MM))
    fd_error = float(((torch.tensor(finite, dtype=torch.float64) - dr) / dr).abs().max())
    assert fd_error < 1e-10
    design.load_state_dict(initial)
    with torch.no_grad():
        design.element_radius_mm[2] += 1.
    resized = design()
    assert torch.equal(resized["channel_xy_mm"], output["channel_xy_mm"])
    assert torch.equal(resized["fresh_enrichment_mass_fraction"], output["fresh_enrichment_mass_fraction"])
    design.load_state_dict(initial)
    with torch.no_grad():
        design.pitch_mm[1] += 1.
    moved = design()
    assert not torch.equal(moved["channel_xy_mm"], output["channel_xy_mm"])
    assert torch.equal(moved["assembly_radius_cm"], output["assembly_radius_cm"])
    assert torch.equal(moved["assembly_volume_cm3"], output["assembly_volume_cm3"])
    design.load_state_dict(initial)
    with torch.no_grad():
        design.enrichment[3] += .001
    enriched = design()
    assert torch.equal(enriched["channel_xy_mm"], output["channel_xy_mm"])
    assert torch.equal(enriched["assembly_radius_cm"], output["assembly_radius_cm"])
    assert int((enriched["fresh_enrichment_mass_fraction"] != output["fresh_enrichment_mass_fraction"]).sum()) == 54
    return {"passed": True, "fuelled_channels": 37, "distinct_axial_assemblies": 111,
            "initial_element_radius_mm": 41.275, "initial_pitch_mm": [234.95] * 3,
            "initial_enrichment_mass_fraction": [.024] * 4,
            "max_center_error_mm": center_error, "max_relative_volume_error": volume_error,
            "max_relative_radius_gradient_error": gradient_error,
            "max_relative_radius_finite_difference_error": fd_error,
            "finite_difference_step_mm": RADIUS_CHECK_STEP_MM,
            "radius_pitch_enrichment_independence": True,
            "scope": "Full-layout geometry and derivative mapping only; no transport calculation."}


def main():
    """Save and report the bounded WR-1 geometry-mapping check."""
    import torch
    torch.set_num_threads(1)
    output = PROJECT_ROOT / "validation-output/wr1_design_mapping.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(design_check(), indent=2) + "\n")
    print("Passed: WR-1 design mapping")


if __name__ == "__main__":
    main()
