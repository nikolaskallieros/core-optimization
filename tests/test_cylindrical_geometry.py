"""Independent annular-fuel geometry exercises the reusable cylinder interface."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from diffusion_surrogate.cylinders import (
    Cylinder, HexagonalDomain, channel_regions, cylinder_fractions, insert_regions,
)
from diffusion_surrogate.hexagonal import C6Grid
from diffusion_surrogate.openmc_cylinders import build


def annular_case(*, full=False):
    """Build a non-WR-1 annular fuel fixture in a hexagonal domain.

    Parameters
    ----------
    full : bool, optional
        Build the full hexagon rather than its rotational sector.

    Returns
    -------
    tuple
        Native region mapping, site IDs, centers in cm and radial interfaces in cm.
    """
    domain = HexagonalDomain(4., full=full)
    sites = ["central"] + [f"outer{i}" for i in range(6)]
    xy = np.array([[0., 0.]] + [[2. * math.cos(i * math.pi / 3),
                                2. * math.sin(i * math.pi / 3)] for i in range(6)])
    interfaces = [[.2, .4]] + [[.15, .3]] * 6
    z = [0., 1., 3., 6., 10.]

    def material(site, shell, level, volume):
        """Supply artificial material properties for one annular shell.

        Parameters
        ----------
        site : int
            Channel site index accepted by the material callback.
        shell : int
            Radial shell index; the outer shell is the fixture fuel.
        level : int
            Axial level index accepted by the callback.
        volume : float
            Complete shell volume in cm³.

        Returns
        -------
        dict
            Manufactured material temperature, atom density and initial heavy-metal mass.
        """
        return {"depletable": shell == 1, "temperature_K": 800. if shell else 500.,
                "atom_density": {"Pu239": .015} if shell else {"Zr90": .04},
                "initial_hm_kg_per_assembly": volume * .005 if shell else 0.}

    background = {"label": "sodium", "kind": "coolant", "depletable": False, "void": False,
                  "atom_density": {"Na23": .03}, "temperature_K": 500.}
    native = channel_regions(domain, sites, xy, interfaces, z, [(0,), tuple(range(1, 7))],
                             material, background)
    return native, sites, xy, interfaces


def test_annular_fuel_volumes_and_live_gradients_with_four_axial_levels():
    """Verify annular fuel volumes and live gradients with four axial levels."""
    native, sites, xy, interfaces = annular_case()
    radii = torch.tensor([.4, .3], dtype=torch.float64, requires_grad=True)
    cylinders = {site: Cylinder(torch.tensor(xy[i], dtype=torch.float64),
                               (radii.new_tensor(interfaces[i][0]), radii[int(i > 0)]))
                 for i, site in enumerate(sites)}
    grid = C6Grid(4., 8, [0., .5, 1., 2., 3., 4.5, 6., 8., 10.])
    volumes = grid.volumes @ cylinder_fractions(grid, native["regions"], cylinders)
    np.testing.assert_allclose(volumes.detach(), [r["volume_cm3"] for r in native["regions"]],
                               rtol=2e-10, atol=1e-12)
    assert len(native["physical_assembly_to_region"]) == 28
    fuels = [(i, r) for i, r in enumerate(native["regions"]) if r["depletable"]]
    assert len(fuels) == 8
    for i, region in fuels:
        group = int(region["label"].startswith("outer"))
        inner, outer = interfaces[group]
        height = region["z_bounds_cm"][1] - region["z_bounds_cm"][0]
        physical_volume = math.pi * (outer**2 - inner**2) * height
        assert region["full_assembly_volume_cm3"] == pytest.approx(physical_volume)
        assert region["initial_hm_kg_per_assembly"] == pytest.approx(.005 * physical_volume)
        gradient = torch.autograd.grad(volumes[i], radii, retain_graph=True)[0]
        expected = torch.zeros_like(radii)
        expected[group] = region["physical_multiplicity"] / 6 * 2 * math.pi * outer * height
        torch.testing.assert_close(gradient, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("full", [False, True])
def test_annular_native_cells_have_one_owner_and_the_declared_material(tmp_path, full):
    """Verify annular native cells have one owner and the declared material.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    full : bool
        Use the complete hexagon instead of a single symmetry sector.
    """
    native, _, _, _ = annular_case(full=full)
    native.update(sampling={"particles": 100, "batches": 10, "inactive": 2, "seed": 9},
                  cross_sections=str(tmp_path / "cross_sections.xml"),
                  temperature_label_tolerance_K=1., group_edges_eV=[0., 2e7],
                  comparison_edges_cm=[[-4., 4.], [-5., 5.], [0., 1., 3., 6., 10.]])
    model, _ = build(native, tmp_path)
    cells = list(model.geometry.get_all_cells().values())
    for z in (.5, 2., 4.5, 8.):
        for radius, nuclide in ((.1, "Zr90"), (.3, "Pu239"), (.6, "Na23")):
            owners = [cell for cell in cells if (radius, 0., z) in cell.region]
            assert len(owners) == 1
            assert owners[0].fill.get_nuclides() == [nuclide]
    expected = (6 if full else 1) * 4. * (4. / math.sqrt(3)) * 10.
    assert sum(region["volume_cm3"] for region in native["regions"]) == pytest.approx(expected)
    assert (tmp_path / "geometry.xml").is_file()


def test_axial_insertion_rejects_unmatched_intervals_before_mutating_a_case():
    """Verify axial insertion rejects unmatched intervals before mutating a case."""
    native, _, _, _ = annular_case()
    before = deepcopy(native)
    backgrounds = [i for i, r in enumerate(native["regions"]) if r["kind"] == "coolant"]
    addition = {"z_bounds_cm": [.5, 1.5], "volume_cm3": .1}
    with pytest.raises(ValueError, match="existing background axial intervals"):
        insert_regions(native, [addition], backgrounds, [])
    assert native == before


@pytest.mark.parametrize("full", [False, True])
def test_native_geometry_uses_declared_radial_and_axial_boundaries(tmp_path, full):
    """Export physical boundary choices instead of hardcoding radial vacuum.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated OpenMC XML output directory supplied by pytest.
    full : bool
        Use the full hexagon or its rotational symmetry sector.
    """
    native, _, _, _ = annular_case(full=full)
    native["geometry"]["boundaries"] = ["reflective"] * 5 + ["vacuum"]
    native.update(sampling={"particles": 100, "batches": 10, "inactive": 2, "seed": 9},
                  cross_sections=str(tmp_path / "cross_sections.xml"),
                  temperature_label_tolerance_K=1., group_edges_eV=[0., 2e7],
                  comparison_edges_cm=[[-4., 4.], [-5., 5.], [0., 1., 3., 6., 10.]])
    model, _ = build(native, tmp_path)
    boundaries = [surface.boundary_type for surface in model.geometry.get_all_surfaces().values()]
    assert boundaries.count("reflective") == (7 if full else 2)
    assert boundaries.count("vacuum") == 1


def test_wr1_diffusion_mesh_inherits_native_physical_boundaries():
    """Keep native and surrogate boundaries aligned after a declared input change."""
    from examples.wr_1.model.case import study
    from examples.wr_1.model.geometry import make_case, mesh

    case = study(1)
    settings = case.settings()
    settings.update(outer_radial_boundary="reflective", axial_boundaries=["reflective", "vacuum"])
    native = make_case(case.configuration.data("model"), settings)
    geometry = mesh(case.design(), native, subdivisions=2, axial_subdivisions=1)
    assert geometry.grid.boundaries == tuple(native["geometry"]["boundaries"])
    assert geometry.grid.boundaries[0] == geometry.grid.boundaries[4] == "reflective"
    assert geometry.grid.boundaries[5] == "vacuum"
