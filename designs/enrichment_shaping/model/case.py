"""Bind physical inputs to the shared native transport and diffusion interfaces."""

from pathlib import Path
import tomllib

import numpy as np
import torch

from geometry.diffusion import discretize, subdivide_intervals
from runtime.paths import NUCLEAR_DATA

from ..build_geometry import build_geometry
from .materials import fuel_recipe, water_densities

DESIGN_ROOT = Path(__file__).resolve().parents[1]
INPUTS = DESIGN_ROOT / "inputs/model.toml"


def settings():
    """Read the single executable definition of this illustrative design.

    Returns
    -------
    dict
        Physical and numerical settings from inputs/model.toml.
    """
    return tomllib.loads(INPUTS.read_text())


def compositions(config, enrichment_wt_percent):
    """Build native fuel-segment and water densities from a physical enrichment field.

    Parameters
    ----------
    config : dict
        Resolved model.toml.
    enrichment_wt_percent : torch.Tensor
        Pin-by-axial-segment enrichment values in wt% U235 within uranium.

    Returns
    -------
    list of dict
        Detached native atom densities in atoms/(barn cm), fuel first, water last.
    """
    atoms = fuel_recipe(config["materials"]).atom_densities(enrichment_wt_percent.flatten() / 100)
    fuels = [{name: float(value[i].detach()) for name, value in atoms.items()}
             for i in range(enrichment_wt_percent.numel())]
    return fuels + [water_densities(config["materials"])]


def prepare(config):
    """Connect the declared geometry, composition and native observation settings.

    Parameters
    ----------
    config : dict
        Resolved physical and numerical settings.

    Returns
    -------
    tuple
        Shared CAD, diffusion geometry, cached material fractions, initial
        enrichment field and the native case dictionary.
    """
    geometry = config["geometry"]
    count = int(np.prod(geometry["shape"]))
    zones = geometry["axial_segments"]
    enrichment = torch.full((count, zones), config["materials"]["initial_enrichment_wt_percent"],
                            dtype=torch.float64)
    cad = build_geometry(geometry)
    mesh = make_mesh(config, cad)
    fractions = mesh.fractions()
    volumes = fractions.T @ mesh.grid.volumes
    atoms = compositions(config, enrichment)
    regions = [{"label": volume.label, "volume_cm3": float(volumes[i]),
                "atom_density": atoms[i], "fresh_fuel_atom_density": atoms[i],
                "inert_atom_density": {}, "temperature_K": config["materials"]["temperature_K"],
                "depletable": volume.material == "fuel", "void": False,
                "thermal_scattering": [] if volume.material == "fuel" else ["c_H_in_H2O"]}
               for i, volume in enumerate(cad.volumes)]
    nx, ny = geometry["shape"]
    px, py = geometry["pitch_cm"]
    case = {"cad_geometry": cad.record(), "regions": regions,
            "cross_sections": str(NUCLEAR_DATA / config["nuclear_data"]["cache"] / "cross_sections.xml"),
            "sampling": config["openmc"],
            "temperature_label_tolerance_K": config["transport"]["temperature_label_tolerance_K"],
            "group_edges_eV": config["transport"]["group_edges_eV"],
            "power_W": config["transport"]["power_W"],
            "comparison_edges_cm": [np.linspace(-nx * px / 2, nx * px / 2, nx + 1).tolist(),
                                    np.linspace(-ny * py / 2, ny * py / 2, ny + 1).tolist(),
                                    np.linspace(0., geometry["fuel_height_cm"], zones + 1).tolist()],
            "statistics_block_size": config["transport"]["statistics_block_size"],
            "nuclides": ["U235", "U238", "O16", "H1"],
            "observation_cross_sections": "macroscopic_tallies", "full_observations": True,
            "material_response": {"region_indices": list(range(count * zones)),
                                  "nuclides": ["U235", "U238", "O16"]},
            "need_depletion_rates": False, "depletion_nuclides": [],
            "depletion_reactions": [], "depletion_rate_estimator": "disabled"}
    return cad, mesh, fractions, enrichment, case


def make_mesh(config, cad, refinement=1):
    """Allocate numerical resolution to fuel and reflector intervals separately.

    Parameters
    ----------
    config : dict
        Physical dimensions and mesh cell counts from model.toml.
    cad : geometry.CadGeometry
        The same physical construction used for native transport.
    refinement : int, optional
        Multiplier on every declared mesh interval's cell count.

    Returns
    -------
    diffusion_surrogate.geometry.DiffusionGeometry
        Nonuniform Cartesian grid with exact shared-CAD material fractions.
    """
    geometry, mesh = config["geometry"], config["mesh"]
    lower, upper = cad.bounds_cm()
    edges = []
    for axis, (count, pitch) in enumerate(zip(geometry["shape"], geometry["pitch_cm"])):
        half = torch.as_tensor(count * pitch / 2, dtype=torch.float64)
        physical = torch.stack((lower[axis], -half, half, upper[axis]))
        cells = [mesh["radial_reflector_cells"], count * mesh["cells_per_pitch"], mesh["radial_reflector_cells"]]
        edges.append(subdivide_intervals(physical, [refinement * value for value in cells]))
    fuel_edges = torch.linspace(0., geometry["fuel_height_cm"], geometry["axial_segments"] + 1, dtype=torch.float64)
    physical = torch.cat((lower[2:3], fuel_edges, upper[2:3]))
    cells = [mesh["axial_reflector_cells"], *([mesh["axial_cells_per_segment"]] * geometry["axial_segments"]), mesh["axial_reflector_cells"]]
    edges.append(subdivide_intervals(physical, [refinement * value for value in cells]))
    return discretize(cad, cartesian_edges=edges)
