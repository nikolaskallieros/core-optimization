"""Symmetry groups and physical response bindings for the shared calibration."""

import torch
import numpy as np

from diffusion_surrogate import SparseDiffusion3D
from diffusion_surrogate.calibration import CalibratedDiffusion
from diffusion_surrogate.regional import FreshFuelSurrogate
from optimization.objectives import relative_power_density

from ..build_geometry import build_geometry
from .case import make_mesh


def correction_groups(config):
    """Group calibration factors by the lattice's exact reflection symmetries.

    Parameters
    ----------
    config : dict
        Resolved geometry with a square pin lattice and axial segmentation.

    Returns
    -------
    list of tuple
        Radial orbit and reflected axial index for each fuel segment. These
        groups share fitted corrections; all design parameters remain separate.
    """
    nx, ny = config["geometry"]["shape"]
    zones = config["geometry"]["axial_segments"]
    return [(*sorted((abs(i - (nx - 1) / 2), abs(j - (ny - 1) / 2))), min(z, zones - 1 - z))
            for j in range(ny) for i in range(nx) for z in range(zones)]


def calibration_basis(config, regions):
    """Build the common material corrections and native power observation map.

    Parameters
    ----------
    config : dict
        Physical symmetries and energy-group structure.
    regions : int
        Fuel segments followed by the water region.

    Returns
    -------
    tuple
        Correction names, fission-reaction basis and symmetry averaging matrix.
    """
    assignments = correction_groups(config)
    unique = sorted(set(assignments))
    groups = len(config["transport"]["group_edges_eV"]) - 1
    basis = torch.zeros((len(unique), regions, groups), dtype=torch.float64)
    for region, category in enumerate(assignments):
        basis[unique.index(category), region] = 1.
    averaging = basis[:, :-1, 0] / basis[:, :-1, 0].sum(-1, keepdim=True)
    return [f"fission:{category}" for category in unique], {"fission": basis}, averaging


def make_surrogate(config, design, state, mesh, fractions):
    """Fit reaction coefficients against native criticality and segment powers.

    Parameters
    ----------
    config : dict
        Resolved model.toml.
    design : Enrichment
        Live independent enrichment coordinates.
    state : dict
        Native case and continuous-energy reference result.
    mesh : diffusion_surrogate.geometry.DiffusionGeometry
        Mesh derived from the shared physical CAD.
    fractions : torch.Tensor
        Cached cell-by-region fractions; geometry is fixed in this study.

    Returns
    -------
    diffusion_surrogate.regional.FreshFuelSurrogate
        Two-group composition response with positive fission-reaction corrections.
        The same correction acts on absorption, production and fission heating.
    """
    names, bases, averaging = calibration_basis(config, fractions.shape[1])
    model = FreshFuelSurrogate(design, state["case"], state["result"],
        geometry=lambda: fractions, solver=SparseDiffusion3D(mesh.grid),
        names=names, bases=bases,
        equivalence=None, fuel_volumes=lambda: design.fuel_volumes_cm3,
        fresh_densities=lambda current, names, regions: current.isotope_densities(names, regions),
        geometry_key=mesh.identity)
    object.__setattr__(model, "responses", lambda result: {
        "calibration_power_W": averaging @ result["power_W"][:-1],
        "relative_power_density": relative_power_density(result["power_W"][:-1], design.fuel_volumes_cm3),
        "u235_mass_kg": design.u235_mass_kg()})
    design.requires_grad_(True)
    return model


def make_size_surrogate(config, design, state):
    """Use the same material calibration with a differentiable moving mesh.

    Parameters
    ----------
    config : dict
        Source numerical configuration and physical geometry.
    design : LatticeSize
        Live common transverse scale; compositions remain uniform and fixed.
    state : dict
        Native transport observation at the current physical size.

    Returns
    -------
    CalibratedDiffusion
        Material-calibrated diffusion with live cell volumes and face metrics.

    Notes
    -----
    Uniform transverse scaling moves pins and the fuel mesh together, preserving
    their exact material fractions. Reflector cells contain only water. Reusing
    those CAD-derived fractions avoids repeated intersections; the physical CAD
    and mesh coordinates are rebuilt at each evaluation.
    """
    current = design.configuration()
    mesh = make_mesh(current, build_geometry(current["geometry"]))
    fractions = mesh.fractions().detach()
    solver = SparseDiffusion3D(mesh.grid)
    names, bases, averaging = calibration_basis(config, fractions.shape[1])

    def geometry():
        """Update mesh dimensions from the live shared CAD before each solve.

        Returns
        -------
        torch.Tensor
            Cell-by-material fractions invariant under this transverse scaling.
        """
        current = design.configuration()
        mesh = make_mesh(current, build_geometry(current["geometry"]))
        solver.grid = mesh.grid
        return fractions

    return CalibratedDiffusion(solver, geometry, state["result"]["macro"], bases, names,
        power_W=config["transport"]["power_W"],
        responses=lambda result: {"calibration_power_W": averaging @ result["power_W"][:-1]})


def fit_is_acceptable(config, report):
    """Require the declared native criticality and grouped spatial fitting accuracy.

    Parameters
    ----------
    config : dict
        Explicit calibration tolerances.
    report : dict
        Shared fitter's completed predictions and reference observations.

    Returns
    -------
    bool
        Whether both criticality and grouped power errors satisfy their tolerances.
    """
    predicted = report["predictions"]
    delta_k = abs(predicted["keff"]["prediction"][0] - predicted["keff"]["reference"][0])
    power = predicted["calibration_power_W"]
    error = np.max(abs(np.asarray(power["prediction"]) / power["reference"] - 1))
    return delta_k <= config["calibration"]["keff_absolute_tolerance"] and error <= config["calibration"]["group_power_relative_tolerance"]


def fitting_reference(config, state):
    """Average symmetry-equivalent power observations for material calibration.

    Parameters
    ----------
    config : dict
        Physical symmetry groups and explicit fitting scales.
    state : dict
        Native reference case and its independently sampled observations.

    Returns
    -------
    dict
        Native k-effective and mean segment power per symmetry group. The
        declared relative power scale is a fitting weight, not an uncertainty.
    """
    assignments = correction_groups(config)
    power = torch.as_tensor(state["result"]["observables"]["region_power_W"]["mean"][:-1], dtype=torch.float64)
    means = torch.stack([power[[i for i, value in enumerate(assignments) if value == group]].mean()
                         for group in sorted(set(assignments))])
    options = config["calibration"]
    return {"observables": {
        "keff": {"mean": state["result"]["combined_keff"][0], "scale": options["keff_scale"]},
        "calibration_power_W": {"mean": means.tolist(), "scale": (means * options["power_relative_scale"]).tolist()}}}
