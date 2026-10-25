"""Calibrated regional diffusion with explicit geometry and material adapters."""
from contextlib import contextmanager
import numpy as np
import torch
from diffusion_surrogate.calibration import CalibratedDiffusion
from diffusion_surrogate.solver import tensor
from diffusion_surrogate.inventory import InventoryCrossSections
from materials.composition import density_change_tables, map_nuclides


class RegionalSurrogate(CalibratedDiffusion):
    """A shared material/response model independent of the caller's geometry.

    geometry() returns live mesh fractions; fuel_volumes() returns represented
    fuel-region volumes in the same order as the native inventory. The caller
    supplies the closure basis and any nuclide substitutions explicitly.
    """
    def __init__(self, design, case, reference, *, geometry, solver,
                 names, bases, equivalence, fuel_volumes, aliases=None,
                 fresh_densities=None, geometry_key=None):
        """Connect regional material responses to explicit geometry and solver adapters.

        Parameters
        ----------
        design : torch.nn.Module
            Live design; its parameters are initially frozen for calibration.
        case : dict
            Native case with explicit material regions, geometry, energy groups and power
            normalization.
        reference : dict
            Native response containing macroscopic data, material-response tables and region
            labels.
        geometry : callable
            Zero-argument callback returning live cell-by-region fractions.
        solver : Diffusion3D
            Diffusion solver implementing the regional solve interface.
        names : sequence of str
            Correction coefficient names in basis order.
        bases : dict
            Positive material-correction bases with a leading coefficient axis.
        equivalence : torch.Tensor or None
            Coefficient-by-region-by-group flux-equivalence basis; None selects shared
            corrections.
        fuel_volumes : callable
            Zero-argument callback returning live fuel-region volumes in cm³.
        aliases : dict or None, optional
            Explicit isotope substitutions used by the transport library.
        fresh_densities : callable or None, optional
            Evaluate fresh_densities(design, nuclide_names, region_indices) in atoms/(barn cm).
        geometry_key : object or None, optional
            Comparable identity of mesh topology and region ordering, enabling safe geometry
            reuse.
        """
        self.case, self.reference = case, reference
        super().__init__(solver, geometry, reference["macro"], bases, names,
                         power_W=case["power_W"], equivalence=equivalence)
        self.design = design
        self.design.requires_grad_(False)
        self.register_buffer("fuel_indices", torch.tensor([i for i, r in enumerate(case["regions"])
                                                           if r["depletable"]]))
        self.fuel_volumes = fuel_volumes
        self.aliases = {} if aliases is None else dict(aliases)
        self.fresh_densities = fresh_densities
        self.geometry_key = geometry_key
        self._inventory_tables = None

    def fresh_material_delta(self):
        """Evaluate fresh-composition changes at an unirradiated reference state.

        Returns
        -------
        dict
            Full-region macroscopic changes with the live enrichment derivative graph.

        Raises
        ------
        ValueError
            The reference contains irradiated fuel.
        """
        for region in self.case["regions"]:
            if not region["depletable"]:
                continue
            expected = map_nuclides(region["fresh_fuel_atom_density"], self.aliases)
            for name, value in map_nuclides(region["inert_atom_density"], self.aliases).items():
                expected[name] = expected.get(name, 0.) + value
            names = sorted(expected.keys() | region["atom_density"].keys())
            if not np.allclose([region["atom_density"].get(name, 0.) for name in names],
                               [expected.get(name, 0.) for name in names]):
                raise ValueError("Fresh enrichment delta cannot be applied to irradiated fuel")
        response = self.reference["material_response"]
        densities = self.fresh_densities(self.design, response["nuclides"], response["region_indices"])
        return density_change_tables(densities - tensor(response["atom_density"]), response["microscopic"],
                                     response["region_indices"], len(self.case["regions"]))


    def boron_material_delta(self, concentration):
        """Evaluate a local moderator boron-composition change.

        Parameters
        ----------
        concentration : float or torch.Tensor
            B-10 concentration in mg per kg of unborated moderator.

        Returns
        -------
        dict
            Full-region macroscopic changes relative to the native boron concentration.

        Raises
        ------
        ValueError
            The requested response lacks a suitable sampled positive-boron reference.
        """
        if not self.reference.get("boron_response"):
            concentration = tensor(concentration)
            if concentration.requires_grad or concentration.detach().item() != 0:
                raise ValueError("This model has no calibrated boron response")
            return {name: torch.zeros_like(getattr(self, "macro_" + name))
                    for name in self.reference["material_response"]["microscopic"]}
        response = self.reference["boron_response"]
        origin = self.case["boron_mg_B10_per_kg"]
        original = tensor(response["atom_density"])
        concentration = tensor(concentration)
        if origin == 0 and concentration.detach().item() == 0 and not concentration.requires_grad:
            return density_change_tables(torch.zeros_like(original), response["microscopic"],
                                         response["region_indices"], len(self.case["regions"]))
        if origin <= 0 or not np.asarray(response["analog_sampled"]).all():
            raise ValueError("A positive-boron response needs a positive-boron calibration with sampled scattering")
        densities = original * concentration / origin
        return density_change_tables(densities - original, response["microscopic"],
                                     response["region_indices"], len(self.case["regions"]))

    def inventory_material_delta(self, inventory, nuclides):
        """Evaluate fuel-composition changes while retaining fixed inert constituents.

        Parameters
        ----------
        inventory : torch.Tensor
            Fuel-region-by-chain-nuclide atom densities in atoms/cm³.
        nuclides : sequence of str
            Depletion-chain names in inventory-column order.

        Returns
        -------
        dict
            Full-region macroscopic changes; physical-inventory mode reconstructs the entire
            live fuel material.

        Raises
        ------
        ValueError
            The composition response omits fuel regions or inventory axes disagree.
        """
        response = self.reference["material_response"]
        if list(response["region_indices"]) != self.fuel_indices.tolist():
            raise ValueError("Composition response must retain every fuel region in inventory order")
        if inventory.shape != (len(self.fuel_indices), len(nuclides)):
            raise ValueError("Inventory axes differ from the selected fuel states and full chain")
        if self.case.get("physical_inventory_response"):
            if self._inventory_tables is None:
                regions = [self.case["regions"][i] for i in self.fuel_indices]
                self._inventory_tables = InventoryCrossSections(response, regions, nuclides, aliases=self.aliases)
            actual = self._inventory_tables(inventory)
            # The shared material interface accepts deltas; construct them from absolute
            # isotope sums, with no frozen fuel contribution left underneath.
            return {name: torch.zeros_like(getattr(self, "macro_" + name)).index_copy(
                0, self.fuel_indices, value - getattr(self, "macro_" + name)[self.fuel_indices])
                    for name, value in actual.items()}
        indices = {name: i for i, name in enumerate(nuclides)}
        fuel = []
        for name in response["nuclides"]:
            names = [n for n in nuclides if self.aliases.get(n, n) == name]
            selected = [indices[n] for n in names]
            fuel.append(inventory[:, selected].sum(-1) * 1e-24)
        fuel = torch.stack(fuel, dim=1)
        inert = [map_nuclides(self.case["regions"][i]["inert_atom_density"], self.aliases) for i in self.fuel_indices]
        fixed = tensor([[d.get(name, 0.) for name in response["nuclides"]] for d in inert])
        return density_change_tables(fuel + fixed - tensor(response["atom_density"]), response["microscopic"],
                                     response["region_indices"], len(self.case["regions"]))

    def forward(self, material_delta=None):
        """Solve regional diffusion and reconstruct physical flux responses.

        Parameters
        ----------
        material_delta : dict or None, optional
            Live changes to measured macroscopic tables, with identical region/group axes and
            units; None leaves the physical reference unchanged.

        Returns
        -------
        dict
            Source-normalized diffusion observables, fuel powers in W and mean regional flux;
            equivalence mode also exposes fitted factors and uncorrected diffusion amplitudes.
        """
        result = super().forward(material_delta)
        self.latest_solver_timing = {key: value for key, value in result.items() if key.endswith("wall_s")}
        result["fuel_power_W"] = result["power_W"][self.fuel_indices]
        result["region_mean_flux_per_source"] = result["region_flux_per_source"] / result["volume_cm3"][:, None]
        return result


class FreshFuelSurrogate(RegionalSurrogate):
    """Steady-state regional diffusion driven by the live fresh-fuel composition."""

    def forward(self):
        """Update isotope-dependent cross sections and solve the fixed physical state.

        Returns
        -------
        dict
            Calibrated diffusion fields with derivatives through the caller's
            fresh_densities callback. No inventory evolution is performed.
        """
        return super().forward(self.fresh_material_delta())


@contextmanager
def shared_geometry(points):
    """Reuse a live material partition across compatible response points.

    Parameters
    ----------
    points : mapping
        Response points exposing model objects with a shared design and geometry_key.

    Yields
    ------
    None
        Control while compatible models share one live fraction tensor.

    Raises
    ------
    ValueError
        Models have different designs, geometry identities or already-active shared
        partitions.

    Notes
    -----
    Models without a geometry identity use their ordinary independent geometry callbacks.
    Temporary shared fractions are cleared on exit.
    """
    models = [point.model for point in points.values() if hasattr(point, "model")]
    if not models or any(getattr(model, "geometry_key", None) is None for model in models):
        yield
        return
    first = models[0]
    for model in models:
        if (model.design is not first.design or model.geometry_key != first.geometry_key
                or getattr(model, "_live_fractions", None) is not None):
            raise ValueError("Shared geometry requires identical, idle spatial partitions")
    fractions = first.geometry()
    try:
        for model in models:
            model._live_fractions = fractions
        yield
    finally:
        for model in models:
            model._live_fractions = None
