"""Manufactured material tables and the small lattice's short depletion chain."""

import openmc.data
import openmc.deplete
import torch

from depletion.sparse import SparseChain
from fuel_cycle.residence import FuelState
from materials.composition import contract_isotopes


class Materials:
    """Manufactured one-group fuel isotope data and fixed surrounding coolant."""

    def __init__(self, settings, regions):
        """Construct microscopic tables from the declared demonstration values.

        Parameters
        ----------
        settings : dict
            Material inputs, in barns, atoms/cm³, eV and cm⁻¹ as named.
        regions : int
            Number of independent fuel regions.
        """
        self.settings, self.regions = settings, regions
        capture, fission, scatter = [torch.tensor(settings[name + "_barns"], dtype=torch.float64)
                                     .reshape(1, -1, 1).expand(regions, -1, -1)
                                     for name in ("capture", "fission", "scatter")]
        self.micro = {"absorption": capture + fission, "fission": fission,
                      "nu_fission": settings["neutrons_per_fission"] * fission,
                      "kappa_fission": settings["energy_per_fission_eV"] * fission,
                      "scatter": scatter[..., None], "nu_scatter": scatter[..., None],
                      "transport": capture + fission + scatter}
        self.rates = torch.stack((fission, capture), dim=2)
        self.initial = torch.tensor(settings["initial_density_atoms_cm3"], dtype=torch.float64).expand(regions, -1)

    def macroscopic(self, inventory):
        """Map the four fuel isotope densities and fixed coolant to all regions.

        Parameters
        ----------
        inventory : torch.Tensor
            Fuel-region-by-nuclide number densities in atoms/cm³.

        Returns
        -------
        dict of torch.Tensor
            Fuel-first, coolant-last macroscopic tables for both solvers.
        """
        fuel = contract_isotopes(inventory, self.micro)
        coolant = self.settings["coolant"]
        a, s = coolant["absorption_per_cm"], coolant["scatter_per_cm"]
        constants = {"absorption": a, "fission": 0., "nu_fission": 0., "kappa_fission": 0.,
                     "scatter": s, "nu_scatter": s, "transport": a + s}
        result = {name: torch.cat((value, torch.full_like(value[:1], constants[name])))
                  for name, value in fuel.items()}
        result["chi"] = torch.ones_like(result["fission"])
        return result

    def fresh_state(self, volumes):
        """Create fresh isotope densities and physical fuel masses for a design.

        Parameters
        ----------
        volumes : torch.Tensor
            Ordered physical fuel volumes in cm³, retaining radius derivatives.

        Returns
        -------
        fuel_cycle.residence.FuelState
            Fresh inventory, zero integrated energy and initial U235 masses in kg.
        """
        mass = (self.initial[:, 0] * volumes * openmc.data.atomic_mass("U235")
                / (1000 * openmc.data.AVOGADRO))
        return FuelState(self.initial.clone(), torch.zeros_like(volumes), mass,
                         tuple(f"fuel_{i:02d}" for i in range(self.regions)))


def build_chain(settings, materials):
    """Build the explicitly manufactured fission/capture/decay demonstration chain.

    Parameters
    ----------
    settings : dict
        Artificial iodine yield and iodine/xenon half-lives in seconds.
    materials : dict
        Ordered nuclide names and the fission energy in eV.

    Returns
    -------
    depletion.sparse.SparseChain
        Four-isotope OpenMC chain with fission and capture rate ordering.
    """
    chain = openmc.deplete.Chain()
    fuel, iodine, xenon, stable = [openmc.deplete.Nuclide(name) for name in materials["nuclides"]]
    fuel.add_reaction("fission", None, materials["energy_per_fission_eV"], 1.)
    fuel.add_reaction("(n,gamma)", None, 0., 1.)
    fuel.yield_data = {0.: {iodine.name: settings["fission_yield_to_iodine"]}}
    iodine.half_life = settings["iodine_half_life_seconds"]
    iodine.add_decay_mode("beta-", xenon.name, 1.)
    xenon.half_life = settings["xenon_half_life_seconds"]
    xenon.add_decay_mode("beta-", None, 1.)
    xenon.add_reaction("(n,gamma)", stable.name, 0., 1.)
    chain.nuclides = [fuel, iodine, xenon, stable]
    chain.nuclide_dict = {nuclide.name: i for i, nuclide in enumerate(chain.nuclides)}
    chain.reactions = ["fission", "(n,gamma)"]
    return SparseChain(chain, materials["nuclides"], chain.reactions)
