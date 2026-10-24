"""Isotopic recipes and explicitly declared transport substitutions."""

from dataclasses import dataclass
from typing import Mapping

import torch
from diffusion_surrogate.solver import tensor


def number_density(nuclide, density_g_cm3):
    """Convert pure-material mass density to atom density.

    Parameters
    ----------
    nuclide : str
        OpenMC nuclide or evaluated natural-element name used for atomic mass.
    density_g_cm3 : float or torch.Tensor
        Material mass density in g/cm³.

    Returns
    -------
    float or torch.Tensor
        Number density in atoms/(barn cm), with the input's derivative behavior.
    """
    import openmc.data
    return density_g_cm3 * openmc.data.AVOGADRO / 1e24 / openmc.data.atomic_mass(nuclide)


def map_nuclides(atoms, aliases):
    """Combine explicitly aliased nuclides without changing density units.

    Parameters
    ----------
    atoms : dict
        Nuclide densities as scalars or tensors; all entries use a common density unit.
    aliases : dict of str to str
        Explicit source-to-transport nuclide substitutions; unmapped names are retained.

    Returns
    -------
    dict
        Densities summed by target nuclide, retaining tensor derivatives.
    """
    result = {}
    for name, value in atoms.items():
        target = aliases.get(name, name)
        result[target] = result.get(target, 0.) + value
    return result


@dataclass(frozen=True)
class UO2Recipe:
    """U235 mass-fraction enrichment at fixed oxide density and volume fraction."""

    density_g_cm3: float
    volume_fraction: float
    oxygen_atom_fractions: Mapping[str, float]

    @property
    def oxygen_atomic_mass(self):
        """Compute the recipe's oxygen-isotope weighted atomic mass.

        Returns
        -------
        float
            Mean oxygen atomic mass in unified atomic mass units.
        """
        import openmc.data
        return sum(weight * openmc.data.atomic_mass(name)
                   for name, weight in self.oxygen_atom_fractions.items())

    def atom_densities(self, enrichment):
        """Evaluate fresh UO2 isotope densities at fixed oxide loading.

        Parameters
        ----------
        enrichment : float or torch.Tensor
            U-235 mass fraction within uranium, preserving any Torch derivative graph.

        Returns
        -------
        dict
            U-235, U-238 and oxygen isotope densities in atoms/(barn cm), with enrichment's
            broadcast shape.
        """
        import openmc.data
        m235, m238 = (openmc.data.atomic_mass(name) for name in ("U235", "U238"))
        molar = 1. / (enrichment / m235 + (1. - enrichment) / m238)
        uranium = (self.volume_fraction * self.density_g_cm3 * openmc.data.AVOGADRO * 1e-24
                   / (molar + 2. * self.oxygen_atomic_mass))
        return {"U235": uranium * molar * enrichment / m235,
                "U238": uranium * molar * (1. - enrichment) / m238,
                **{name: 2. * uranium * weight
                   for name, weight in self.oxygen_atom_fractions.items()}}

    def live_inventory(self, enrichment, nuclides, volumes, aliases, *, reference=None):
        """Evaluate live fresh inventories, volumes and heavy-metal masses.

        Parameters
        ----------
        enrichment : torch.Tensor
            One U-235 uranium mass fraction per fuel region.
        nuclides : sequence of str
            Requested isotope column order.
        volumes : torch.Tensor
            Live fuel-region volumes in cm³.
        aliases : dict of str to str
            Explicit source-to-transport nuclide substitutions; unmapped names are retained.
        reference : tuple or None, optional
            Reference enrichment and native atom-density array in atoms/cm³, used to preserve
            the exact native loading while retaining recipe derivatives.

        Returns
        -------
        inventory : torch.Tensor
            Region-by-nuclide atom densities in atoms/cm³.
        volumes : torch.Tensor
            The supplied live region volumes in cm³.
        mass : torch.Tensor
            Initial uranium mass per region in kg.

        Notes
        -----
        reference=(enrichment, inventory) preserves the native values exactly at that loading
        while keeping the recipe's physical composition derivatives.
        """
        import openmc.data

        def densities(values):
            """Map fresh isotope densities into the requested inventory ordering.

            Parameters
            ----------
            values : torch.Tensor
                U-235 uranium mass fractions for the selected regions.

            Returns
            -------
            torch.Tensor
                Region-by-nuclide atom densities in atoms/cm³; absent isotopes receive zero.
            """
            atoms = map_nuclides(self.atom_densities(values), aliases)
            zero = torch.zeros_like(values)
            return torch.stack([atoms.get(name, zero) for name in nuclides], dim=-1) * 1e24

        inventory = densities(enrichment)
        if reference is not None:
            composition, native = (torch.as_tensor(value, dtype=enrichment.dtype,
                                                  device=enrichment.device).detach() for value in reference)
            inventory = native + (inventory - densities(composition))
        mass = sum(inventory[:, nuclides.index(name)] * openmc.data.atomic_mass(name)
                   for name in ("U235", "U238"))
        mass = mass * volumes / (1000. * openmc.data.AVOGADRO)
        return inventory, volumes, mass


def contract_isotopes(number_density_atoms_cm3, microscopic_b):
    """Contract isotope densities and microscopic reaction tables.

    Parameters
    ----------
    number_density_atoms_cm3 : torch.Tensor
        Region-by-nuclide number densities in atoms/cm³.
    microscopic_b : dict of str to torch.Tensor
        Reaction tables in barns with leading region and nuclide axes.

    Returns
    -------
    dict
        Region-wise macroscopic reaction tables in cm⁻¹, preserving density derivatives.
    """
    density = number_density_atoms_cm3 * 1e-24
    return {name: torch.einsum("zi,zi...->z...", density, values)
            for name, values in microscopic_b.items()}


def reaction_rates(microscopic_b, group_flux_cm2_s):
    """Collapse microscopic cross sections into per-parent reaction rates.

    Parameters
    ----------
    microscopic_b : torch.Tensor
        Microscopic data in barns with region, nuclide, reaction and energy-group axes.
    group_flux_cm2_s : torch.Tensor
        Region-by-group neutron flux in cm⁻² s⁻¹.

    Returns
    -------
    torch.Tensor
        Region-by-nuclide-by-reaction rates in s⁻¹.
    """
    return (microscopic_b * group_flux_cm2_s[:, None, None, :] * 1e-24).sum(-1)


def density_change_tables(density_change, microscopic, region_indices, region_count):
    """Contract density changes with microscopic data for selected regions.

    Parameters
    ----------
    density_change : array_like
        Selected-region-by-nuclide density changes in atoms/(barn cm).
    microscopic : dict
        Microscopic tables in barns with selected-region, nuclide and group axes.
    region_indices : sequence of int
        Unique destination region indices for the selected density rows.
    region_count : int
        Number of regions in the full native case.

    Returns
    -------
    dict
        Full-region macroscopic changes in cm⁻¹, with exactly zero changes outside the
        selected regions.

    Raises
    ------
    ValueError
        Region indices are invalid or the microscopic and density axes disagree.

    Notes
    -----
    Density units are atom/barn-cm and microscopic units are barns. Axes are selected
    region, nuclide, incoming group[, outgoing group]. Unselected regions receive exactly
    zero change. No nuclide has an implicit role.
    """
    density = tensor(density_change)
    indices = torch.as_tensor(region_indices, dtype=torch.long, device=density.device)
    if (density.ndim != 2 or indices.shape != density.shape[:1] or
            len(set(indices.tolist())) != len(indices) or bool(((indices < 0) | (indices >= region_count)).any())):
        raise ValueError('Density changes need unique, valid region indices')
    result = {}
    for name, value in microscopic.items():
        value = torch.as_tensor(value, dtype=density.dtype, device=density.device)
        if value.shape[:2] != density.shape:
            raise ValueError(f'Microscopic axes differ from number densities: {name}')
        selected = torch.einsum('mn,mn...->m...', density, value)
        result[name] = selected.new_zeros((region_count, *selected.shape[1:])).index_add(0, indices, selected)
    return result


def microscopic_to_macroscopic(atom_density, microscopic, chi):
    """Collapse microscopic data for arbitrary material-region assignments.

    Parameters
    ----------
    atom_density : array_like
        Region-by-nuclide densities in atoms/(barn cm).
    microscopic : dict
        Microscopic reaction arrays in barns, with leading region and nuclide axes.
    chi : array_like
        Region-by-energy-group fission emission spectrum.

    Returns
    -------
    dict
        Macroscopic reaction tables in cm⁻¹, fission spectrum and reaction-consistent total
        and transport cross sections.
    """
    density = tensor(atom_density)
    macro = {name: torch.einsum('mn,mn...->m...', density, tensor(value))
             for name, value in microscopic.items()}
    macro['chi'] = tensor(chi)
    macro['total'] = macro['absorption']+macro['scatter'].sum(-1)
    macro['transport'] = macro['total']-macro['scatter_p1']
    return macro


def transport_atoms(atoms, aliases):
    """Convert positive aliased nuclide densities to transport scalars.

    Parameters
    ----------
    atoms : dict
        Nuclide densities as scalars or tensors; all entries use a common density unit.
    aliases : dict of str to str
        Explicit source-to-transport nuclide substitutions; unmapped names are retained.

    Returns
    -------
    dict of str to float
        Positive transport densities in the original density unit; nonpositive entries are
        omitted.
    """
    return {name: float(value) for name, value in map_nuclides(atoms, aliases).items() if value > 0}
