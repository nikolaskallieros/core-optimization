"""Physical isotope contraction shared by diffusion and Bateman depletion."""
import torch
from materials.composition import contract_isotopes


class InventoryCrossSections:
    """Build the whole fuel material from live fuel and fixed inert inventories.

    The microscopic spectrum/temperature tables are native data at each stage.
    They are frozen locally; the number densities are always live. Scattering
    keeps its incoming/outgoing group matrix and first angular moment.
    """
    def __init__(self, response, regions, chain_names, *, aliases=None):
        """Bind live fuel inventories to native microscopic response tables.

        Parameters
        ----------
        response : dict
            Native microscopic tables in barns with their transport nuclide ordering.
        regions : sequence of dict
            Fuel regions in response order, each declaring inert_atom_density in atoms/(barn
            cm).
        chain_names : sequence of str
            Depletion-chain nuclide ordering used by live inventories.
        aliases : dict or None, optional
            Explicit chain-to-transport isotope substitutions.
        """
        from materials.composition import map_nuclides
        aliases = {} if aliases is None else aliases
        self.names = tuple(response["nuclides"])
        self.microscopic = {name: torch.as_tensor(value, dtype=torch.float64)
                            for name, value in response["microscopic"].items()}
        index = {name: i for i, name in enumerate(chain_names)}
        source, destination = [], []
        for j, name in enumerate(self.names):
            for isotope in (n for n in chain_names if aliases.get(n, n) == name):
                if isotope in index:
                    source.append(index[isotope])
                    destination.append(j)
        self.source = torch.tensor(source, dtype=torch.long)
        self.destination = torch.tensor(destination, dtype=torch.long)
        inert = [map_nuclides(region["inert_atom_density"], aliases) for region in regions]
        self.inert = torch.tensor([[atoms.get(name, 0.) * 1e24 for name in self.names]
                                   for atoms in inert], dtype=torch.float64)

    def densities(self, inventory):
        """Combine aliased live fuel and fixed inert nuclide densities.

        Parameters
        ----------
        inventory : torch.Tensor
            Fuel-region-by-chain-nuclide atom densities in atoms/cm³.

        Returns
        -------
        torch.Tensor
            Fuel-region-by-transport-nuclide densities in atoms/cm³.
        """
        live = self.inert.new_zeros(self.inert.shape).index_add(1, self.destination,
                                                             inventory[:, self.source])
        return self.inert + live

    def __call__(self, inventory):
        """Contract the complete fuel material into macroscopic reaction tables.

        Parameters
        ----------
        inventory : torch.Tensor
            Fuel-region-by-chain-nuclide atom densities in atoms/cm³.

        Returns
        -------
        dict
            Region-wise reaction cross sections in cm⁻¹, with kappa_fission in eV/cm and live
            inventory derivatives.
        """
        return contract_isotopes(self.densities(inventory), self.microscopic)
