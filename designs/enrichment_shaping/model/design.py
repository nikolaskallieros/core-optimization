"""The enrichment field and the corresponding physical U235 inventory."""

import openmc.data
import torch
from copy import deepcopy

from .materials import fuel_recipe


class LatticeSize(torch.nn.Module):
    """One common transverse dimension while preparing the critical reference."""

    def __init__(self, config, scale=1.):
        """Bind the physical input dimensions to one live scale parameter.

        Parameters
        ----------
        config : dict
            Source physical dimensions and fixed uniform enrichment.
        scale : float, optional
            Initial common multiplier on fuel radii and lattice pitches.
        """
        super().__init__()
        self.config = deepcopy(config)
        self.radial_scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

    def configuration(self, *, native=False):
        """Resolve dimensions for the shared CAD and its two numerical adapters.

        Parameters
        ----------
        native : bool, optional
            Return detached Python values for native transport and saved inputs.
            The default retains the live Torch geometry derivatives.

        Returns
        -------
        dict
            Complete configuration with scaled physical pitches and fuel radius.
        """
        config = deepcopy(self.config)
        scale = self.radial_scale
        pitch = torch.as_tensor(config["geometry"]["pitch_cm"], dtype=torch.float64) * scale
        radius = config["geometry"]["fuel_radius_cm"] * scale
        config["geometry"].update(pitch_cm=pitch.detach().tolist() if native else pitch,
                                  fuel_radius_cm=float(radius.detach()) if native else radius)
        return config


class Enrichment(torch.nn.Module):
    """One physical enrichment coordinate per fuel pin and axial segment."""

    def __init__(self, config, values, fuel_volumes_cm3):
        """Bind physical enrichment values to the shared UO2 isotope recipe.

        Parameters
        ----------
        config : dict
            Resolved model.toml configuration.
        values : torch.Tensor
            Pin-by-axial-segment enrichments in wt% U235 within uranium.
        fuel_volumes_cm3 : torch.Tensor
            Ordered physical segment volumes in cm³.
        """
        super().__init__()
        self.enrichment_wt_percent = torch.nn.Parameter(values.clone())
        self.recipe = fuel_recipe(config["materials"])
        self.register_buffer("fuel_volumes_cm3", fuel_volumes_cm3)

    def isotope_densities(self, names, region_indices):
        """Evaluate fresh densities in explicitly requested native isotope order.

        Parameters
        ----------
        names : sequence of str
            Isotope order supplied by the native material response.
        region_indices : sequence of int
            Selected fuel segments in physical region order.

        Returns
        -------
        torch.Tensor
            Selected-region-by-isotope densities in atoms/(barn cm).
        """
        values = self.enrichment_wt_percent.flatten()[region_indices] / 100
        densities = self.recipe.atom_densities(values)
        return torch.stack([densities[name] for name in names], dim=-1)

    def u235_mass_kg(self):
        """Compute total U235 mass from physical volumes and live isotope densities.

        Returns
        -------
        torch.Tensor
            Scalar total U235 mass in kg, retaining all enrichment derivatives.
        """
        atoms = self.recipe.atom_densities(self.enrichment_wt_percent.flatten() / 100)["U235"]
        return (atoms * self.fuel_volumes_cm3).sum() * 1e24 * openmc.data.atomic_mass("U235") / (1000 * openmc.data.AVOGADRO)
