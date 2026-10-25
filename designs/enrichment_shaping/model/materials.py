"""Cold UO2 and light-water definitions; isotope conversion is shared."""

import openmc.data

from materials.composition import UO2Recipe


def fuel_recipe(settings):
    """Bind the shared uranium-oxide recipe to the declared fuel density.

    Parameters
    ----------
    settings : dict
        The material settings from model.toml.

    Returns
    -------
    materials.composition.UO2Recipe
        U235/U238/O16 densities as a differentiable function of enrichment.
    """
    return UO2Recipe(settings["fuel_density_g_cm3"], 1., settings["oxygen_atom_fractions"])


def water_densities(settings):
    """Convert the declared water mass density to H1 and O16 atom densities.

    Parameters
    ----------
    settings : dict
        Material settings with water_density_g_cm3.

    Returns
    -------
    dict of str to float
        H1 and O16 number densities in atoms/(barn cm).
    """
    molecules = (settings["water_density_g_cm3"] * openmc.data.AVOGADRO * 1e-24
                 / (2 * openmc.data.atomic_mass("H1") + openmc.data.atomic_mass("O16")))
    return {"H1": 2 * molecules, "O16": molecules}
