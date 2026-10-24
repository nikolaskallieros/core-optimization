"""WR-1 fuel recipes and loading policy, using resolved model inputs."""
import numpy as np
import torch
from materials.composition import UO2Recipe


def fuel_recipe(definition):
    """Create the UO2 recipe from the resolved physical model.

    Parameters
    ----------
    definition : dict
        Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.

    Returns
    -------
    UO2Recipe
        Recipe using the declared density, homogenized volume fraction and oxygen isotope
        fractions.
    """
    fuel = definition["fuel"]
    return UO2Recipe(fuel["density_g_cm3"], fuel["volume_fraction"], fuel["oxygen_atom_fractions"])


def fuel_volumes(design, case):
    """Evaluate the live volumes of native depletable regions.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    case : dict
        Native region, material and symmetry description returned by ``make_case``.

    Returns
    -------
    torch.Tensor
        One volume in cm³ per fuel region, including physical multiplicity and the modeled
        symmetry fraction.
    """
    mapped = design()
    indices = {name: i for i, name in enumerate(design.assembly_ids)}
    regions = [region for region in case["regions"] if region["depletable"]]
    return torch.stack([mapped["assembly_volume_cm3"][indices[region["physical_assembly_ids"][0]]]
                        * region["physical_multiplicity"] / case.get("symmetry_multiplier", 6)
                        for region in regions])


def live_recipe(physical, design, recipe, aliases):
    """Build live fresh inventories anchored to the native reference loading.

    Parameters
    ----------
    physical : IsotopicInventory
        Reference physical inventory with fuel regions, isotope names and fresh atom
        densities.
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    recipe : UO2Recipe
        Resolved fresh-fuel density and isotope recipe.
    aliases : dict
        Nuclide substitutions used by the transport library.

    Returns
    -------
    tuple
        Fresh atom densities in atoms/cm³, live fuel volumes in cm³ and initial heavy-metal
        masses in kg.
    """
    rings = torch.tensor([region["ring"] for region in physical.regions])
    return recipe.live_inventory(design.enrichment[rings], physical.names,
        fuel_volumes(design, physical.base_case), aliases,
        reference=(torch.tensor(physical.design_values["enrichment"], dtype=torch.float64)[rings], physical.fresh))


def initialize_loading(values, keff, options):
    """Increase enrichment toward the declared beginning-of-life target.

    Parameters
    ----------
    values : dict
        Complete named design parameters in their declared units and array shapes.
    keff : float
        Current beginning-of-life effective multiplication factor.
    options : dict
        Loading policy containing bol_target_keff and maximum_enrichment.

    Returns
    -------
    dict
        Proposed design with enrichment raised within the declared bound.

    Raises
    ------
    ValueError
        The enrichment bound prevents any further increase.
    """
    enrichment = np.asarray(values["enrichment"])
    increment = max(.01, .1 * (options["bol_target_keff"] - keff))
    proposed = np.minimum(enrichment + increment, options["maximum_enrichment"])
    if np.array_equal(proposed, enrichment):
        raise ValueError(f"No critical loading within the declared enrichment bound: k={keff:.5f}")
    return {**values, "enrichment": proposed.tolist()}
