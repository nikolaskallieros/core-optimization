"""Fresh UO2 loading and an exact, bounded radius projection."""
import math

from optimization.solvers import project_quadratic_bound


def fuel_mass_kg(radius_mm, basis):
    """Compute full-core fresh UO2 mass from physical ring radii.

    Parameters
    ----------
    radius_mm : sequence of float or torch.Tensor
        Homogenized fuel-element radii in mm, ordered by physical fuel ring.
    basis : dict
        Declared physical counts, UO2 density, homogenized fuel fraction and atomic-mass
        constants.

    Returns
    -------
    float or torch.Tensor
        Fresh oxide mass in kg, preserving derivatives for tensor radii.

    Raises
    ------
    ValueError
        The radius vector and physical ring weights have different lengths.
    """
    weights = basis["fuel_mass_kg_per_mm2_by_ring"]
    if len(radius_mm) != len(weights):
        raise ValueError("Fuel radii and physical ring counts differ")
    return sum(weight * radius * radius for weight, radius in zip(weights, radius_mm))


def heavy_metal_mass_kg(radius_mm, enrichment, basis):
    """Compute the initial uranium mass within the fresh UO2 loading.

    Parameters
    ----------
    radius_mm : sequence of float or torch.Tensor
        Homogenized fuel-element radii in mm, ordered by physical fuel ring.
    enrichment : sequence of float or torch.Tensor
        U-235 mass fractions within uranium, in the same ring order as the radii.
    basis : dict
        Declared physical counts, UO2 density, homogenized fuel fraction and atomic-mass
        constants.

    Returns
    -------
    float or torch.Tensor
        Initial heavy-metal mass in kg, preserving radius and enrichment derivatives.
    """
    masses = basis["atomic_mass_u"]
    if len(enrichment) != len(radius_mm):
        raise ValueError("Fuel radii and enrichments differ")
    total = 0.
    for weight, radius, fraction in zip(basis["fuel_mass_kg_per_mm2_by_ring"], radius_mm, enrichment):
        uranium = 1. / (fraction / masses["U235"] + (1. - fraction) / masses["U238"])
        total = total + weight * radius * radius * uranium / (uranium + 2. * masses["oxygen_mean"])
    return total


def fuel_mass_check(values, policy):
    """Check fresh-fuel mass against the declared full-core ceiling.

    Parameters
    ----------
    values : dict
        Design values containing the policy's radius and enrichment parameters.
    policy : dict
        Fresh-fuel mass policy containing a physical accounting basis and
        maximum_fresh_fuel_mass_kg.

    Returns
    -------
    dict
        Oxide and heavy-metal masses, mass-limit fraction, excess mass, roundoff tolerance
        and acceptance flag.

    Raises
    ------
    ValueError
        Radii, enrichments or the mass ceiling are physically invalid or nonfinite.
    """
    radii, enrichment = values[policy["radius_parameter"]], values[policy["enrichment_parameter"]]
    if (not all(math.isfinite(x) and x > 0 for x in radii)
            or not all(math.isfinite(x) and 0 < x < 1 for x in enrichment)):
        raise ValueError("Invalid fresh fuel design")
    mass = float(fuel_mass_kg(radii, policy["basis"]))
    limit = policy["maximum_fresh_fuel_mass_kg"]
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("The fresh fuel mass ceiling must be finite and positive")
    # Floating-point comparison allowance only; no physical mass allowance.
    tolerance = 32 * math.ulp(limit)
    return {"basis": "full_core_fresh_UO2", "fuel_mass_kg": mass,
            "initial_heavy_metal_mass_kg": float(heavy_metal_mass_kg(radii, enrichment, policy["basis"])),
            "maximum_fresh_fuel_mass_kg": limit, "mass_fraction_of_limit": mass / limit,
            "excess_fuel_mass_kg": max(0., mass - limit), "roundoff_tolerance_kg": tolerance,
            "passed": mass <= limit + tolerance}


def native_fuel_mass_check(case, policy):
    """Cross-check native volumes and isotope inventories against fuel loading.

    Parameters
    ----------
    case : dict
        Native case with physical multiplicities, fresh isotope densities, volumes and
        heavy-metal metadata.
    policy : dict
        Fresh-fuel mass policy containing a physical accounting basis and
        maximum_fresh_fuel_mass_kg.

    Returns
    -------
    dict
        Full-core mass check augmented with native accounting comparisons and physical
        assembly counts.

    Raises
    ------
    ArithmeticError
        Native stoichiometry, masses or physical assembly counts disagree with the declared
        basis.
    """
    basis = policy["basis"]
    rows = [region for region in case["regions"] if region["depletable"]]
    multiplier = case.get("symmetry_multiplier", 1)
    atomic = basis["atomic_mass_u"]
    atom_mass, atom_hm, declared_hm = 0., 0., 0.
    for region in rows:
        fresh = region["fresh_fuel_atom_density"]
        uranium = fresh.get("U235", 0.) + fresh.get("U238", 0.)
        oxygen = sum(value for name, value in fresh.items() if name.startswith("O"))
        if not math.isclose(oxygen, 2. * uranium, rel_tol=1e-12):
            raise ArithmeticError("Native fresh fuel no longer has the declared UO2 stoichiometry")
        full_volume = multiplier * region["volume_cm3"]
        u = sum(fresh.get(name, 0.) * atomic[name] for name in ("U235", "U238"))
        scale = 1e24 * full_volume / basis["avogadro_per_mol"] / 1000.
        atom_hm += u * scale
        atom_mass += (u + oxygen * atomic["oxygen_mean"]) * scale
        declared_hm += region["initial_hm_kg_per_assembly"] * region["physical_multiplicity"]
    full_volume = multiplier * sum(row["volume_cm3"] for row in rows)
    volume_mass = full_volume * basis["fuel_density_g_cm3"] * basis["fuel_volume_fraction_of_homogenized_element"] / 1000.
    expected = fuel_mass_check(case["design"], policy)
    comparisons = {"atom_vs_volume_fuel_relative_error": abs(atom_mass - volume_mass) / volume_mass,
                   "atom_vs_recorded_HM_relative_error": abs(atom_hm - declared_hm) / declared_hm,
                   "native_vs_design_fuel_relative_error": abs(volume_mass - expected["fuel_mass_kg"]) / volume_mass,
                   "native_vs_design_HM_relative_error": abs(atom_hm - expected["initial_heavy_metal_mass_kg"]) / atom_hm}
    if any(error > 1e-11 for error in comparisons.values()):
        raise ArithmeticError("Native fuel volumes, isotopic masses and design loading disagree")
    physical_assemblies = sum(region["physical_multiplicity"] for region in rows)
    if physical_assemblies != sum(basis["physical_channels_by_ring"]) * basis["axial_assemblies_per_channel"]:
        raise ArithmeticError("Native fresh fuel mass omitted or duplicated physical assemblies")
    return {**expected, "fuel_mass_kg": volume_mass, "initial_heavy_metal_mass_kg": atom_hm,
            "native_homogenized_element_volume_cm3": full_volume, "native_fuel_atom_mass_kg": atom_mass,
            "native_recorded_initial_HM_mass_kg": declared_hm, "physical_axial_assemblies": physical_assemblies,
            "native_material_accounting": comparisons,
            "passed": volume_mass <= expected["maximum_fresh_fuel_mass_kg"] + expected["roundoff_tolerance_kg"]}


def project_design_fuel_mass(design, anchor, config):
    """Apply the declared mass ceiling to a live design in place.

    Parameters
    ----------
    design : torch.nn.Module
        Design exposing the named radius tensor.
    anchor : dict
        Reference design values defining the current trust region.
    config : dict
        Optimization settings with fresh_fuel_mass, bounds and trust_half_width entries.
    """
    import torch

    policy = config["fresh_fuel_mass"]
    name = policy["radius_parameter"]
    lower, upper = config["bounds"][name]
    width = config["trust_half_width"][name]
    lo = [max(lower, x - width) for x in anchor[name]]
    hi = [min(upper, x + width) for x in anchor[name]]
    with torch.no_grad():
        radii = getattr(design, name)
        projected = project_quadratic_bound(
            radii.detach().tolist(), lo, hi,
            policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            policy["maximum_fresh_fuel_mass_kg"],
        )
        radii.copy_(radii.new_tensor(projected))
