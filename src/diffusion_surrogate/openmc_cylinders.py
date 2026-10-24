"""OpenMC transport and observations for cylindrical regions in hexagonal domains."""
from functools import reduce
import math
import operator
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

def build(case, directory):
    """Export a cylindrical OpenMC model with the shared tally conventions.

    Parameters
    ----------
    case : dict
        Native cylindrical case with material regions, hexagonal domain, sampling and
        required response/depletion tallies.
    directory : str or pathlib.Path
        Calculation directory containing the OpenMC XML and native output files.

    Returns
    -------
    tuple
        OpenMC model and materials in native region order.

    Raises
    ------
    ValueError
        The symmetry boundary or requested depletion estimator is unsupported.
    """
    import openmc
    from diffusion_surrogate.native3d import material_list

    openmc.reset_auto_ids()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec, sampling = case["geometry"], case["sampling"]
    materials = material_list(case)
    for i in case.get("boron_response", {}).get("region_indices", []):
        for name in case["boron_response"]["nuclides"]:
            if name not in materials[i].get_nuclides():
                materials[i].add_nuclide(name, 0.)
    boundaries = spec.get("boundaries", ["vacuum"] * 6)
    bottom = openmc.Plane(a=0., b=0., c=-1., d=-spec["lower_cm"][2], boundary_type=boundaries[4])
    top = openmc.ZPlane(z0=spec["upper_cm"][2], boundary_type=boundaries[5])
    if spec["symmetry"] == "c6":
        a = spec["apothem_cm"]
        symmetry_boundary = spec.get("symmetry_boundary_type", "periodic")
        if symmetry_boundary not in ("reflective", "periodic"):
            raise ValueError("Sector faces must use a symmetry boundary")
        lower = openmc.Plane(a=.5, b=math.sqrt(3) / 2, boundary_type=symmetry_boundary)
        upper = openmc.Plane(a=.5, b=-math.sqrt(3) / 2, boundary_type=symmetry_boundary)
        if symmetry_boundary == "periodic":
            lower.periodic_surface = upper
        radial = [openmc.XPlane(x0=a, boundary_type=boundaries[0])]
        domain = +lower & +upper & -radial[0] & -bottom & -top
        periodic = [lower, upper] if symmetry_boundary == "periodic" else []
    else:
        a = spec["apothem_cm"]
        radial = [openmc.Plane(a=math.cos(i * math.pi / 3), b=math.sin(i * math.pi / 3),
                              d=a, boundary_type=boundaries[0]) for i in range(6)]
        domain = reduce(operator.and_, [-p for p in radial]) & -bottom & -top
        periodic = []
    cylinders = {}

    def cylinder(piece, name):
        """Reuse a cylindrical surface for a given center and interface radius.

        Parameters
        ----------
        piece : dict
            Cylinder-piece record with xy_cm and named interface radii in cm.
        name : str
            Radius-field key, such as inner_radius_cm or outer_radius_cm.

        Returns
        -------
        openmc.ZCylinder
            Cached axial cylinder surface.
        """
        radius = piece[name]
        key = (*piece["xy_cm"], radius)
        if key not in cylinders:
            cylinders[key] = openmc.ZCylinder(x0=key[0], y0=key[1], r=radius)
        return cylinders[key]

    z_planes = {z: openmc.ZPlane(z0=z) for entry in case["regions"] for z in entry["z_bounds_cm"]}
    z_planes[spec["lower_cm"][2]] = bottom
    z_planes[spec["upper_cm"][2]] = top
    cells, sources = [], []
    for i, entry in enumerate(case["regions"]):
        low, high = (z_planes[z] for z in entry["z_bounds_cm"])
        region = domain & (-low if low is bottom else +low) & -high
        if "pieces" in entry:
            pieces = []
            for piece in entry["pieces"]:
                part = -cylinder(piece, "outer_radius_cm")
                if piece["inner_radius_cm"]:
                    part &= +cylinder(piece, "inner_radius_cm")
                pieces.append(part)
            region &= reduce(operator.or_, pieces)
        else:
            for shape in spec["channel_shapes"]:
                # Sleeve thickness is local to an axial position, not full height.
                if "z_bounds_cm" not in shape or shape["z_bounds_cm"] == entry["z_bounds_cm"]:
                    region &= +cylinder(shape, "outer_radius_cm")
        cell = openmc.Cell(i + 1, name=entry["label"], region=region,
                           fill=None if entry["void"] else materials[i])
        cells.append(cell)
        if entry["depletable"]:
            for piece in entry["pieces"]:
                source = openmc.IndependentSource(space=openmc.stats.CylindricalIndependent(
                    r=openmc.stats.PowerLaw(0., piece["outer_radius_cm"], 1),
                    phi=openmc.stats.Uniform(0., 2 * math.pi),
                    z=openmc.stats.Uniform(*entry["z_bounds_cm"]), origin=(*piece["xy_cm"], 0.)),
                    constraints={"domains": [cell]}, strength=entry["volume_cm3"] / len(entry["pieces"]))
                source.energy = openmc.stats.Discrete([2e6], [1.])
                sources.append(source)
    model = openmc.Model(geometry=openmc.Geometry(cells),
                         materials=openmc.Materials([m for m, r in zip(materials, case["regions"]) if not r["void"]]))
    model.materials.cross_sections = case["cross_sections"]
    settings = model.settings
    settings.run_mode = "eigenvalue"
    settings.particles, settings.batches, settings.inactive = (sampling[key] for key in ("particles", "batches", "inactive"))
    settings.seed, settings.generations_per_batch = sampling["seed"], 1
    # OpenMC's table labels round temperatures to integer kelvin. The HDF5 kT
    # values were independently checked against the requested processing values.
    settings.temperature = {"method": "interpolation",
                            "tolerance": case["temperature_label_tolerance_K"]}
    if case.get("need_depletion_rates"):
        # A parent already loaded in a cold structural/moderator material must
        # also have the fuel-temperature table available for rate collapse.
        fuel_temperatures = [r["temperature_K"] for r in case["regions"] if r["depletable"]]
        settings.temperature["range"] = [min(fuel_temperatures), max(fuel_temperatures)]
    settings.ptables, settings.resonance_scattering = True, {"enable": False}
    settings.source = sources
    settings.output = {"path": ".", "summary": True, "tallies": False}
    settings.statepoint, settings.sourcepoint = {"batches": [sampling["batches"]]}, {"write": False}
    settings.entropy_mesh = openmc.RegularMesh()
    settings.entropy_mesh.lower_left, settings.entropy_mesh.upper_right = spec["lower_cm"], spec["upper_cm"]
    settings.entropy_mesh.dimension = [5, 5, 5]
    comparison = openmc.RectilinearMesh()
    comparison.x_grid, comparison.y_grid, comparison.z_grid = case["comparison_edges_cm"]
    domain_filter, mesh_filter = openmc.CellFilter(cells), openmc.MeshFilter(comparison)
    energy, outgoing = openmc.EnergyFilter(case["group_edges_eV"]), openmc.EnergyoutFilter(case["group_edges_eV"])
    legendre = openmc.LegendreFilter(1)
    tallies = []

    def tally(uid, filters, scores, estimator="tracklength", names=None, microscopic=False):
        """Append a native tally with explicit filters, estimator and isotope axes.

        Parameters
        ----------
        uid : int
            Stable native tally identifier used by the extraction interface.
        filters : sequence of openmc.Filter
            Ordered tally filters.
        scores : sequence of str
            Requested OpenMC reaction or response scores.
        estimator : str, optional
            OpenMC estimator, normally tracklength or analog.
        names : sequence of str or None, optional
            Nuclide bins; None requests the total material score.
        microscopic : bool, optional
            Disable density multiplication for microscopic reaction scoring.
        """
        item = openmc.Tally(uid)
        item.filters, item.scores, item.estimator = filters, scores, estimator
        if names:
            item.nuclides = names
        if microscopic:
            item.multiply_density = False
        tallies.append(item)

    tally(1, [mesh_filter], ["kappa-fission", "nu-fission"])
    tally(2, [domain_filter, energy], ["flux"])
    tally(3, [domain_filter, energy], ["total", "absorption", "fission", "nu-fission", "kappa-fission"])
    tally(4, [domain_filter, energy, outgoing], ["scatter", "nu-scatter"], "analog")
    tally(5, [domain_filter, energy, legendre], ["scatter"], "analog")
    # A tracklength denominator also exists in void regions. Its expected
    # integral matches analog flux, and zero void reaction rates remain exact.
    tally(6, [domain_filter, energy], ["flux"])
    tally(7, [domain_filter, outgoing], ["nu-fission"], "analog")
    tally(8, [mesh_filter, energy], ["flux"])
    tally(10, [domain_filter, energy], ["absorption", "fission"])
    tally(11, [domain_filter, energy, outgoing], ["scatter"], "analog")
    response = case.get("material_response")
    if response:
        selected = openmc.CellFilter([cells[i] for i in response["region_indices"]])
        tally(16, [selected, energy], ["total", "absorption", "fission", "nu-fission", "kappa-fission"],
              names=response["nuclides"], microscopic=True)
        tally(17, [selected, energy, outgoing], ["scatter", "nu-scatter"], "analog", names=response["nuclides"])
        tally(18, [selected, energy, legendre], ["scatter"], "analog", names=response["nuclides"])
    tally(19, [openmc.SurfaceFilter(radial + [bottom, top]), energy], ["current"], "analog")
    if periodic:
        tally(20, [openmc.SurfaceFilter(periodic), energy], ["current"], "analog")
    tally(21, [domain_filter], ["kappa-fission", "nu-fission"])
    if case.get("boron_response"):
        boron_response = case["boron_response"]
        selected = openmc.CellFilter([cells[i] for i in boron_response["region_indices"]])
        tally(22, [selected, energy], ["total", "absorption", "fission", "nu-fission", "kappa-fission"],
              names=boron_response["nuclides"], microscopic=True)
        tally(23, [selected, energy, outgoing], ["scatter", "nu-scatter"], "analog", names=boron_response["nuclides"])
        tally(24, [selected, energy, legendre], ["scatter"], "analog", names=boron_response["nuclides"])
    if case.get("need_depletion_rates"):
        from diffusion_surrogate.native3d import depletion_edges, FLUX_RATES
        if case["depletion_rate_estimator"] != FLUX_RATES:
            raise ValueError("This transport driver uses flux collapse with direct key reaction rates")
        depleted = openmc.CellFilter([cells[i] for i, r in enumerate(case["regions"]) if r["depletable"]])
        tally(9, [depleted], case["depletion_direct_reactions"],
              names=case["depletion_direct_nuclides"], microscopic=True)
        tally(12, [depleted, openmc.EnergyFilter(depletion_edges(case))], ["flux"])
        if case.get("need_depletion_response"):
            tally(25, [depleted, energy], case["depletion_direct_reactions"],
                  names=case["depletion_direct_nuclides"], microscopic=True)
        if case.get("physical_inventory_response"):
            tally(26, [depleted, energy], case["depletion_reactions"],
                  names=case["material_response"]["nuclides"], microscopic=True)
    model.tallies = openmc.Tallies(tallies)
    model.export_to_xml(directory)
    return model, materials


def run(case, directory):
    """Build and run native OpenMC, then extract regional transport responses.

    Parameters
    ----------
    case : dict
        Native cylindrical case with material regions, hexagonal domain, sampling and
        required response/depletion tallies.
    directory : str or pathlib.Path
        Calculation directory containing the OpenMC XML and native output files.

    Returns
    -------
    dict
        Completed native observations, microscopic/material responses and optional physical
        depletion rates.
    """
    from diffusion_surrogate.reference import native_batches

    directory = Path(directory)
    tick = time.perf_counter()
    model, materials = build(case, directory)
    collapsed = {}

    def postprocess(dll):
        """Save collapsed depletion scores while native nuclide data remain loaded.

        Parameters
        ----------
        dll : ctypes.CDLL
            Initialized OpenMC library with the completed model and tally data.
        """
        from materials.inventory import collapse_rates
        collapsed.update(collapse_rates(dll, case))
        np.savez_compressed(directory / "depletion_spectrum.npz", **collapsed)

    batches = native_batches(directory, case["sampling"]["inactive"], threads=case["sampling"]["threads"],
                             batch_checkpoint=directory / "batch_scores.npz",
                             postprocess=postprocess if case.get("need_depletion_rates") else None)
    return extract_completed(case, directory, batches=batches, materials=materials, started=tick,
                             collapsed=collapsed)


def extract_completed(case, directory, *, batches=None, materials=None, started=None, collapsed=None):
    """Recover completed cylindrical transport and composition responses.

    Parameters
    ----------
    case : dict
        Native cylindrical case with material regions, hexagonal domain, sampling and
        required response/depletion tallies.
    directory : str or pathlib.Path
        Calculation directory containing the OpenMC XML and native output files.
    batches : numpy.ndarray or None, optional
        Active-batch scores; None loads batch_scores.npz from the calculation directory.
    materials : sequence or None, optional
        Native materials in region order; None reconstructs them from the case.
    started : float or None, optional
        perf_counter start time for elapsed-time accounting.
    collapsed : dict or None, optional
        Saved native collapse scores; None loads the spectrum artifact when depletion
        responses are required.

    Returns
    -------
    dict
        Native results augmented with regional powers, leakage, composition responses and
        physical geometry identities.

    Raises
    ------
    ArithmeticError
        Group depletion scores or isotope contractions disagree with the native total
        reactions.
    """
    import openmc
    from diffusion_surrogate.native3d import extract, material_list, tally_cross_sections

    directory = Path(directory)
    batches = np.load(directory / "batch_scores.npz")["batch_scores"] if batches is None else batches
    materials = material_list(case) if materials is None else list(materials)
    for i, region in enumerate(case["regions"]):
        if region["void"]:
            # An empty openmc.Material cannot calculate its average molar mass.
            # The transport cell was filled with None, and extraction needs only
            # these metadata plus the correct empty number-density mapping.
            materials[i] = SimpleNamespace(name=region["label"], volume=region["volume_cm3"],
                                           depletable=False, get_nuclide_atom_densities=lambda: {})
    tick = time.perf_counter() if started is None else started
    if case.get("need_depletion_rates") and collapsed is None:
        collapsed = dict(np.load(directory / "depletion_spectrum.npz"))
    record = extract(case, directory, batches, materials=materials, started=tick,
                     collapsed=collapsed, factorized_statistics=True)
    with openmc.StatePoint(directory / f"statepoint.{case['sampling']['batches']}.h5") as state:
        leak = state.get_tally(id=19)
        record["observables"]["leakage_per_source"] = {
            "mean": leak.mean.reshape(-1, len(case["group_edges_eV"]) - 1).sum(0),
            "scale": leak.std_dev.reshape(-1, len(case["group_edges_eV"]) - 1).sum(0)}
        regional = state.get_tally(id=21)
        heating = regional.mean[:, 0, 0]
        heating_sd = regional.std_dev[:, 0, 0]
        norm = case["power_W"] / heating.sum()
        volumes = np.array([r["volume_cm3"] for r in case["regions"]])
        record["observables"]["region_power_W"] = {"mean": norm * heating, "scale": norm * heating_sd}
        record["observables"]["region_power_density_W_cm3"] = {"mean": norm * heating / volumes,
                                                                "scale": norm * heating_sd / volumes}
        flux = record["observables"]["region_flux_per_source"]
        record["observables"]["region_mean_flux_per_source"] = {"mean": flux["mean"] / volumes[:, None],
                                                                 "scale": flux["scale"] / volumes[:, None]}
        if case.get("need_depletion_response"):
            fuel = record["depletion_region_indices"]
            groups = len(case["group_edges_eV"]) - 1
            rate_groups = collapsed["rates_by_group"].copy()
            direct = state.get_tally(id=25).mean.reshape(len(fuel), groups,
                        len(case["depletion_direct_nuclides"]), len(case["depletion_direct_reactions"]))
            direct = direct.transpose(0, 2, 3, 1)
            for j, name in enumerate(case["depletion_direct_nuclides"]):
                parent = case["depletion_nuclides"].index(name)
                for k, reaction in enumerate(case["depletion_direct_reactions"]):
                    rate_groups[:, parent, case["depletion_reactions"].index(reaction)] = direct[:, j, k]
            rate_groups *= record["source_rate_per_s"] / (1e24 * volumes[fuel, None, None, None])
            if not np.allclose(rate_groups.sum(-1), record["depletion_rates_per_parent_s"], rtol=1e-11, atol=0):
                raise ArithmeticError("Group-resolved direct depletion scores differ from their total tally")
            if case.get("physical_inventory_response"):
                response = record["material_response"]
                names = response["nuclides"]
                reaction_tally = state.get_tally(id=26).mean.reshape(
                    len(fuel), groups, len(names), len(case["depletion_reactions"])).transpose(0, 2, 3, 1)
                microscopic = reaction_tally / flux["mean"][fuel, None, None, :]
                density = np.asarray(response["atom_density"])
                # Direct reaction scores and diffusion material scores sample
                # the same isotope, tracks and spectrum. Histogram collapse
                # remains available for parents absent from this native fuel.
                for j, name in enumerate(names):
                    if name in case["depletion_nuclides"]:
                        parent = case["depletion_nuclides"].index(name)
                        direct_rates = reaction_tally[:, j] * record["source_rate_per_s"] / (1e24 * volumes[fuel, None, None])
                        rate_groups[:, parent] = np.where(density[:, j, None, None] > 0,
                                                          direct_rates, rate_groups[:, parent])
                record["depletion_rates_per_parent_s"] = rate_groups.sum(-1)
                response.update(reaction_microscopic_b=microscopic,
                                reaction_order=case["depletion_reactions"],
                                reconstruction="Absolute isotope contraction, including fixed inert constituents.")
                reconstructed = {name: np.einsum("zi,zi...->z...", density, value)
                                 for name, value in response["microscopic"].items()}
                errors = {name: float(np.max(abs(value - record["macro"][name][fuel]))
                                     / max(float(np.max(abs(record["macro"][name][fuel]))), 1e-30))
                          for name, value in reconstructed.items()}
                fission = microscopic[:, :, case["depletion_reactions"].index("fission")]
                fission_error = float(np.max(abs(fission - response["microscopic"]["fission"]))
                                      / max(float(np.max(abs(fission))), 1e-30))
                if max(*errors.values(), fission_error) > 1e-9:
                    raise ArithmeticError("Isotope reconstruction or diffusion/depletion fission data disagree")
                record["physical_inventory_checks"] = {
                    "status": "checked", "macro_reconstruction_relative_errors": errors,
                    "diffusion_depletion_fission_relative_error": fission_error,
                    "tolerance": 1e-9,
                    "scope": "All fuel-region absorption, fission and scattering tables reconstruct from isotope densities. This checks data consistency, not Monte Carlo convergence."}
            record["depletion_response"] = {
                "per_parent_s_by_group": rate_groups,
                "mean_group_flux_cm2_s": flux["mean"][fuel] * record["source_rate_per_s"] / volumes[fuel, None],
                "group_edges_eV": case["group_edges_eV"],
                "scope": "Native fine-spectrum rates resolved for local multigroup flux sensitivities; selected capture/fission use direct group tallies. A straddling fine histogram bin is split by energy width."}
            if case.get("physical_inventory_response"):
                record["depletion_response"]["microscopic_b"] = (rate_groups * 1e24
                    / record["depletion_response"]["mean_group_flux_cm2_s"][:, None, None, :])
                record["depletion_response"]["scope"] = "Physical microscopic group cross sections times flux. Every chain reaction of a present fuel-element nuclide uses direct scores; absent parents retain evaluated-data histogram collapse. Spectra are refreshed by OpenMC at each state and frozen locally during differentiation."
        if case.get("boron_response"):
            selected = np.array(case["boron_response"]["region_indices"])
            nuclides = case["boron_response"]["nuclides"]
            bflux = flux["mean"][selected]
            density = np.array([[materials[i].get_nuclide_atom_densities().get(n, 0.) for n in nuclides]
                                for i in selected])
            response = tally_cross_sections(state, (22, 23, 24), bflux, bflux, len(nuclides), atom_density=density)
            record["boron_response"] = {"region_indices": selected, "nuclides": nuclides,
                                        "atom_density": density, "microscopic": response, "analog_sampled": density > 0}
        if (case["geometry"]["symmetry"] == "c6"
                and case["geometry"].get("symmetry_boundary_type", "periodic") == "periodic"):
            periodic = state.get_tally(id=20)
            record["periodic_current_per_source"] = periodic.mean.reshape(2, -1)
            record["periodic_current_scale"] = periodic.std_dev.reshape(2, -1)
    record.update({key: case[key] for key in ("geometry", "design", "geometry_checks", "physical_assembly_to_region",
                                             "effective_approximations") if key in case},
                  boron_mg_B10_per_kg=case.get("boron_mg_B10_per_kg", 0.),
                  cross_section_flux_estimator="tracklength, including void regions")
    return record
