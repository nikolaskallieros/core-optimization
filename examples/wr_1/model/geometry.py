"""WR-1 ring parameters for the shared Torch design interface.

This maps design parameters to individual assembly cylinders. It does not
replace the C6 transport boundary or collapse independent assembly histories.
"""
import math

import torch
from diffusion_surrogate.geometry import DiffusionGeometry
from diffusion_surrogate.hexagonal import C6Grid
from torch import nn
import numpy as np
from runtime.paths import NUCLEAR_DATA
from materials.composition import transport_atoms
from diffusion_surrogate.cylinders import (
    Cylinder, HexagonalDomain, channel_regions, cylinder_clearances, cylinder_fractions,
)
from .materials import fuel_recipe


class ChannelDesign(nn.Module):
    """Map declared WR-1 lattice sites and ring parameters to axial assemblies."""
    def __init__(self, definition, specification):
        """Bind resolved model definitions to trainable channel geometry.

        Parameters
        ----------
        definition : dict
            Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
        specification : dict
            Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.

        Notes
        -----
        The case adapter loads model.toml and study.toml once and passes their resolved
        dictionaries here. This constructor registers design parameters and fixed lattice
        buffers without reading files.
        """
        super().__init__()
        for name, value in specification["design"].items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value, dtype=torch.float64)))
        sites = definition["lattice"]["sites"]
        channel_types = definition["channel_types"]
        channels = [channel_types[site["channel_type"]] for site in sites]
        self.channel_ids = tuple(site["id"] for site in sites)
        rings = [max(abs(site["q"]), abs(site["r"]), abs(site["q"] + site["r"])) for site in sites]
        self.register_buffer("ring", torch.tensor(rings, dtype=torch.long))
        self.register_buffer("fuelled", torch.tensor([channel["fuelled"] for channel in channels], dtype=torch.bool))
        self.register_buffer("lattice_xy", torch.tensor([
            [site["q"] + site["r"] / 2, math.sqrt(3) * site["r"] / 2] for site in sites], dtype=torch.float64))
        self.axial_count = specification["axial_positions"]
        self.register_buffer("z_edges_mm", torch.arange(self.axial_count + 1, dtype=torch.float64)
                             * specification["assembly_height_mm"])
        self.reference_interfaces = tuple(tuple(channel["reference_interfaces_mm"]) for channel in channels)
        self.channel_materials = tuple(tuple(channel["materials"]) for channel in channels)
        self.assembly_ids = tuple(f"{site}/axial/{level}" for site, fuelled in zip(self.channel_ids, self.fuelled)
                                  if fuelled for level in range(self.axial_count))

    def ring_pitches(self):
        """Return the live radial ring-spacing increments.

        Returns
        -------
        torch.Tensor
            Ring-spacing increments in mm, in outward ring order.
        """
        return self.pitch_mm

    def forward(self):
        """Map live ring parameters to channel and axial assembly geometry.

        Returns
        -------
        dict
            Channel coordinates and interfaces in mm; assembly coordinates, radii and axial
            bounds in cm; assembly volumes in cm³ and fresh enrichment mass fractions. Axial
            assembly identities run top to bottom.

        Raises
        ------
        ValueError
            A pitch, radius or enrichment is nonfinite or outside its physical range.
        """
        if (not bool(torch.isfinite(self.pitch_mm).all() & torch.isfinite(self.element_radius_mm).all()
                     & torch.isfinite(self.enrichment).all())
                or bool((self.pitch_mm <= 0).any() | (self.element_radius_mm <= 0).any()
                        | (self.enrichment <= 0).any() | (self.enrichment >= 1).any())):
            raise ValueError("Invalid ring pitch, homogenized element radius or enrichment")
        pitches = self.ring_pitches()
        scales = torch.cat((pitches.new_zeros(1), pitches.cumsum(0)
                            / torch.arange(1, len(pitches) + 1, dtype=pitches.dtype,
                                           device=pitches.device)))
        centers_mm = self.lattice_xy * scales[self.ring, None]
        interfaces_mm = []
        for i, values in enumerate(self.reference_interfaces):
            radii = self.pitch_mm.new_tensor(values)
            if bool(self.fuelled[i]):
                radii = radii + self.element_radius_mm[self.ring[i]] - radii[0]
            interfaces_mm.append(radii)
        fuel_ring = self.ring[self.fuelled].repeat_interleave(self.axial_count)
        # Assembly zero is at the top; physical z increases bottom-to-top.
        bounds_cm = torch.stack((self.z_edges_mm[:-1], self.z_edges_mm[1:]), dim=1).flip(0) / 10
        bounds_cm = bounds_cm.repeat(int(self.fuelled.sum()), 1)
        radius_cm = self.element_radius_mm[fuel_ring] / 10
        return {
            "channel_xy_mm": centers_mm,
            "channel_interfaces_mm": tuple(interfaces_mm),
            "assembly_xy_cm": centers_mm[self.fuelled].repeat_interleave(self.axial_count, 0) / 10,
            "assembly_radius_cm": radius_cm,
            "assembly_z_bounds_cm": bounds_cm,
            "assembly_volume_cm3": math.pi * radius_cm.square() * (bounds_cm[:, 1] - bounds_cm[:, 0]),
            "fresh_enrichment_mass_fraction": self.enrichment[fuel_ring],
        }


class Design(ChannelDesign):
    """Select the bare-channel, graphite-pin or graphite-sleeve design variables."""
    def __init__(self, definition, spec):
        """Select and initialize the variables declared by one WR-1 scenario.

        Parameters
        ----------
        definition : dict
            Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
        spec : dict
            Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.

        Raises
        ------
        ValueError
            The declared parameter names or shapes do not match the selected bare-channel,
            graphite-pin or graphite-sleeve model.
        """
        super().__init__(definition, spec)
        self.register_buffer("fixed_outer_pitch_mm", torch.tensor(spec["fixed_outer_pitch_mm"], dtype=torch.float64))
        expected = {"pitch_mm": (3,), "element_radius_mm": (4,), "enrichment": (4,), "cycle_days": (1,)}
        if "graphite" in spec and spec["graphite"].get("mode") != "assembly_sleeves":
            from .graphite import initialize_graphite
            expected.update(graphite_pitch_mm=(1,), graphite_radius_mm=(4,))
            initialize_graphite(self, spec["graphite"])
        elif "graphite" in spec:
            from .sleeves import initialize_sleeves
            expected.update(graphite_thickness_mm=(21,))
            initialize_sleeves(self, spec["graphite"])
        if {name: tuple(p.shape) for name, p in self.named_parameters()} != expected:
            raise ValueError("WR-1 design parameters do not match the selected optimization step")

    def forward(self):
        """Extend channel geometry with the selected live graphite geometry.

        Returns
        -------
        dict
            Channel and assembly geometry, plus graphite cylinder or sleeve tensors when
            configured. Length-field suffixes specify mm or cm.
        """
        mapped = super().forward()
        if hasattr(self, "graphite_position_index"):
            from .sleeves import sleeve_mapping
            mapped.update(sleeve_mapping(self, mapped))
        elif hasattr(self, "graphite_spec"):
            from .graphite import graphite_mapping
            mapped.update(graphite_mapping(self))
        return mapped

    def ring_pitches(self):
        """Append the fixed passive-ring spacing to the live fuel-ring pitches.

        Returns
        -------
        torch.Tensor
            Four radial spacing increments in mm, including the fixed outer ring.
        """
        # Passive outer sites retain their reference spacing beyond the fuelled rings.
        return torch.cat((self.pitch_mm, self.fixed_outer_pitch_mm))


def make_case(definition, scenario, *, design_values=None, boron=None, sampling=None, fuel_states=None):
    """Build a native WR-1 case from resolved model and scenario inputs.

    Parameters
    ----------
    definition : dict
        Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
    scenario : dict
        Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.
    design_values : dict or None, optional
        Replacement design values; None uses the scenario's initial design.
    boron : float or None, optional
        B-10 concentration in mg per kg of unborated moderator; None uses the scenario
        value.
    sampling : dict or None, optional
        OpenMC particle, batch, inactive-batch, seed and thread settings; None uses the
        scenario settings.
    fuel_states : dict or None, optional
        Nuclide atom densities in atoms/(barn cm), keyed by representative assembly
        identity. None creates fresh fuel.

    Returns
    -------
    dict
        Native materials, regions, physical assembly mapping, tally settings, design values
        and geometry checks for the selected symmetry domain.

    Raises
    ------
    ValueError
        Geometry overlaps, violates symmetry or boundaries, or has invalid boron or design
        values.
    ArithmeticError
        The physical assembly-to-region map is incomplete.

    Notes
    -----
    The definition dictionary originates in model.toml and the scenario dictionary in
    study.toml. The case adapter resolves both; this function performs no configuration-file
    reads.
    """
    import openmc.data
    from diffusion_surrogate.symmetry import audit_c6_orbits
    from optimization.design import set_parameters

    spec = definition["transport"]
    aliases = spec["nuclide_aliases"]
    recipe = fuel_recipe(definition)
    design = Design(definition, scenario)
    if design_values is not None:
        set_parameters(design, design_values)
    mapped = design()
    centers = mapped["channel_xy_mm"].detach().numpy() / 10
    radii = [value.detach().numpy() / 10 for value in mapped["channel_interfaces_mm"]]
    fuelled, ring = design.fuelled.numpy(), design.ring.numpy()
    labels = [(bool(f), tuple(r), float(design.enrichment[ring[i]].detach()) if f else None)
              for i, (f, r) in enumerate(zip(fuelled, radii))]
    orbits = audit_c6_orbits(centers, labels, tolerance_cm=1e-9)
    identities = {i: min(design.channel_ids[j] for j in orbit) for orbit in orbits for i in orbit}
    domain = HexagonalDomain(scenario["apothem_cm"])
    clearance = cylinder_clearances(domain, centers, [r[-1] for r in radii])
    if min(clearance.values()) <= 0:
        raise ValueError("Channel interfaces overlap or cross the outer moderator boundary")
    if scenario["outer_radial_boundary"] not in ("vacuum", "reflective"):
        raise ValueError("The outer radial boundary must be vacuum or reflective")
    z = design.z_edges_mm.numpy() / 10
    boron = scenario["initial_boron_mg_B10_per_kg"] if boron is None else float(boron)
    if boron < 0 or not math.isfinite(boron):
        raise ValueError("Boron concentration must be finite and nonnegative")

    def material(name):
        """Resolve one material composition and apply moderator boron.

        Parameters
        ----------
        name : str
            Material key in the resolved physical model.

        Returns
        -------
        dict
            Transport-compatible atom densities in atoms/(barn cm) and temperature in K.
        """
        entry = definition["materials"][name]
        atoms = dict(entry["atom_density"])
        if name == "moderator" and boron:
            n10 = boron * 1e-6 * entry["density_g_cm3"] / openmc.data.atomic_mass("B10") * openmc.data.AVOGADRO / 1e24
            ratio = spec["boron"]["natural_atom_fractions"]
            atoms.update(B10=n10, B11=n10 * ratio["B11"] / ratio["B10"])
        return {"atom_density": transport_atoms(atoms, aliases), "temperature_K": entry["temperature_K"]}

    def shell_material(site, shell, level, volume):
        """Assign fresh or depleted material to one channel shell.

        Parameters
        ----------
        site : int
            Physical channel index in the lattice definition.
        shell : int
            Radial shell index; zero denotes the homogenized fuel element in a fuelled channel.
        level : int
            Axial assembly index, counted from the top.
        volume : float
            Volume of one physical shell segment in cm³.

        Returns
        -------
        dict
            Material composition, temperature and depletion metadata, including fresh
            heavy-metal mass for fuel.
        """
        if fuelled[site] and shell == 0:
            fresh = recipe.atom_densities(float(design.enrichment[ring[site]].detach()))
            inert = definition["fuel"]["inert_atom_density"]
            fuel = fresh if fuel_states is None else fuel_states[f"{identities[site]}/axial/{level}"]
            atoms = dict(inert)
            for nuclide, number in fuel.items():
                atoms[nuclide] = atoms.get(nuclide, 0.) + number
            hm_density = sum(fresh[n] * openmc.data.atomic_mass(n) for n in ("U235", "U238")) * 1e24 / openmc.data.AVOGADRO / 1000
            return {"atom_density": transport_atoms(atoms, aliases), "temperature_K": spec["homogenized_element_temperature_K"],
                    "depletable": True, "fresh_fuel_atom_density": fresh, "inert_atom_density": inert,
                    "initial_hm_kg_per_assembly": hm_density * volume}
        binding = design.channel_materials[site][shell]
        return {**material(binding), "depletable": False,
                **({"thermal_scattering": spec["moderator_thermal_scattering"]} if binding == "moderator" else {})}

    background = {**material("moderator"), "label": "moderator", "kind": "wr1_background",
                  "depletable": False, "thermal_scattering": spec["moderator_thermal_scattering"], "void": False}
    case = channel_regions(domain, design.channel_ids, centers, radii, z, orbits, shell_material, background,
                           site_metadata=[{"ring": int(value)} for value in ring], region_kind="wr1_shell")
    if len(case["physical_assembly_to_region"]) != int(fuelled.sum()) * design.axial_count:
        raise ArithmeticError("The WR-1 assembly identity map is incomplete")
    regions = case["regions"]
    lower, upper = domain.bounds(z)
    nx, ny = spec["comparison_mesh_subdivisions"]
    case.update({
        "power_W": scenario["power_W"], "symmetry_multiplier": domain.multiplier,
        "boron_mg_B10_per_kg": boron, "nuclides": sorted({name for entry in regions for name in entry["atom_density"]}),
        "design": {name: value.detach().tolist() for name, value in design.named_parameters()},
        "sampling": dict(scenario["sampling"] if sampling is None else sampling),
        "statistics_block_size": spec["statistics_block_size"], "group_edges_eV": spec["group_edges_eV"],
        "comparison_edges_cm": [np.linspace(lower[0], upper[0], nx + 1).tolist(),
                                np.linspace(lower[1], upper[1], ny + 1).tolist(), z.tolist()],
        "cross_sections": str(NUCLEAR_DATA / "wr_1/cross_sections.xml"),
        "temperature_label_tolerance_K": spec["temperature_label_tolerance_K"],
        "full_observations": True, "observation_cross_sections": "macroscopic_tallies",
        "need_depletion_rates": False, "depletion_rate_estimator": "native_microscopic_tracklength_for_all_chain_parents_with_transport_data",
        "depletion_nuclides": [], "depletion_reactions": [],
        "material_response": {"region_indices": [i for i, r in enumerate(regions) if r["depletable"]],
                              "nuclides": spec["material_response_nuclides"]},
        "boron_response": {"region_indices": [i for i, r in enumerate(regions) if r.get("thermal_scattering")],
                           "nuclides": list(spec["boron"]["natural_atom_fractions"])},
        "effective_approximations": {name: spec[name] for name in ("temperature_basis", "carbon_transport", "oxygen_transport", "organic_thermal_scattering")},
    })
    case["geometry"].update(symmetry_boundary_type=scenario["symmetry_boundary_type"],
                            boundaries=[scenario["outer_radial_boundary"], "reflective", "reflective", "reflective",
                                        *scenario["axial_boundaries"]])
    if "graphite" in scenario:
        if scenario["graphite"].get("mode") == "assembly_sleeves":
            from .sleeves import add_sleeves as add_graphite
        else:
            from .graphite import add_graphite
        case = add_graphite(case, design, mapped, NUCLEAR_DATA / "graphite/cross_sections.xml")
    return case


def fractions(grid, design, case):
    """Map live WR-1 cylinders into the shared diffusion mesh.

    Parameters
    ----------
    grid : C6Grid
        Fixed triangular-prism diffusion mesh with coordinates in cm.
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    case : dict
        Native region, material and symmetry description returned by ``make_case``.

    Returns
    -------
    torch.Tensor
        Cell-by-region volume fractions, retaining derivatives with respect to design
        geometry.
    """
    mapped = design()
    cylinders = {}
    z = design.z_edges_mm.detach().numpy() / 10
    for i, site in enumerate(design.channel_ids):
        radii = tuple(mapped["channel_interfaces_mm"][i] / 10)
        if "sleeve_outer_radius_mm" in mapped and bool(design.fuelled[i]):
            by_level = mapped["sleeve_outer_radius_mm"][i] / 10
            radii += ({(float(z[-2-level]), float(z[-1-level])): by_level[level]
                       for level in range(design.axial_count)},)
        cylinders[site] = Cylinder(mapped["channel_xy_mm"][i] / 10, radii)
    if "graphite_xy_mm" in mapped:
        cylinders.update({site: Cylinder(xy / 10, radius.reshape(1) / 10)
                          for site, xy, radius in zip(design.graphite_ids, mapped["graphite_xy_mm"], mapped["graphite_radius_mm"])})
    return cylinder_fractions(grid, case["regions"], cylinders)


def mesh(design, case, *, subdivisions=24, axial_subdivisions=2):
    """Construct the WR-1 mesh and its live material-fraction callback.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    case : dict
        Native region, material and symmetry description returned by ``make_case``.
    subdivisions : int, optional
        Number of radial triangular subdivisions in the C6 diffusion mesh.
    axial_subdivisions : int, optional
        Number of diffusion cells within each axial assembly interval.

    Returns
    -------
    DiffusionGeometry
        Native physical regions on a C6Grid, with live fractions and an explicit identity
        for reuse across native states with the same design.
    """
    z = design.z_edges_mm.detach().numpy() / 10
    z = np.concatenate([np.linspace(lo, hi, axial_subdivisions + 1)[:-1]
                        for lo, hi in zip(z[:-1], z[1:])] + [z[-1:]])
    grid = C6Grid(case["geometry"]["apothem_cm"], subdivisions, z,
                  symmetry_boundary=case["geometry"].get("symmetry_boundary_type", "periodic"),
                  radial_boundary=case["geometry"]["boundaries"][0],
                  axial_boundaries=case["geometry"]["boundaries"][4:])
    labels = tuple(region["label"] for region in case["regions"])
    identity = (grid.shape, tuple(grid.z_edges.tolist()), case["geometry"]["apothem_cm"],
                case["geometry"].get("symmetry_boundary_type", "periodic"),
                tuple(grid.boundaries),
                tuple((region["label"], tuple(region["z_bounds_cm"])) for region in case["regions"]))
    return DiffusionGeometry(grid, lambda: fractions(grid, design, case), labels, identity)
