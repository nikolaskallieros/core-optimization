"""Full-chain inventories and native OpenMC rate collapse with explicit case inputs."""
import ctypes as ct
from pathlib import Path

import numpy as np
import openmc.data
import openmc.deplete
from runtime.io import sha
from materials.composition import map_nuclides
from diffusion_surrogate.native3d import FLUX_RATES, collapsed_depletion


def collapse_rates(dll, case, spectrum=None):
    """Collapse native reaction rates at the actual fuel temperatures.

    Parameters
    ----------
    dll : ctypes.CDLL
        Initialized OpenMC shared library containing the active case.
    case : mapping
        Transport regions, depletion nuclide/reaction order, energy bins and optional kernel
        cache.
    spectrum : mapping or None, optional
        Optional energy_eV boundaries and region-by-bin flux; None obtains the case's native
        flux tally.

    Returns
    -------
    dict
        Native histogram flux and collapsed reaction rates, with group-resolved rates when
        requested.

    Notes
    -----
    Loading without temperatures reads every available table. Supplying the actual
    temperatures limits memory while retaining native interpolation. Already loaded cold
    parents obtain hot tables from settings.temperature.range.
    """
    if case.get("depletion_kernel_cache"):
        from diffusion_surrogate.rate_cache import collapse_cached
        return collapse_cached(dll, case, spectrum)
    values = np.ascontiguousarray(sorted({r["temperature_K"] for r in case["regions"] if r["depletable"]}))
    pointer = ct.POINTER(ct.c_double)
    dll.openmc_load_nuclide.argtypes = [ct.c_char_p, pointer, ct.c_int]
    dll.openmc_load_nuclide.restype = ct.c_int
    for name in case["depletion_nuclides"]:
        code = dll.openmc_load_nuclide(name.encode(), values.ctypes.data_as(pointer), len(values))
        if code:
            raise RuntimeError(f"Cannot load depletion parent {name} at {values.tolist()} K ({code})")
    result = collapsed_depletion(dll, case, spectrum=spectrum)
    if case.get("need_depletion_response"):
        edges = result["energy_eV"]
        groups = []
        collapse_case = dict(case, depletion_flux_refinement_check=False)
        for low, high in zip(case["group_edges_eV"][:-1], case["group_edges_eV"][1:]):
            fraction = np.maximum(0., np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)) / np.diff(edges)
            partial = {"energy_eV": edges, "flux": result["flux"] * fraction[None, :]}
            groups.append(collapsed_depletion(dll, collapse_case, spectrum=partial)["rates"])
        result["rates_by_group"] = np.stack(groups, axis=-1)
        if not np.allclose(result["rates_by_group"].sum(-1), result["rates"], rtol=1e-12, atol=0):
            raise ArithmeticError("Group-resolved depletion rates do not preserve the native histogram total")
    return result

class IsotopicInventory:
    """A chain/library binding; geometry and live fresh recipes come from the caller."""
    def __init__(self, *, case_factory, recipe, settings, chain_path, cross_sections,
                 design_values=None, aliases=None, chain_sha256=None, kernel_cache=None,
                 physical_inventory_response=True,
                 native_driver="coupling.openmc:run"):
        """Bind a full depletion chain to explicit case and recipe interfaces.

        Parameters
        ----------
        case_factory : callable
            Build a transport case from design_values, fuel_states, boron and sampling.
        recipe : callable
            Construct live atom densities, volumes and initial heavy-metal masses from this
            binding and a differentiable design.
        settings : mapping
            Transport density cutoff, sampling and depletion-collapse controls.
        chain_path : str or pathlib.Path
            OpenMC depletion-chain XML file.
        cross_sections : str or pathlib.Path
            OpenMC continuous-energy library XML with resolvable neutron-data paths.
        design_values : mapping or None, optional
            Initial design values passed to the case factory.
        aliases : mapping or None, optional
            Explicit isotope aliases for the transport/depletion convention.
        chain_sha256 : str or None, optional
            Expected depletion-chain file digest; None omits this optional identity check.
        kernel_cache : str, pathlib.Path or None, optional
            Optional precomputed native depletion-rate cache.
        physical_inventory_response : bool, optional
            Include all transported fuel and fixed inert constituents in requested material
            responses.
        native_driver : str, optional
            Importable module:function native transport driver.

        Raises
        ------
        ValueError
            The chain file does not match its supplied source digest.
        """
        self.case_factory, self.recipe = case_factory, recipe
        self.spec = dict(settings)
        self.aliases = {} if aliases is None else dict(aliases)
        self.cross_sections = Path(cross_sections)
        self.kernel_cache = kernel_cache
        self.physical_inventory_response = physical_inventory_response
        self.native_driver = native_driver
        if chain_sha256 is not None and sha(chain_path) != chain_sha256:
            raise ValueError("The depletion chain differs from its declared source hash")
        self.chain = openmc.deplete.Chain.from_xml(chain_path)
        self.names = tuple(n.name for n in self.chain.nuclides)
        self.name_index = {name: i for i, name in enumerate(self.names)}
        self.reactions = tuple(self.chain.reactions)
        self.library = openmc.data.DataLibrary.from_xml(self.cross_sections)
        self.available = {name for item in self.library if item["type"] == "neutron" for name in item["materials"]}
        self.data_paths = {name: Path(item["path"]) for item in self.library if item["type"] == "neutron" for name in item["materials"]}
        self.base_case = case_factory(design_values=design_values)
        self.design_values = self.base_case["design"]
        self.regions = [region for region in self.base_case["regions"] if region["depletable"]]
        self.labels = [region["label"] for region in self.regions]
        self.volumes = np.array([region["volume_cm3"] for region in self.regions])
        self.multiplicity = np.array([region["physical_multiplicity"] for region in self.regions])
        self.symmetry_multiplier = self.base_case.get("symmetry_multiplier", 1)
        self.hm_kg = (np.array([region["initial_hm_kg_per_assembly"] for region in self.regions])
                      * self.multiplicity / self.symmetry_multiplier)
        self.fresh = np.zeros((len(self.regions), len(self.names)))
        for row, region in enumerate(self.regions):
            for name, value in map_nuclides(region["fresh_fuel_atom_density"], self.aliases).items():
                self.fresh[row, self.name_index[name]] = value * 1e24

    def live_recipe(self, design):
        """Evaluate the caller's fresh-fuel recipe on a live design.

        Parameters
        ----------
        design : torch.nn.Module
            Differentiable design exposing the geometry and composition parameters expected by
            the recipe.

        Returns
        -------
        tuple of torch.Tensor
            Fresh full-chain densities in atoms/cm³, fuel volumes in cm³ and initial heavy-metal
            masses in kg, in region order.
        """
        return self.recipe(self, design)

    def case(self, inventory, boron, *, sampling=None, refinement=None, derivatives=False):
        """Map a full-chain inventory into an explicitly ordered CE case.

        Parameters
        ----------
        inventory : array_like
            Region-by-chain atom densities in atoms/cm³.
        boron : float
            Soluble boron concentration in ppm under the case's concentration convention.
        sampling : mapping or None, optional
            Native sampling overrides; None uses the configured depletion sampling.
        refinement : bool or None, optional
            Override the native flux-bin refinement check; None uses the configured value.
        derivatives : bool, optional
            Request material responses and group-resolved depletion rates.

        Returns
        -------
        dict
            Transport case with declared parent/reaction order, density-cutoff accounting and
            optional response/cache requests.

        Raises
        ------
        ValueError
            Inventory dimensions or values are invalid, or the factory changes region ordering.
        """
        inventory = np.asarray(inventory)
        if inventory.shape != self.fresh.shape or not np.isfinite(inventory).all() or np.min(inventory) < 0:
            raise ValueError("Fuel inventory shape, values or ordering are invalid")
        states, cutoff_omitted, no_data = {}, [], []
        cutoff = self.spec["transport_density_cutoff_atom_b_cm"]
        for row, label in enumerate(self.labels):
            numbers = inventory[row] * 1e-24
            # Keep the complete chain inventory separately. The declared C/O
            # transport convention is applied only when preparing the CE case.
            mapped = map_nuclides(dict(zip(self.names, numbers)), self.aliases)
            states[label] = {n: v for n, v in mapped.items() if n in self.available and v > cutoff}
            cutoff_omitted.append(sum(v for n, v in mapped.items() if n in self.available and v <= cutoff))
            no_data.append(sum(v for n, v in mapped.items() if n not in self.available))
        case = self.case_factory(design_values=self.design_values, fuel_states=states, boron=boron,
                                 sampling=self.spec["sampling"] if sampling is None else sampling)
        if [r["label"] for r in case["regions"] if r["depletable"]] != self.labels:
            raise ValueError("Native transport reordered assembly inventories")
        parents = sorted(self.available.intersection(self.names))
        present = {n for values in states.values() for n, v in values.items() if v > 0}
        direct = sorted(present.intersection(self.spec["direct_nuclides"]).intersection(parents))
        case.update(cross_sections=str(self.cross_sections), need_depletion_rates=True,
                    depletion_rate_estimator=FLUX_RATES, depletion_nuclides=parents,
                    depletion_reactions=self.reactions,
                    depletion_reaction_mts=[openmc.data.REACTION_MT[n] for n in self.reactions],
                    depletion_flux_groups=self.spec["depletion_flux_groups"],
                    depletion_flux_refinement_check=self.spec["depletion_flux_refinement_check"] if refinement is None else refinement,
                    depletion_direct_nuclides=direct, depletion_direct_reactions=self.spec["direct_reactions"],
                    transport_cutoff_omitted_atom_b_cm=cutoff_omitted,
                    transport_unevaluated_inventory_atom_b_cm=no_data,
                    depletion_scope=self.spec["scope"])
        if derivatives:
            case["need_depletion_response"] = True
            case["material_response"]["nuclides"] = sorted(present)
            case["material_response"]["basis"] = "All transported fuel nuclides, including its fission and activation products."
        if self.kernel_cache is not None:
            case["depletion_kernel_cache"] = str(self.kernel_cache)
        case["physical_inventory_response"] = self.physical_inventory_response
        if derivatives and self.physical_inventory_response:
            case["material_response"]["nuclides"] = sorted({
                n for region in case["regions"] if region["depletable"] for n in region["atom_density"]})
            case["material_response"]["basis"] = "All transported fuel nuclides and fixed inert constituents."
        return case
