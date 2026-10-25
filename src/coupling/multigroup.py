"""Native OpenMC transport of shared CAD with supplied multigroup materials."""

from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

import numpy as np
import openmc
import torch

from geometry import CadGeometry
from geometry.openmc import from_cad


def write_library(path, energy_edges_eV, macro, names):
    """Write isotropic, nonmultiplying-scatter material tables for OpenMC.

    Parameters
    ----------
    path : pathlib.Path
        Destination HDF5 file.
    energy_edges_eV : array_like
        Increasing energy boundaries in eV. Material axes run fast to thermal.
    macro : dict of torch.Tensor
        Region/group macroscopic tables in cm⁻¹, with kappa_fission in eV/cm.
        Scattering axes are incoming then outgoing group.
    names : sequence of str
        OpenMC macroscopic names in the same order as the material tables.

    Raises
    ------
    NotImplementedError
        Supplied scattering requires multiplication or angular moments.
    """
    if not torch.equal(macro["scatter"], macro["nu_scatter"]):
        raise NotImplementedError("The multigroup writer requires nonmultiplying scattering")
    if "scatter_p1" in macro and bool(macro["scatter_p1"].detach().any()):
        raise NotImplementedError("The multigroup writer requires isotropic scattering")
    arrays = {key: value.detach().cpu().numpy() for key, value in macro.items()}
    groups = openmc.mgxs.EnergyGroups(energy_edges_eV)
    library = openmc.MGXSLibrary(groups)
    for index, name in enumerate(names):
        data = openmc.XSdata(name, groups)
        data.order = 0
        data.set_total(arrays["absorption"][index] + arrays["scatter"][index].sum(-1))
        data.set_absorption(arrays["absorption"][index])
        data.set_scatter_matrix(arrays["scatter"][index][..., None])
        for key in ("fission", "nu_fission", "kappa_fission", "chi"):
            getattr(data, "set_" + key)(arrays[key][index])
        library.add_xsdata(data)
    library.export_to_hdf5(path)


class MultigroupTransport:
    """Reuse one converted CAD geometry across composition-dependent native states."""

    def __init__(self, cad, directory, energy_edges_eV, sampling, *, power_W):
        """Prepare native material identities, geometry and regional flux tallies.

        Parameters
        ----------
        cad : geometry.CadGeometry
            Shared physical geometry and boundary assignments.
        directory : pathlib.Path
            New directory for this geometry's STEP and successive native states.
        energy_edges_eV : array_like
            Increasing energy boundaries in eV.
        sampling : dict
            OpenMC particles, batches, inactive generations, seed and threads.
        power_W : float
            Prescribed total fission power in W.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.energy_edges = energy_edges_eV
        self.sampling, self.power_W = sampling, power_W
        self.records = []
        step = cad.write_step(self.directory / "geometry.step")
        fixed = CadGeometry.read_step(step)
        self.volumes = torch.tensor([volume.shape.volume / 1000 for volume in fixed.volumes],
                                    dtype=torch.float64)
        self.names = tuple(f"region_{i}" for i in range(len(fixed.volumes)))
        openmc.reset_auto_ids()
        materials = []
        for volume, name in zip(fixed.volumes, self.names):
            material = openmc.Material(name=volume.label)
            material.set_density("macro", 1.)
            material.add_macroscopic(name)
            materials.append(material)
        self.adapted = from_cad(fixed, {volume.label: material
                                      for volume, material in zip(fixed.volumes, materials)})
        self.model = openmc.Model(self.adapted.geometry, openmc.Materials(materials))
        self.model.settings.energy_mode = "multi-group"
        for key in ("particles", "batches", "inactive"):
            setattr(self.model.settings, key, sampling[key])
        self.model.settings.verbosity = 5
        cells = openmc.CellFilter(self.adapted.cells)
        flux = openmc.Tally(name="regional_flux")
        flux.filters = [cells, openmc.EnergyFilter(energy_edges_eV)]
        flux.scores = ["flux"]
        heating = openmc.Tally(name="regional_heating")
        heating.filters = [cells]
        heating.scores = ["kappa-fission"]
        self.model.tallies = openmc.Tallies([flux, heating])

    def __call__(self, macro):
        """Run one native state and normalize regional responses to physical power.

        Parameters
        ----------
        macro : dict of torch.Tensor
            Current regional macroscopic tables, ordered as the shared CAD volumes.

        Returns
        -------
        dict of torch.Tensor
            Native keff and its reported standard error, integrated regional group
            flux per source, source rate in s⁻¹, powers in W and CAD volumes in cm³.
        """
        started = time.perf_counter()
        directory = self.directory / f"state_{len(self.records):02d}"
        directory.mkdir()
        library = directory / "mgxs.h5"
        write_library(library, self.energy_edges, macro, self.names)
        self.model.materials.cross_sections = str(library)
        self.model.settings.seed = self.sampling["seed"] + len(self.records)
        sources = []
        for index, boxes in enumerate(self.adapted.source_bounds_cm):
            if bool(macro["nu_fission"][index].detach().sum() > 0):
                sources.extend(openmc.IndependentSource(space=openmc.stats.Box(lower, upper),
                               constraints={"fissionable": True}) for lower, upper in boxes)
        self.model.settings.source = sources
        with (directory / "openmc.log").open("w") as log, redirect_stdout(log):
            statepoint = self.model.run(cwd=directory, threads=self.sampling["threads"],
                geometry_debug=True, openmc_exec=str(Path(sys.prefix) / "bin/openmc"))
        with openmc.StatePoint(statepoint) as state:
            # EnergyFilter bins ascend; the material/solver group axis descends.
            flux = state.get_tally(name="regional_flux").mean.reshape(len(self.names), -1)[:, ::-1].copy()
            heating = state.get_tally(name="regional_heating").mean.reshape(-1)
            mean, error = float(state.keff.nominal_value), float(state.keff.std_dev)
        normalization = self.power_W / (heating.sum() * 1.602176634e-19)
        self.records.append({"directory": str(directory), "keff": mean,
                             "standard_error": error, "wall_s": time.perf_counter() - started})
        return {"keff": torch.tensor(mean, dtype=torch.float64),
                "standard_error": torch.tensor(error, dtype=torch.float64),
                "region_flux_per_source": torch.from_numpy(flux),
                "source_rate_per_s": torch.tensor(normalization, dtype=torch.float64),
                "power_W": torch.from_numpy(np.asarray(heating / heating.sum() * self.power_W)),
                "volume_cm3": self.volumes}
