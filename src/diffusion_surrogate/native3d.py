"""Original continuous-energy OpenMC transport for explicit 3D material regions.

This adapter does not differentiate Monte Carlo histories. It supplies the current
design's measured fields, microscopic data and physical depletion rates.
"""
import ctypes as ct
from pathlib import Path
import time

import numpy as np
import openmc

from .reference import statistics

DIRECT_RATES = 'native_microscopic_tracklength_for_all_chain_parents_with_transport_data'
FLUX_RATES = 'native_flux_collapse_with_direct_capture_and_fission'


def depletion_edges(case):
    """Build the zero-starting energy histogram for depletion-rate collapse.

    Parameters
    ----------
    case : dict
        Native model description with ordered regions, sampling, energy groups and
        depletion-tally settings.

    Returns
    -------
    numpy.ndarray
        Energy-bin edges in eV, followed by a logarithmic grid up to the transport upper
        energy bound.
    """
    return np.concatenate(([0.], np.geomspace(1e-5, case['group_edges_eV'][-1], case['depletion_flux_groups'])))


def collapsed_depletion(dll, case, spectrum=None):
    """Collapse depletion reactions through OpenMC's native nuclide API.

    Parameters
    ----------
    dll : ctypes.CDLL
        Initialized native OpenMC shared library with the current model and tally data
        loaded.
    case : dict
        Native model description with ordered regions, sampling, energy groups and
        depletion-tally settings.
    spectrum : dict or None, optional
        Saved energy_eV edges and fuel-region-by-bin flux scores; None reads the current
        native flux tally.

    Returns
    -------
    dict
        Microscopic flux-collapse scores by fuel region, parent and reaction, the
        source-normalized flux histogram, its eV edges and optional coarsened-grid checks.

    Raises
    ------
    ValueError
        A supplied spectrum does not match the declared region and energy grid.
    RuntimeError
        A native nuclide loading or collapse operation fails.

    Notes
    -----
    Transport is continuous-energy. Only the depletion-rate quadrature uses the fine energy
    histogram. Caller-selected capture/fission rates remain direct.
    """
    pointer = ct.POINTER(ct.c_double)
    signatures = {'openmc_load_nuclide': [ct.c_char_p, pointer, ct.c_int],
                  'openmc_get_nuclide_index': [ct.c_char_p, ct.POINTER(ct.c_int)],
                  'openmc_nuclide_collapse_rate': [ct.c_int, ct.c_int, ct.c_double,
                                                   pointer, pointer, ct.c_int, pointer]}
    for name, arguments in signatures.items():
        getattr(dll, name).argtypes, getattr(dll, name).restype = arguments, ct.c_int

    def call(name, *args):
        """Invoke a configured native function and check its return code.

        Parameters
        ----------
        name : str
            OpenMC function symbol.
        *args : tuple
            ctypes arguments for the selected function.

        Raises
        ------
        RuntimeError
            The native function returns a nonzero error code.
        """
        code = getattr(dll, name)(*args)
        if code:
            raise RuntimeError(f'{name} failed ({code})')

    energy = depletion_edges(case)
    regions = [entry for entry in case['regions'] if entry.get('depletable', False)]
    if spectrum is None:
        index, data, shape = ct.c_int32(), pointer(), (ct.c_size_t*3)()
        call('openmc_get_tally_index', 12, ct.byref(index))
        call('openmc_tally_results', index, ct.byref(data), ct.byref(shape))
        active = case['sampling']['batches']-case['sampling']['inactive']
        flux = (np.ctypeslib.as_array(data, tuple(shape))[:, 0, 1]/active).reshape(len(regions), -1).copy()
    else:
        energy, flux = np.ascontiguousarray(spectrum['energy_eV']), np.ascontiguousarray(spectrum['flux'])
        if flux.shape != (len(regions), len(energy)-1) or not np.array_equal(energy, depletion_edges(case)):
            raise ValueError('Saved flux spectrum does not match the specified collapse grid')
    grids = {'rates': (energy, flux)}
    if case.get('depletion_flux_refinement_check', False):
        for stride in (2, 4):
            grids[f'rates_stride_{stride}'] = (np.ascontiguousarray(energy[::stride]), flux.reshape(len(regions), -1, stride).sum(-1))
    answers = {name: np.zeros((len(regions), len(case['depletion_nuclides']), len(case['depletion_reactions'])))
               for name in grids}
    for parent, name in enumerate(case['depletion_nuclides']):
        call('openmc_load_nuclide', name.encode(), None, 0)
        index = ct.c_int()
        call('openmc_get_nuclide_index', name.encode(), ct.byref(index))
        for key, (edges, spectrum) in grids.items():
            for zone, entry in enumerate(regions):
                for reaction, mt in enumerate(case['depletion_reaction_mts']):
                    value = ct.c_double()
                    call('openmc_nuclide_collapse_rate', index, mt, entry['temperature_K'],
                         edges.ctypes.data_as(pointer), spectrum[zone].ctypes.data_as(pointer), len(edges)-1, ct.byref(value))
                    answers[key][zone, parent, reaction] = value.value
    return {**answers, 'flux': flux, 'energy_eV': energy}


def material_list(case):
    """Construct ordered OpenMC materials from the native case definition.

    Parameters
    ----------
    case : dict
        Native model description with ordered regions, sampling, energy groups and
        depletion-tally settings.

    Returns
    -------
    list of openmc.Material
        Materials with atom densities in atoms/(barn cm), temperatures in K, volumes in cm³
        and declared thermal-scattering/depletion settings.
    """
    materials = []
    for i, entry in enumerate(case['regions']):
        material = openmc.Material(i+1, name=entry['label'])
        material.set_density('sum')
        for name, density in entry['atom_density'].items():
            if density > 0:
                material.add_nuclide(name, density)
        for name in entry.get('thermal_scattering', []):
            material.add_s_alpha_beta(name)
        material.temperature = entry['temperature_K']
        material.volume = entry['volume_cm3']
        material.depletable = entry.get('depletable', False)
        materials.append(material)
    return materials


def tally_cross_sections(state, tally_ids, flux, scattering_flux, nuclides, *, atom_density=None):
    """Decode reaction and scattering tallies with matching flux estimators.

    Parameters
    ----------
    state : openmc.StatePoint
        Completed statepoint containing the requested tallies.
    tally_ids : tuple of int
        Reaction, scattering-transfer and first-moment tally IDs, in that order.
    flux : numpy.ndarray
        Region-by-group tracklength flux scores used to divide reaction tallies.
    scattering_flux : numpy.ndarray
        Region-by-group flux scores matching the scattering estimator.
    nuclides : int
        Number of nuclide bins in each tally; one for macroscopic scoring.
    atom_density : numpy.ndarray or None, optional
        Region-by-nuclide densities in atoms/(barn cm), required to remove analog isotope
        density weighting.

    Returns
    -------
    dict of str to numpy.ndarray
        Cross sections with region/nuclide/incoming-group axes and an outgoing-group axis
        for scattering. Microscopic data use barns, macroscopic data cm⁻¹; kappa_fission
        additionally carries eV.

    Notes
    -----
    Reaction scores use tracklength flux; scattering uses the caller's matching estimator.
    Analog isotope scores already include density, so supply atom_density to recover
    microscopic scattering in barns. Absent isotopes receive zero sampled scattering,
    without inventing an extrapolation.
    """
    regions, groups = flux.shape
    reaction_id, scatter_id, moment_id = tally_ids
    reaction = state.get_tally(id=reaction_id).mean.reshape(regions, groups, nuclides, 5) / flux[:, :, None, None]
    scattering = (state.get_tally(id=scatter_id).mean.reshape(regions, groups, groups, nuclides, 2)
                  / scattering_flux[:, :, None, None, None])
    moment = (state.get_tally(id=moment_id).mean.reshape(regions, groups, 2, nuclides)[:, :, 1, :]
              / scattering_flux[:, :, None])
    tables = {name: reaction[..., i].transpose(0, 2, 1)
              for i, name in enumerate(('total_tracklength', 'absorption', 'fission', 'nu_fission', 'kappa_fission'))}
    tables.update(scatter=scattering[..., 0].transpose(0, 3, 1, 2),
                  nu_scatter=scattering[..., 1].transpose(0, 3, 1, 2), scatter_p1=moment.transpose(0, 2, 1))
    if atom_density is not None:
        for name in ('scatter', 'nu_scatter', 'scatter_p1'):
            value = tables[name]
            atoms = atom_density.reshape(regions, nuclides, *([1] * (value.ndim - 2)))
            tables[name] = np.divide(value, atoms, out=np.zeros_like(value), where=atoms > 0)
    return tables


def extract(case, directory, batch_scores, *, materials=None, started=None, collapsed=None,
            factorized_statistics=False):
    """Extract completed native tallies and physical normalization records.

    Parameters
    ----------
    case : dict
        Native model description with ordered regions, sampling, energy groups and
        depletion-tally settings.
    directory : str or pathlib.Path
        Prepared calculation directory containing the completed statepoint.
    batch_scores : numpy.ndarray
        Active-batch scores with spatial-bin and final heating/eigenvalue-score axes.
    materials : sequence of openmc.Material or None, optional
        Ordered native materials; None reconstructs them from the case.
    started : float or None, optional
        perf_counter start time for elapsed-time accounting; None starts at extraction.
    collapsed : dict or None, optional
        Completed histogram-collapse scores, required for flux-collapse depletion
        estimators.
    factorized_statistics : bool, optional
        Retain a low-rank covariance factor instead of a dense statistical covariance.

    Returns
    -------
    dict
        Sampling, criticality, source rate, physical per-parent depletion rates in s⁻¹ and,
        when requested, measured cross sections and response observables.

    Raises
    ------
    ValueError
        Statepoint settings, required collapse data or sampled flux groups are inconsistent.
    ArithmeticError
        Microscopic data fail to reconstruct the native reaction tallies.

    Notes
    -----
    This reads the completed statepoint without rerunning transport. It may save
    microscopic.npz and depletion_spectrum.npz as extracted companion artifacts.
    """
    directory = Path(directory)
    started = time.perf_counter() if started is None else started
    materials = material_list(case) if materials is None else materials
    sampling = case['sampling']
    m, g, n = len(materials), len(case['group_edges_eV'])-1, len(case['nuclides'])
    density = np.array([[material.get_nuclide_atom_densities().get(name, 0.) for name in case['nuclides']]
                        for material in materials])
    volumes = np.array([material.volume for material in materials])
    shape = tuple(len(e)-1 for e in case['comparison_edges_cm'])
    dx, dy, dz = (np.diff(e) for e in case['comparison_edges_cm'])
    bin_volumes = dx[:, None, None]*dy[None, :, None]*dz[None, None, :]

    def xyz(array):
        """Reorder OpenMC mesh bins into the public x/y/z axis order.

        Parameters
        ----------
        array : numpy.ndarray
            Flat mesh data in OpenMC's x-fastest ordering.

        Returns
        -------
        numpy.ndarray
            Three-dimensional array with shape (nx, ny, nz).
        """
        # OpenMC mesh indices have x fastest; the public interface uses x,y,z.
        return array.reshape(shape[::-1]).transpose(2, 1, 0)

    with openmc.StatePoint(directory/f'statepoint.{sampling["batches"]}.h5') as state:
        if state.n_particles != sampling['particles'] or state.seed != sampling['seed'] or state.n_inactive != sampling['inactive']:
            raise ValueError('The saved statepoint does not match the requested sampling')
        stat = statistics(batch_scores, case['power_W'], case['statistics_block_size'],
                          factorized=factorized_statistics)
        heating_per_source = state.get_tally(id=1).mean.reshape(-1, 2)[:, 0].sum()
        source_rate = case['power_W']/(heating_per_source*1.602176634e-19)
        depletable = np.array([i for i, material in enumerate(materials) if material.depletable])
        dn = len(case['depletion_nuclides'])
        if not case.get('need_depletion_rates', True):
            rates, collapse_check = None, None
        elif case['depletion_rate_estimator'] == DIRECT_RATES:
            rates = state.get_tally(id=9).mean.reshape(len(depletable), dn, len(case['depletion_reactions']))
            collapse_check = None
        else:
            if not collapsed:
                raise ValueError('Completed native flux-collapse data are required')
            rates = collapsed['rates'].copy()
            direct = state.get_tally(id=9).mean.reshape(len(depletable), len(case['depletion_direct_nuclides']),
                                                       len(case['depletion_direct_reactions']))
            checks = []
            for j, name in enumerate(case['depletion_direct_nuclides']):
                parent = case['depletion_nuclides'].index(name)
                for r, reaction in enumerate(case['depletion_direct_reactions']):
                    mt = case['depletion_reactions'].index(reaction)
                    denominator = np.sum(np.abs(direct[:, j, r]))
                    if denominator > 0:
                        checks.append({'nuclide': name, 'reaction': reaction,
                                       'summed_relative_error': float((rates[:, parent, mt].sum()-direct[:, j, r].sum())/denominator)})
                    rates[:, parent, mt] = direct[:, j, r]
            collapse_check = {'groups': len(collapsed['energy_eV'])-1, 'direct_comparison': checks,
                              'direct_rates_override_histogram': True}
            np.savez_compressed(directory/'depletion_spectrum.npz', **collapsed)
        per_parent = None if rates is None else rates*source_rate/(1e24*volumes[depletable, None, None])
        record = {'status': 'completed', 'sampling': sampling,
                  'combined_keff': [state.keff.n, state.keff.s], 'statistics': stat,
                  'region_labels': [m.name for m in materials], 'region_volumes_cm3': volumes,
                  'nuclides': case['nuclides'], 'depletion_reactions': case['depletion_reactions'],
                  'depletion_nuclides': case['depletion_nuclides'],
                  'depletion_rate_estimator': case['depletion_rate_estimator'],
                  'depletion_flux_check': collapse_check,
                  'depletion_region_indices': depletable,
                  'depletion_rates_per_parent_s': per_parent, 'source_rate_per_s': source_rate,
                  'entropy': state.entropy.tolist(), 'k_generation': state.k_generation.tolist(),
                  'runtime_s': dict(state.runtime), 'wall_s': time.perf_counter()-started,
                  'full_observations': case.get('full_observations', True),
                  'observation_cross_sections': case.get('observation_cross_sections', 'microscopic_tallies'),
                  'need_depletion_rates': case.get('need_depletion_rates', True),
                  'normalization': 'Specified fission power; source-normalized flux and reaction tallies'}
        if not case.get('full_observations', True):
            return record
        flux = state.get_tally(id=2).mean.reshape(m, g)
        analog_flux = state.get_tally(id=6).mean.reshape(m, g)
        if min(flux.min(), analog_flux.min()) <= 0:
            raise ValueError('Unsampled region/energy group in the native reference')
        micro = case.get('observation_cross_sections', 'microscopic_tallies') == 'microscopic_tallies'
        scored = n if micro else 1
        emission = state.get_tally(id=7).mean.reshape(m, g)
        chi = np.zeros((m, g))
        chi[:, -1] = 1.
        emitting = emission.sum(-1) > 0
        chi[emitting] = emission[emitting]/emission[emitting].sum(-1, keepdims=True)
        microscopic = tally_cross_sections(state, (3, 4, 5), flux, analog_flux, scored,
                                            atom_density=density if micro else None)
        macro = {name: np.einsum('mn,mn...->m...', density, value) if micro else value[:, 0]
                 for name, value in microscopic.items()}
        if case.get('audit_microscopic_observations', False):
            audit = tally_cross_sections(state, (13, 14, 15), flux, analog_flux, n, atom_density=density)
            errors = {name: float(np.max(np.abs(np.einsum('mn,mn...->m...', density, value)-macro[name])) /
                                  max(1e-30, np.max(np.abs(macro[name])))) for name, value in audit.items()}
            record['microscopic_observation_audit'] = errors
        macro['chi'] = chi
        macro['transport'] = macro['absorption']+macro['scatter'].sum(-1)-macro['scatter_p1']
        response = case.get('material_response')
        if response:
            selected = np.array(response['region_indices'], dtype=int)
            names = response['nuclides']
            atoms = np.array([[materials[i].get_nuclide_atom_densities().get(name, 0.) for name in names]
                              for i in selected])
            response_micro = tally_cross_sections(state, (16, 17, 18), flux[selected], analog_flux[selected],
                                                  len(names), atom_density=atoms)
            record['material_response'] = {'region_indices': selected, 'nuclides': names,
                                           'atom_density': atoms, 'microscopic': response_micro,
                                           'analog_sampled': atoms > 0,
                                           'spectrum': 'Frozen at this native reference; updated after a new reference'}
        raw = state.get_tally(id=10).mean.reshape(m, g, 2)
        reconstructed = np.stack((macro['absorption']*flux, macro['fission']*flux), axis=-1)
        reconstruction_error = np.max(np.abs(reconstructed-raw))/max(1e-30, np.max(np.abs(raw)))
        if reconstruction_error > 1e-8:
            raise ArithmeticError('Microscopic data do not reconstruct native macroscopic tallies')
        axes = [{'name': name, 'edges': e, 'unit': 'cm'}
                for name, e in zip(('x', 'y', 'z'), case['comparison_edges_cm'])]
        layout = {'axes': axes, 'unit': 'W/cm^3', 'normalization': f'{case["power_W"]:g}_W_total',
                  'measure': 'volume_average'}
        observations = {'keff': {'mean': stat['mean'][0], 'scale': stat['se'][0]},
                        'power_shape': {'mean': xyz(np.array(stat['mean'][1:]))/bin_volumes,
                                        'scale': xyz(np.array(stat['se'][1:]))/bin_volumes, 'layout': layout},
                        'region_flux_per_source': {'mean': flux,
                                                  'scale': state.get_tally(id=2).std_dev.reshape(m, g)}}
        # The sum of SDs is a conservative scale when cross-tally covariance is absent.
        capture = (raw[..., 0]-raw[..., 1]).sum(-1)
        capture_scale = state.get_tally(id=10).std_dev.reshape(m, g, 2).sum((1, 2))
        observations['capture_per_source'] = {'mean': capture, 'scale': capture_scale}
        for group in range(g):
            flux_mesh = state.get_tally(id=8).mean.reshape(-1, g)[:, group]
            flux_sd = state.get_tally(id=8).std_dev.reshape(-1, g)[:, group]
            group_layout = {'axes': axes, 'unit': '1/cm^2/source', 'normalization': 'source_neutron',
                            'measure': 'volume_average', 'energy_interval_eV': case['group_edges_eV'][group:group+2]}
            observations[f'flux_shape_{group}'] = {'mean': xyz(flux_mesh)/bin_volumes,
                                                  'scale': xyz(flux_sd)/bin_volumes, 'layout': group_layout}
        # Group transfer is separately identifiable from integrated flux/capture.
        transfer = state.get_tally(id=11).mean.reshape(m, g, g)
        transfer_sd = state.get_tally(id=11).std_dev.reshape(m, g, g)
        observations['scatter_per_source'] = {'mean': transfer, 'scale': transfer_sd}
        record.update(observables=observations, macro=macro,
                      microscopic_reconstruction_error=float(reconstruction_error),
                      analog_isotope_normalization='Native analog isotope scores include density; divide by density before microscopic contraction')
        if micro:
            np.savez_compressed(directory/'microscopic.npz', atom_density=density,
                                **{name: value for name, value in microscopic.items()}, chi=chi,
                                batch_scores=batch_scores)
    return record
