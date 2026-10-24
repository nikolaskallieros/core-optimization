"""Native OpenMC batch access and statistics shared by transport adapters."""
import atexit
import ctypes as ct
import os
from pathlib import Path

import numpy as np
from runtime.processes import native_library_path

LIBRARY = native_library_path()
_MPI_CLEANUP_REGISTERED = False


def _finalize_owned_mpi(dll):
    """Finalize adapter-owned MPI at process exit if it remains active.

    Parameters
    ----------
    dll : ctypes.CDLL
        Initialized native OpenMC shared library with the current model and tally data
        loaded.
    """
    initialized, finalized = ct.c_int(), ct.c_int()
    dll.MPI_Initialized(ct.byref(initialized))
    dll.MPI_Finalized(ct.byref(finalized))
    if initialized.value and not finalized.value:
        dll.MPI_Finalize()


def native_batches(directory, inactive, *, threads=1, postprocess=None, batch_checkpoint=None):
    """Collect active-batch scores through the native OpenMC C API.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing prepared OpenMC XML inputs.
    inactive : int
        Number of source-convergence batches excluded from returned scores.
    threads : int, optional
        Positive OpenMP thread count for the native calculation.
    postprocess : callable or None, optional
        Callback receiving the loaded OpenMC library after simulation finalization but
        before model cleanup.
    batch_checkpoint : str or pathlib.Path or None, optional
        Optional NPZ destination for active-batch score recovery.

    Returns
    -------
    numpy.ndarray
        Independent active-batch scores from tally 1, ordered by batch, tally bin and score.

    Raises
    ------
    ValueError
        The thread count is not a positive integer.
    RuntimeError
        A native OpenMC API call reports failure.

    Notes
    -----
    Run postprocess while native data remain loaded, then finalize transport and restore the
    working directory even if evaluation fails.
    """
    global _MPI_CLEANUP_REGISTERED
    if type(threads) is not int or threads < 1:
        raise ValueError('Native thread count must be a positive integer')
    dll = ct.CDLL(str(LIBRARY))
    had_mpi = ct.c_int(1)
    if hasattr(dll, 'MPI_Initialized'):
        for name in ('MPI_Initialized', 'MPI_Finalized'):
            getattr(dll, name).argtypes, getattr(dll, name).restype = [ct.POINTER(ct.c_int)], ct.c_int
        dll.MPI_Finalize.argtypes, dll.MPI_Finalize.restype = [], ct.c_int
        dll.MPI_Initialized(ct.byref(had_mpi))
    signatures = {
        'openmc_init': [ct.c_int, ct.POINTER(ct.c_char_p), ct.c_void_p],
        'openmc_simulation_init': [], 'openmc_simulation_finalize': [], 'openmc_finalize': [],
        'openmc_next_batch': [ct.POINTER(ct.c_int)],
        'openmc_get_tally_index': [ct.c_int32, ct.POINTER(ct.c_int32)],
        'openmc_tally_results': [ct.c_int32, ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_size_t*3)],
    }
    for name, arguments in signatures.items():
        getattr(dll, name).argtypes, getattr(dll, name).restype = arguments, ct.c_int

    def call(name, *args):
        """Invoke an OpenMC C function and surface its native error message.

        Parameters
        ----------
        name : str
            Native function symbol to invoke.
        *args : tuple
            ctypes arguments required by the selected function.

        Raises
        ------
        RuntimeError
            The native call returns a nonzero error code.
        """
        code = getattr(dll, name)(*args)
        if code:
            try:
                detail = (ct.c_char*256).in_dll(dll, 'openmc_err_msg').value.decode()
            except ValueError:
                detail = ''
            raise RuntimeError(f'{name} failed ({code}): {detail}')

    previous_cwd = Path.cwd()
    os.chdir(directory)
    initialized = False
    try:
        argv = (ct.c_char_p*4)(b'openmc', b'-s', str(threads).encode(), None)
        call('openmc_init', 3, argv, None)
        initialized = True
        if not had_mpi.value and not _MPI_CLEANUP_REGISTERED:
            # openmc_finalize releases OpenMC, not MPI. Defer MPI finalization
            # to process exit so repeated library simulations remain possible.
            atexit.register(_finalize_owned_mpi, dll)
            _MPI_CLEANUP_REGISTERED = True
        call('openmc_simulation_init')
        index = ct.c_int32()
        call('openmc_get_tally_index', 1, ct.byref(index))
        status, samples, previous, batch = ct.c_int(), [], None, 0
        while True:
            call('openmc_next_batch', ct.byref(status))
            batch += 1
            if batch > inactive:
                ptr, shape = ct.POINTER(ct.c_double)(), (ct.c_size_t*3)()
                call('openmc_tally_results', index, ct.byref(ptr), ct.byref(shape))
                cumulative = np.ctypeslib.as_array(ptr, tuple(shape))[:, :, 1].copy()
                samples.append(cumulative if previous is None else cumulative-previous)
                previous = cumulative
            if status.value:
                break
        call('openmc_simulation_finalize')
        if batch_checkpoint is not None:
            np.savez_compressed(batch_checkpoint, batch_scores=np.array(samples))
        if postprocess is not None:
            postprocess(dll)
        return np.array(samples)
    finally:
        if initialized:
            call('openmc_finalize')
        os.chdir(previous_cwd)


def statistics(batch_scores, power, block_size, *, factorized=False):
    """Estimate criticality and fixed-power shapes using a block jackknife.

    Parameters
    ----------
    batch_scores : numpy.ndarray
        Active-batch scores with spatial-bin and final heating/eigenvalue-score axes.
    power : float
        Total power in W used to normalize the spatial heating scores.
    block_size : int
        Number of consecutive active batches per statistical block.
    factorized : bool, optional
        Store covariance as block-deviation factors instead of a dense matrix.

    Returns
    -------
    dict
        Response mean, standard errors, covariance or its factor, block count and batches
        per block. The response vector starts with keff, followed by spatial powers in W.

    Raises
    ------
    ValueError
        The active batches do not form at least two complete blocks.

    Notes
    -----
    Each batch stores heating and eigenvalue contributions in its last axis. A factorized
    covariance stores block deviations instead of a dense matrix; both forms retain
    covariance between eigenvalue and normalized power bins.
    """
    if len(batch_scores) % block_size:
        raise ValueError('Active batches must be divisible by block size')
    blocks = batch_scores.reshape(-1, block_size, *batch_scores.shape[1:]).mean(1)
    if len(blocks) < 2:
        raise ValueError('Statistics require at least two active blocks')

    def response(scores):
        """Normalize heating scores to the requested fixed total power.

        Parameters
        ----------
        scores : numpy.ndarray
            Mean or leave-one-block-out scores with final heating and eigenvalue-contribution
            columns.

        Returns
        -------
        numpy.ndarray
            Response vectors containing summed keff contributions and normalized spatial powers.
        """
        q = scores[..., 0]
        return np.concatenate((scores[..., 1].sum(-1, keepdims=True), power*q/q.sum(-1, keepdims=True)), axis=-1)

    mean = blocks.mean(0)
    leave_one_out = response((blocks.sum(0)-blocks)/(len(blocks)-1))
    centered = leave_one_out-leave_one_out.mean(0)
    if factorized:
        factor = centered * np.sqrt((len(blocks) - 1) / len(blocks))
        variance = np.square(factor).sum(0)
        covariance = {'covariance_factor': factor,
                      'covariance_definition': 'factor.T @ factor; jackknife block deviations'}
    else:
        matrix = (len(blocks)-1)/len(blocks)*centered.T@centered
        variance, covariance = np.diag(matrix), {'covariance': matrix.tolist()}
    return {'mean': response(mean).tolist(), 'se': np.sqrt(variance).tolist(),
            **covariance, 'blocks': len(blocks), 'batches_per_block': block_size}
