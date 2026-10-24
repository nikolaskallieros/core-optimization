"""Fingerprint evaluated tables and integrate their piecewise-linear rates."""
import hashlib
import numpy as np


def dataset_hashes(path):
    """Fingerprint HDF5 dataset values, shapes, dtypes and attributes.

    Parameters
    ----------
    path : str or pathlib.Path
        Processed nuclear-data HDF5 file.

    Returns
    -------
    dict of str to str
        Dataset paths mapped to SHA-256 fingerprints for comparing temperature extensions.
    """
    import h5py
    values = {}
    with h5py.File(path) as handle:
        def visit(name, item):
            """Record the content fingerprint of one visited HDF5 dataset.

            Parameters
            ----------
            name : str
                Dataset or group path within the HDF5 file.
            item : h5py.Dataset or h5py.Group
                Visited object; groups are ignored.
            """
            if isinstance(item, h5py.Dataset):
                value = item[()]
                metadata = repr((item.shape, str(item.dtype), sorted((k, repr(v)) for k, v in item.attrs.items())))
                values[name] = hashlib.sha256(metadata.encode() + value.tobytes()).hexdigest()
        handle.visititems(visit)
    return values


def table_collapse(path, edges, spectrum, *, temperature, mt):
    """Independently integrate piecewise-linear evaluated reaction tables.

    Parameters
    ----------
    path : str or pathlib.Path
        OpenMC neutron-data HDF5 file.
    edges : numpy.ndarray
        Increasing energy histogram edges in eV.
    spectrum : numpy.ndarray
        Integrated spectrum weights for the histogram bins.
    temperature : float
        Requested interpolation temperature in K.
    mt : int
        ENDF reaction MT identifier.

    Returns
    -------
    float
        Temperature-interpolated sum of bin-mean microscopic cross sections in barns times
        the supplied spectrum weights.

    Raises
    ------
    ValueError
        A reaction table uses an interpolation law other than linear-linear.
    """
    import openmc.data
    data = openmc.data.IncidentNeutron.from_hdf5(path)
    values = []
    temperatures = np.array(data.kTs) / openmc.data.K_BOLTZMANN
    for label in data.temperatures:
        xs = data.reactions[mt].xs[label]
        if not np.all(xs.interpolation == 2):
            raise ValueError("Independent rate check requires linear-linear evaluated tables")
        x, y = xs.x, xs.y
        cumulative = xs.integral()
        points = np.clip(edges, x[0], x[-1])
        i = np.clip(np.searchsorted(x, points, side="right") - 1, 0, len(x) - 2)
        dx = points - x[i]
        integral = cumulative[i] + y[i] * dx + .5 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * dx**2
        values.append(float(np.sum(np.diff(integral) * spectrum / np.diff(edges))))
    order = np.argsort(temperatures)
    return float(np.interp(temperature, temperatures[order], np.array(values)[order]))
