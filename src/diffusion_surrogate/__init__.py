"""Torch diffusion with a separate native OpenMC evaluation interface.

Load Torch only when requested. Native OpenMC subprocesses do not need a second
copy of the Torch runtime while holding their continuous-energy nuclear data.
"""
from importlib import import_module

_EXPORTS = {name: module for module, names in {
    'solver': ('Diffusion', 'Grid', 'Materials', 'circle_fractions'),
    'calibration': ('CalibratedDiffusion',),
    'responses': ('Response', 'rebin', 'cartesian_responses'),
    'solver3d': ('Grid3D', 'Diffusion3D', 'cylinder_regions'),
    'regional': ('RegionalSurrogate',),
    'geometry': ('DiffusionGeometry',),
    'equivalence': ('equivalence_bases', 'shared_bases'),
    'cylinders': ('Cylinder', 'HexagonalDomain', 'channel_regions', 'cylinder_fractions'),
}.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load a public diffusion interface on first attribute access.

    Parameters
    ----------
    name : str
        Name declared in the package's public export mapping.

    Returns
    -------
    object
        Imported interface, cached in the package namespace.

    Raises
    ------
    AttributeError
        The requested name is not exported.
    """
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = _EXPORTS[name]
    target = f'diffusion_surrogate.{module}'
    value = getattr(import_module(target), name)
    globals()[name] = value
    return value
