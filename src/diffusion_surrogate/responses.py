"""Model-defined response axes and conservative spatial averaging for fitting."""
from dataclasses import dataclass
import copy
import math

import torch


@dataclass(frozen=True)
class Response:
    """A tensor with its physical comparison layout, independent of the solver.

    layout contains unit, normalization, measure, and ordered axes. Each axis
    has a name and either labels (regions/materials) or edges plus a unit.
    """
    values: torch.Tensor
    layout: dict


def cartesian_responses(result, edges_cm, comparison_edges_cm, group_edges_eV, power_W):
    """Rebin Cartesian diffusion fields onto an explicit comparison mesh.

    Parameters
    ----------
    result : dict
        Solver outputs containing power_density_W_cm3 and flux_per_source, with energy
        group as the final flux axis.
    edges_cm : sequence of array_like
        Source mesh edges in x, y and optionally z order, in cm.
    comparison_edges_cm : sequence of array_like
        Target comparison-bin edges in the same coordinate order and units.
    group_edges_eV : sequence of float
        Energy-group edges in eV, ordered as the solver's group axis.
    power_W : float
        Prescribed total power in W used to label the normalization.

    Returns
    -------
    dict of str to Response
        Power density and per-group source-normalized flux on the comparison mesh.

    Notes
    -----
    This optional observation adapter changes neither cross sections nor physical geometry.
    Other mesh types can supply their own response adapter to the same calibrated solver.
    """
    axes = [{'name': name, 'edges': torch.as_tensor(edge, dtype=torch.float64).tolist(), 'unit': 'cm'}
            for name, edge in zip(('x', 'y', 'z'), edges_cm)]
    power_layout = {'axes': axes, 'unit': 'W/cm^3', 'normalization': f'{power_W:g}_W_total',
                    'measure': 'volume_average'}
    group_edges = torch.as_tensor(group_edges_eV, dtype=torch.float64).tolist()
    fields = {'power_shape': rebin(Response(result['power_density_W_cm3'], power_layout), comparison_edges_cm)}
    for group in range(len(group_edges) - 1):
        layout = {'axes': axes, 'unit': '1/cm^2/source', 'normalization': 'source_neutron',
                  'measure': 'volume_average', 'energy_interval_eV': group_edges[group:group + 2]}
        fields[f'flux_shape_{group}'] = rebin(Response(result['flux_per_source'][..., group], layout),
                                              comparison_edges_cm)
    return fields


def validate_layout(layout, shape):
    """Validate physical response metadata against its tensor shape.

    Parameters
    ----------
    layout : dict
        Response units, normalization, measure and ordered axes with labels or dimensioned
        edges.
    shape : sequence of int
        Shape of the response tensor.

    Raises
    ------
    ValueError
        Units, axis identities, labels or bin edges are missing, ambiguous or incompatible.
    """
    for key in ('unit', 'normalization', 'measure'):
        if not isinstance(layout.get(key), str) or not layout[key]:
            raise ValueError(f'Response layout needs an explicit {key}')
    axes = layout['axes']
    if len(axes) != len(shape) or len({axis['name'] for axis in axes}) != len(axes):
        raise ValueError('Response axes must uniquely name every tensor dimension')
    for axis, size in zip(axes, shape):
        if ('edges' in axis) == ('labels' in axis):
            raise ValueError('Each axis needs either bin edges or ordered labels')
        if 'edges' in axis:
            edges = axis['edges']
            if (not axis.get('unit') or len(edges) != size+1 or
                    not all(math.isfinite(v) for v in edges) or
                    any(b <= a for a, b in zip(edges, edges[1:]))):
                raise ValueError('Axis edges must be finite, increasing, dimensioned and match the tensor')
        elif len(axis['labels']) != size or len(set(axis['labels'])) != size:
            raise ValueError('Axis labels must be unique and match the tensor')


def prediction_values(prediction, reference):
    """Require matching prediction and reference shapes and physical layouts.

    Parameters
    ----------
    prediction : torch.Tensor or Response
        Live model prediction, including a layout when the reference declares one.
    reference : dict
        Reference mean and optional physical layout.

    Returns
    -------
    torch.Tensor
        Validated live prediction tensor.

    Raises
    ------
    ValueError
        Tensor shape or physical layout differs from the reference.
    """
    if isinstance(prediction, Response):
        value = prediction.values
        validate_layout(prediction.layout, value.shape)
        if prediction.layout != reference.get('layout'):
            raise ValueError('Prediction/reference axes, bin edges, units or averaging differ')
    else:
        value = prediction
        if 'layout' in reference:
            raise ValueError('The model must declare the reference response layout')
    if value.shape != torch.as_tensor(reference['mean']).shape:
        raise ValueError('Prediction and reference tensor shapes differ')
    return value


def rebin(response, edges):
    """Project a piecewise-constant field onto shared physical intervals.

    Parameters
    ----------
    response : Response
        Field carrying ordered spatial edges and either integral or volume_average
        semantics.
    edges : sequence of array_like
        Target bin edges in the source axis order and units, within the source domain.

    Returns
    -------
    Response
        Conservatively rebinned field with updated layout and retained value derivatives.

    Raises
    ------
    ValueError
        Layouts, averaging semantics or target bounds are incompatible.

    Notes
    -----
    Edges follow the response's declared axis order and units. Integrals conserve totals;
    volume averages use geometric overlap weights, including unequal bin lengths. The field
    is piecewise constant within each input bin. This is an averaging operation, not a
    finer-resolution reconstruction or a transport solve. Ordinary Torch operations retain
    derivatives with respect to field values.
    """
    validate_layout(response.layout, response.values.shape)
    layout = copy.deepcopy(response.layout)
    axes = layout['axes']
    if len(edges) != len(axes) or not all('edges' in axis for axis in axes):
        raise ValueError('Spatial rebinning needs edges for every ordered axis')
    measure = layout['measure']
    if measure not in ('volume_average', 'integral'):
        raise ValueError('Spatial rebinning requires volume_average or integral data')
    for axis, new_edges in zip(layout['axes'], edges):
        axis['edges'] = list(new_edges)
    validate_layout(layout, tuple(len(values)-1 for values in edges))
    values = response.values
    for dim, (axis, new_edges) in enumerate(zip(response.layout['axes'], edges)):
        if new_edges[0] < axis['edges'][0] or new_edges[-1] > axis['edges'][-1]:
            raise ValueError('Comparison bins extend outside the calculated field')
        source = torch.as_tensor(axis['edges'], dtype=values.dtype, device=values.device)
        target = torch.as_tensor(new_edges, dtype=values.dtype, device=values.device)
        overlap = (torch.minimum(target[1:, None], source[None, 1:])-
                   torch.maximum(target[:-1, None], source[None, :-1])).clamp_min(0)
        widths = target.diff()[:, None] if measure == 'volume_average' else source.diff()[None, :]
        values = torch.tensordot(overlap/widths, values, dims=([1], [dim])).movedim(0, dim)
    return Response(values, layout)
