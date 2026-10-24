"""Manufactured power objective used to check propagated uncertainty."""
import torch
from diffusion_surrogate.solver import tensor
from diffusion_surrogate.responses import Response


def power_objective(outputs):
    """Compute a volume-weighted eighth-power norm of relative power.

    Parameters
    ----------
    outputs : mapping
        A power_shape Response with Cartesian bin edges and power-density values.

    Returns
    -------
    torch.Tensor
        Smooth dimensionless scalar power-shape objective.
    """
    response = outputs['power_shape']
    widths = [tensor(axis['edges']).diff() for axis in response.layout['axes']]
    volume = widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]
    mean = (response.values*volume).sum()/volume.sum()
    return (((response.values/mean)**8*volume).sum()/volume.sum())**(1/8)


@torch.enable_grad()
def objective_se(snapshot):
    """Propagate retained joint power covariance to objective uncertainty.

    Parameters
    ----------
    snapshot : Snapshot
        Reference snapshot containing joint blocked power means, covariance and layout.

    Returns
    -------
    float
        Delta-method standard error of the dimensionless power objective.
    """
    record = snapshot.state['reference']
    observation = record['observables']['power_shape']
    shape = tuple(len(axis['edges'])-1 for axis in observation['layout']['axes'])
    widths = [tensor(axis['edges']).diff() for axis in observation['layout']['axes']]
    volume = widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]
    q = tensor(record['statistics']['mean'][1:]).requires_grad_()
    values = q.reshape(shape[::-1]).permute(2, 1, 0)/volume
    value = power_objective({'power_shape': Response(values, observation['layout'])})
    derivative, = torch.autograd.grad(value, q)
    covariance = tensor(record['statistics']['covariance'])[1:, 1:]
    return float((derivative@covariance@derivative).clamp_min(0).sqrt())
