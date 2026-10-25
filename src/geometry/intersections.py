"""Exact differentiable intersections used by CAD-to-diffusion mappings."""

import math

import torch


def _quarter_area(x, y):
    """Integrate the unit disk over signed rectangular limits.

    Parameters
    ----------
    x : torch.Tensor
        Dimensionless signed x limits, broadcast with y.
    y : torch.Tensor
        Dimensionless signed y limits, broadcast with x.

    Returns
    -------
    torch.Tensor
        Signed disk area between the origin and each pair of limits.
    """
    sx, sy = torch.where(x < 0, -1., 1.), torch.where(y < 0, -1., 1.)
    a, b = (sx*x).clamp(max=1.), (sy*y).clamp(max=1.)
    # Finite derivatives at tangencies; the 1e-14 floor changes area only at
    # roundoff and avoids 0*infinity from differentiating sqrt(0).
    ra, rb = (1-a*a).clamp_min(1e-28).sqrt(), (1-b*b).clamp_min(1e-28).sqrt()
    outside = .5*(a*ra+b*rb+torch.atan2(a, ra)+torch.atan2(b, rb)-math.pi/2)
    return sx*sy*torch.where(a*a+b*b <= 1., a*b, outside)


def circle_fractions(x_edges, y_edges, centers, radii):
    """Compute analytical circle intersections with Cartesian cells.

    Parameters
    ----------
    x_edges : torch.Tensor
        Increasing x-cell edges in cm.
    y_edges : torch.Tensor
        Increasing y-cell edges in cm.
    centers : torch.Tensor
        Cylinder centers in cm with shape (n_cylinders, 2).
    radii : torch.Tensor
        Cylinder radii in cm with shape (n_cylinders,).

    Returns
    -------
    torch.Tensor
        Circle area fractions with shape (ny * nx, n_cylinders), in row-major y/x cell
        order.
    """
    x = (x_edges[None, :, None]-centers[:, None, 0:1])/radii[:, None, None]
    y = (y_edges[None, None, :]-centers[:, None, 1:2])/radii[:, None, None]
    primitive = _quarter_area(x, y)*radii[:, None, None]**2
    area = primitive[:, 1:, 1:]-primitive[:, :-1, 1:]-primitive[:, 1:, :-1]+primitive[:, :-1, :-1]
    cell_area = torch.diff(x_edges)[:, None]*torch.diff(y_edges)[None, :]
    return (area/cell_area).permute(2, 1, 0).reshape(-1, len(radii)).clamp(0., 1.)


def circle_triangle_area(vertices, center, radius):
    """Integrate exact circle-polygon intersections with live derivatives.

    Parameters
    ----------
    vertices : torch.Tensor
        Counterclockwise polygon vertices in cm, shaped (n_polygons, n_vertices, 2).
    center : torch.Tensor
        Circle x/y center in cm.
    radius : torch.Tensor
        Scalar circle radius in cm.

    Returns
    -------
    torch.Tensor
        Intersection area in cm² for each polygon; triangles are the common case.
    """
    p = vertices - center
    q = torch.roll(p, -1, dims=1)
    delta = q - p
    aa, bb = delta.square().sum(-1), 2 * (p * delta).sum(-1)
    aa = torch.where(aa > 0, aa, torch.ones_like(aa))
    cc = p.square().sum(-1) - radius.square()
    discriminant = bb.square() - 4 * aa * cc
    root = torch.sqrt(torch.where(discriminant > 0, discriminant, torch.ones_like(discriminant)))
    left = torch.where(discriminant > 0, (-bb - root) / (2 * aa), torch.zeros_like(aa)).clamp(0, 1)
    right = torch.where(discriminant > 0, (-bb + root) / (2 * aa), torch.zeros_like(aa)).clamp(0, 1)
    cuts = torch.stack((torch.zeros_like(aa), left, right, torch.ones_like(aa)), dim=-1)
    u = p[..., None, :] + cuts[..., :-1, None] * delta[..., None, :]
    v = p[..., None, :] + cuts[..., 1:, None] * delta[..., None, :]
    cross, dot = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1)
    dot = torch.where((cross == 0) & (dot == 0), torch.ones_like(dot), dot)
    inside = ((u + v) / 2).square().sum(-1) <= radius.square()
    contribution = torch.where(inside, cross, radius.square() * torch.atan2(cross, dot)) / 2
    return contribution.sum(dim=(1, 2))
