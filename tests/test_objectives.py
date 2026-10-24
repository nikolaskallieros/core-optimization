import pytest
import torch

from optimization.objectives import aggregate_channel_power, discharge_burnup_objective, evaluate_objectives, peak_burnup_constraint, radial_power_objective, scalarize, assembly_power_objective


def tensor(value, **kwargs):
    """Create a float64 tensor for objective checks.

    Parameters
    ----------
    value : array_like
        Numerical fixture values.
    **kwargs : object
        Additional torch.tensor options such as requires_grad.

    Returns
    -------
    torch.Tensor
        Float64 fixture tensor.
    """
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def metrics():
    """Evaluate a small explicit burnup and power objective fixture.

    Returns
    -------
    ObjectiveMetrics
        Dimensionless mass-weighted burnup and time-weighted power objectives.
    """
    return evaluate_objectives(
        discharge_burnup=tensor([90, 120]), discharge_initial_hm_mass=tensor([1, 2]),
        target_burnup=100, bundle_power=tensor([[[1, 2], [3, 1]], [[2, 2], [1, 1]]]),
        time_weights=tensor([1, 3]),
    )


def test_target_deviation_includes_bias_even_when_spread_is_zero():
    """Verify target deviation includes bias even when spread is zero."""
    value = discharge_burnup_objective(tensor([120, 120]), tensor([1, 5]), 100)
    assert value.item() == pytest.approx(0.04)
    value = discharge_burnup_objective(tensor([90, 120]), tensor([1, 2]), 100)
    assert value.item() == pytest.approx((0.01 + 2 * 0.04) / 3)


def test_channel_aggregation_conserves_power_and_sums_axial_stack():
    """Verify channel aggregation conserves power and sums axial stack."""
    power = tensor([[[1, 3], [2, 2]], [[2, 6], [4, 4]]])
    channels = aggregate_channel_power(power)
    torch.testing.assert_close(channels, tensor([[4, 4], [8, 8]]))
    torch.testing.assert_close(channels.sum(dim=1), power.sum(dim=(1, 2)))
    assert radial_power_objective(power, tensor([1, 3])).item() == 0


def test_assembly_power_resolves_axial_shape_and_matches_full_core():
    """Verify assembly power resolves axial shape and matches full core."""
    power = tensor([[1., 3., 2., 2.], [2., 2., 3., 1.]], requires_grad=True)
    mult = torch.tensor([1, 1, 6, 6])
    expanded = power.repeat_interleave(mult, dim=1)
    expected = ((expanded / expanded.mean(dim=1, keepdim=True) - 1).square().mean(dim=1)
                * tensor([1., 3.])).sum() / 4
    actual = assembly_power_objective(power, [1., 3.], assembly_multiplicities=mult.tolist())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, power)[0], torch.autograd.grad(expected, power)[0])
    assert actual > 0
    assert radial_power_objective(power.reshape(2, 2, 2), [1., 3.], channel_multiplicities=[1, 6]) == 0
    assert assembly_power_objective(torch.ones_like(power), [1., 3.], assembly_multiplicities=mult.tolist()) == 0


def test_full_core_and_orbit_weighted_objectives_are_identical():
    """Verify full core and orbit weighted objectives are identical."""
    orbit_power = tensor([[[1, 2], [3, 1]], [[2, 2], [1, 1]]])
    full_power = orbit_power.repeat_interleave(torch.tensor([1, 6]), dim=1)
    orbit_burnup, orbit_mass = tensor([90, 120]), tensor([1, 2])
    full_burnup = orbit_burnup.repeat_interleave(torch.tensor([1, 6]))
    full_mass = orbit_mass.repeat_interleave(torch.tensor([1, 6]))
    orbit_b = discharge_burnup_objective(orbit_burnup, orbit_mass, 100, multiplicities=[1, 6])
    full_b = discharge_burnup_objective(full_burnup, full_mass, 100)
    orbit_p = radial_power_objective(orbit_power, [1, 3], channel_multiplicities=[1, 6])
    full_p = radial_power_objective(full_power, [1, 3])
    torch.testing.assert_close(orbit_b, full_b)
    torch.testing.assert_close(orbit_p, full_p)
    torch.testing.assert_close((aggregate_channel_power(orbit_power) * tensor([1, 6])).sum(1),
                               aggregate_channel_power(full_power).sum(1))
    assert not torch.isclose(orbit_b, discharge_burnup_objective(orbit_burnup, orbit_mass, 100))


def test_time_weights_are_used_and_uniform_power_scale_cancels():
    """Verify time weights are used and uniform power scale cancels."""
    power = tensor([[[1], [3]], [[2], [2]]])
    assert radial_power_objective(power, [1, 3]).item() == pytest.approx(0.25 / 4)
    torch.testing.assert_close(radial_power_objective(power, [1, 3]),
                               radial_power_objective(13 * power, [2, 6]))
    assert radial_power_objective(power, [0, 3]).item() == 0


def test_scalarization_is_explicit_and_zero_weights_cannot_hide_invalid_metrics():
    """Verify scalarization is explicit and zero weights cannot hide invalid metrics."""
    result = metrics()
    torch.testing.assert_close(scalarize(result, burnup_weight=2, power_weight=3),
                               2 * result.discharge_burnup_mse + 3 * result.radial_power_variance)
    torch.testing.assert_close(scalarize(result, burnup_weight=0, power_weight=1),
                               result.radial_power_variance)
    for weights in [(0, 0), (-1, 2), (1, float("nan"))]:
        with pytest.raises(ValueError):
            scalarize(result, burnup_weight=weights[0], power_weight=weights[1])
    with pytest.raises(TypeError):
        scalarize(result)
    with pytest.raises(ValueError):
        radial_power_objective(tensor([[[1], [3]], [[0], [0]]]), [1, 0])


def test_peak_limit_is_separate_required_and_keeps_constraint_sign():
    """Verify peak limit is separate required and keeps constraint sign."""
    burnup = tensor([[80, 95], [105, 110]])
    assert peak_burnup_constraint(burnup, peak_burnup_limit=110).item() == 0
    assert peak_burnup_constraint(burnup, peak_burnup_limit=100).item() == pytest.approx(0.1)
    assert peak_burnup_constraint(burnup, peak_burnup_limit=120).item() < 0
    with pytest.raises(TypeError):
        peak_burnup_constraint(burnup)
    for bound in [None, 0, -1, float("nan")]:
        with pytest.raises(ValueError):
            peak_burnup_constraint(burnup, peak_burnup_limit=bound)


@pytest.mark.parametrize("mass,target", [([0, 1], 100), ([-1, 2], 100), ([1, 2], 0)])
def test_nonpositive_mass_or_target_rejected(mass, target):
    """Verify nonpositive mass or target rejected.

    Parameters
    ----------
    mass : sequence of float
        Candidate initial heavy-metal masses containing an invalid nonpositive value.
    target : float
        Candidate discharge burnup target, invalid when nonpositive.
    """
    with pytest.raises(ValueError):
        discharge_burnup_objective(tensor([90, 120]), mass, target)


@pytest.mark.parametrize("weights", [[0, 0], [1, -1], [1, float("nan")]])
def test_invalid_time_weights_rejected(weights):
    """Verify invalid time weights rejected.

    Parameters
    ----------
    weights : sequence of float
        Candidate time weights containing an invalid value or zero total.
    """
    with pytest.raises(ValueError):
        radial_power_objective(tensor([[[1], [3]], [[1], [3]]]), weights)


@pytest.mark.parametrize("multiplicity", [[0, 6], [1, 1.5], [1], [1, float("inf")]])
def test_invalid_multiplicities_rejected(multiplicity):
    """Verify invalid multiplicities rejected.

    Parameters
    ----------
    multiplicity : sequence of float
        Candidate physical-copy counts with invalid values or shape.
    """
    with pytest.raises(ValueError):
        discharge_burnup_objective(tensor([90, 120]), [1, 2], 100, multiplicities=multiplicity)
    with pytest.raises(ValueError):
        radial_power_objective(tensor([[[1], [3]]]), [1], channel_multiplicities=multiplicity)


def test_gradients_through_burnup_mass_target_power_time_weights_match_finite_difference():
    """Verify gradients through burnup mass target power time weights match finite difference."""
    def objective(x):
        """Perturb every continuous objective input through two coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Two live perturbation coordinates.

        Returns
        -------
        torch.Tensor
            Scalar weighted burnup/power objective for finite-difference comparison.
        """
        result = evaluate_objectives(
            discharge_burnup=torch.stack((90 + x[0], 120 - 2 * x[0])),
            discharge_initial_hm_mass=torch.stack((1 + 0.1 * x[1], 2 + 0.2 * x[0])),
            target_burnup=100 + x[1],
            bundle_power=torch.stack((1 + x[1], 2 + x[0], 3 - x[1], 2 - 0.2 * x[0])).reshape(2, 2, 1),
            time_weights=torch.stack((1 + 0.1 * x[0], 3 + x[1])),
            discharge_multiplicities=[1, 6], channel_multiplicities=[1, 6],
        )
        return scalarize(result, burnup_weight=0.7, power_weight=0.3)
    x = tensor([0.2, 0.1], requires_grad=True)
    gradient, = torch.autograd.grad(objective(x), x)
    step = 1e-5
    numerical = []
    for index in range(2):
        direction = torch.zeros_like(x)
        direction[index] = step
        numerical.append((objective(x + direction) - objective(x - direction)) / (2 * step))
    torch.testing.assert_close(gradient, torch.stack(numerical), rtol=1e-7, atol=1e-9)


def test_peak_constraint_has_gradient_away_from_ties():
    """Verify peak constraint has gradient away from ties."""
    burnup = tensor([90, 110], requires_grad=True)
    gradient, = torch.autograd.grad(peak_burnup_constraint(burnup, peak_burnup_limit=120), burnup)
    torch.testing.assert_close(gradient, tensor([0, 1 / 120]))
