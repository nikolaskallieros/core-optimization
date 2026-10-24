"""Check native normalization with known cross sections and block statistics."""
from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_surrogate.native3d import tally_cross_sections
from diffusion_surrogate.reference import statistics


@pytest.mark.parametrize("different_inventory", [False, True])
def test_native_reuse_compares_case_values_and_full_inventory(tmp_path, different_inventory):
    """Reuse equal transport inputs while distinguishing inventory below the cutoff.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for a retained native-state fixture.
    different_inventory : bool
        Change a trace inventory without changing the cutoff transport description.
    """
    from coupling.native import evaluate_native
    from runtime.io import save_state

    case = {"group_edges_eV": np.array([0., 1., 2.]), "power_W": 100.}
    inventory = np.array([[1., 1e-14]])
    path = tmp_path / "point.json"
    save_state(path, {"case": case, "fuel_inventory_atoms_cm3": inventory,
                      "result": {"combined_keff": [1., .001]}})
    physical = SimpleNamespace(case=lambda *args, **kwargs: {
        "power_W": 100., "group_edges_eV": [0., 1., 2.]})
    if different_inventory:
        inventory[0, 1] = 2e-14
        with pytest.raises(ValueError, match="different full-chain inventory"):
            evaluate_native(physical, inventory, 0., {}, path, tmp_path / "work")
    else:
        state = evaluate_native(physical, inventory, 0., {}, path, tmp_path / "work")
        np.testing.assert_array_equal(state["fuel_inventory_atoms_cm3"], inventory)
        assert state["result"]["combined_keff"] == [1., .001]


@pytest.mark.parametrize("factorized", [False, True])
def test_block_statistics_preserve_fixed_power_and_covariance(factorized):
    """Verify block statistics preserve fixed power and covariance.

    Parameters
    ----------
    factorized : bool
        Return a covariance factor instead of the full statistical covariance.
    """
    # Constant total heating makes normalization linear in this known sample.
    heating = np.array([[1., 3.], [2., 2.], [3., 1.], [2., 2.]])
    eigenvalue = np.array([1., 2., 3., 2.])
    scores = np.stack((heating, np.column_stack((eigenvalue, np.zeros(4)))), axis=-1)
    result = statistics(scores, 40., 1, factorized=factorized)
    observations = np.column_stack((eigenvalue, heating * 10.))
    expected = np.cov(observations, rowvar=False) / 4
    covariance = (result["covariance_factor"].T @ result["covariance_factor"]
                  if factorized else np.array(result["covariance"]))
    np.testing.assert_allclose(result["mean"], [2., 20., 20.])
    np.testing.assert_allclose(covariance, expected, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(np.square(result["se"]), np.diag(expected))
    np.testing.assert_allclose(covariance @ [0., 1., 1.], 0., atol=1e-14)


def test_statistics_require_independent_blocks_for_an_uncertainty_estimate():
    """Verify statistics require independent blocks for an uncertainty estimate."""
    with pytest.raises(ValueError, match="at least two active blocks"):
        statistics(np.ones((4, 2, 2)), 40., 4)


def test_tallies_keep_group_order_and_use_the_matching_flux_estimator():
    """Verify tallies keep group order and use the matching flux estimator."""
    flux = np.array([[10., 20.]])
    analog = np.array([[7., 13.]])
    density = np.array([[.25, 0.]])
    reaction = np.empty((1, 2, 2, 5))
    scattering = np.empty((1, 2, 2, 2, 2))
    moment = np.zeros((1, 2, 2, 2))
    for group in range(2):
        for isotope in range(2):
            for score in range(5):
                reaction[0, group, isotope, score] = (1. + group + 3 * isotope + score) * flux[0, group]
            for outgoing in range(2):
                sigma = .2 + group + 2 * outgoing
                scattering[0, group, outgoing, isotope] = np.array([sigma, 1.5 * sigma]) * analog[0, group] * density[0, isotope]
            moment[0, group, 1, isotope] = .1 * (group + 1) * analog[0, group] * density[0, isotope]
    tallies = {3: reaction, 4: scattering, 5: moment}
    state = SimpleNamespace(get_tally=lambda *, id: SimpleNamespace(mean=tallies[id]))
    result = tally_cross_sections(state, (3, 4, 5), flux, analog, 2, atom_density=density)
    np.testing.assert_allclose(result["absorption"], [[[2., 3.], [5., 6.]]])
    np.testing.assert_allclose(result["scatter"][0, 0], [[.2, 2.2], [1.2, 3.2]])
    np.testing.assert_allclose(result["nu_scatter"], 1.5 * result["scatter"])
    np.testing.assert_allclose(result["scatter_p1"], [[[.1, .2], [0., 0.]]])
    assert not result["scatter"][0, 1].any()
    assert np.isfinite(result["scatter"]).all()
