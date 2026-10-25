from examples.wr_1 import WR1_ROOT
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fuel_cycle.residence import FuelState, FuelMap, schedule, cecm, DAY

ROOT = WR1_ROOT
FIXTURES = Path(__file__).resolve().parent / "fixtures/wr1"
DTYPE = torch.float64


def fixture():
    """Build three represented channels with six tracked axial positions.

    Returns
    -------
    tuple
        FuelMap and manufactured two-nuclide FuelState with distinct identities.
    """
    labels = [f"{c}/axial/{j}" for c in ("center", "lower", "upper") for j in range(6)]
    physical = SimpleNamespace(labels=labels, symmetry_multiplier=6,
                               multiplicity=np.repeat([1, 3, 3], 6),
                               regions=[{"label": label, "ring": 0 if label.startswith("center") else 1}
                                        for label in labels])
    mapping = FuelMap(physical)
    state = FuelState(torch.arange(36, dtype=DTYPE).reshape(18, 2),
                      torch.arange(1, 19, dtype=DTYPE), mapping.fractions.clone(),
                      tuple(labels))
    return mapping, state


def test_sector_volumes_and_physical_multiplicity():
    """Verify sector volumes and physical multiplicity."""
    from fixtures.small_case import SmallDesign, make_case
    from diffusion_surrogate.hexagonal import C6Grid
    from geometry.diffusion import material_fractions
    from examples.wr_1.build_geometry import parameterized_geometry
    design, case = SmallDesign(ROOT), make_case(ROOT)
    grid = C6Grid(case["geometry"]["apothem_cm"], 8, design.z_edges_mm.numpy() / 10,
                  symmetry_boundary="reflective")
    actual = (material_fractions(parameterized_geometry(design, case), grid).T @ grid.volumes).detach().numpy()
    np.testing.assert_allclose(actual, [r["volume_cm3"] for r in case["regions"]], rtol=1e-11)
    fuel = [r for r in case["regions"] if r["depletable"]]
    assert len(fuel) == 9 and sum(r["physical_multiplicity"] for r in fuel) == 21
    assert len(case["physical_assembly_to_region"]) == 21
    np.testing.assert_allclose(design.z_edges_mm, [0., 990., 1980., 2970.])
    assert case["power_W"] == pytest.approx(1324324.3243243243)
    assert not grid.periodic_pairs
    assert grid.boundaries[0] == grid.boundaries[4] == grid.boundaries[5] == "vacuum"


def test_ring_transfer_preserves_other_ring_energy_mass_and_ids():
    """Verify ring transfer preserves other ring energy mass and ids."""
    mapping, state = fixture()
    fresh = torch.full_like(state.inventory, .5, requires_grad=True)
    moved, discharged = mapping.shuffle(state, (0,), fresh, state.initial_hm_kg, "new")
    torch.testing.assert_close(moved.inventory[6:], state.inventory[6:])
    torch.testing.assert_close(moved.inventory[1:6], state.inventory[:5])
    torch.testing.assert_close(moved.inventory[0], fresh[0])
    assert moved.ids[1:6] == state.ids[:5]
    assert discharged["ids"] == (state.ids[5],)
    assert moved.energy_j.sum() + discharged["energy_j"].sum() == state.energy_j.sum()
    assert moved.initial_hm_kg.sum() == state.initial_hm_kg.sum()
    moved.inventory.sum().backward()
    assert torch.count_nonzero(fresh.grad) == 2


def test_all_rings_load_together_and_conserve_carried_energy_and_mass():
    """Verify all rings load together and conserve carried energy and mass."""
    mapping, state = fixture()
    fresh = torch.full_like(state.inventory, .5, requires_grad=True)
    moved, discharged = mapping.shuffle(state, (0, 1), fresh, state.initial_hm_kg, "common/BOL")
    assert discharged["ids"] == tuple(state.ids[i] for i in (5, 11, 17))
    for start in (0, 6, 12):
        torch.testing.assert_close(moved.inventory[start], fresh[start])
        torch.testing.assert_close(moved.inventory[start + 1:start + 6], state.inventory[start:start + 5])
        assert moved.ids[start + 1:start + 6] == state.ids[start:start + 5]
    torch.testing.assert_close(moved.energy_j.sum() + discharged["energy_j"].sum(), state.energy_j.sum())
    torch.testing.assert_close(moved.initial_hm_kg.sum(), state.initial_hm_kg.sum())
    moved.inventory.sum().backward()
    assert torch.count_nonzero(fresh.grad) == 6


def test_common_period_derivative_passes_through_every_depletion_step():
    """Verify common period derivative passes through every depletion step."""
    period = torch.tensor([80.], dtype=DTYPE, requires_grad=True)
    events = schedule(period, [0., .0125, .125, .375, .6875, 1.])
    np.testing.assert_allclose([float(t.detach()) for t, _ in events], [1., 10., 30., 55., 80.])
    state = FuelState(torch.tensor([[2.]], dtype=DTYPE), torch.zeros(1, dtype=DTYPE),
                      torch.ones(1, dtype=DTYPE), ("fuel",))
    rate = .004
    output = {"rates": torch.tensor([[[rate / DAY]]], dtype=DTYPE), "zone_power_w": torch.ones(1, dtype=DTYPE)}
    decay = lambda n, rates, seconds: n * torch.exp(-rates[..., 0] * seconds)
    previous = 0.
    for end, _ in events:
        state, _, _ = cecm(state, end - previous, output, lambda n: output, decay)
        previous = end
    expected = 2 * torch.exp(-rate * period)
    torch.testing.assert_close(state.inventory.reshape(-1), expected)
    gradient = torch.autograd.grad(state.inventory.sum(), period)[0]
    torch.testing.assert_close(gradient, -rate * expected)
    torch.testing.assert_close(state.energy_j, period * DAY)
    for invalid in (0., -1., float("nan"), [80., 90.]):
        with pytest.raises(ValueError, match="one finite positive"):
            schedule(invalid, [0., 1.])


def test_cecm_differentiates_bateman_and_passes_depleted_composition_to_next_diffusion():
    """Verify CE/CM differentiates bateman and passes depleted composition to next diffusion."""
    x = torch.tensor(.3, dtype=DTYPE, requires_grad=True)
    dt = torch.tensor(.4, dtype=DTYPE, requires_grad=True)
    state = FuelState(torch.tensor([[2.]], dtype=DTYPE), torch.zeros(1, dtype=DTYPE),
                      torch.ones(1, dtype=DTYPE), ("fuel",))
    seen = []

    def diffusion(n):
        """Record the live inventory and return manufactured reaction rates.

        Parameters
        ----------
        n : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.

        Returns
        -------
        dict
            Composition-dependent rates and one-watt zone power.
        """
        seen.append(n)
        return {"rates": (x * n / DAY)[..., None], "zone_power_w": torch.ones(1, dtype=DTYPE)}

    def bateman(n, rates, seconds):
        """Propagate the manufactured uncoupled exponential depletion law.

        Parameters
        ----------
        n : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        rates : torch.Tensor
            First-reaction removal frequencies in s⁻¹.
        seconds : torch.Tensor
            Propagation interval in seconds.

        Returns
        -------
        torch.Tensor
            Depleted inventory retaining rate and duration derivatives.
        """
        return n * torch.exp(-rates[..., 0] * seconds)

    start = diffusion(state.inventory)
    final, half, _ = cecm(state, dt, start, diffusion, bateman)
    expected_half = 2 * torch.exp(-x * 2 * dt / 2)
    expected_final = 2 * torch.exp(-x * expected_half * dt)
    torch.testing.assert_close(final.inventory.squeeze(), expected_final)
    assert seen[1] is half and half.requires_grad
    got = torch.autograd.grad(final.inventory.sum(), (x, dt), retain_graph=True)
    want = torch.autograd.grad(expected_final, (x, dt))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b)


def test_zero_boron_test_never_calls_a_critical_boron_solver():
    """Verify zero boron test never calls a critical boron solver."""
    from fuel_cycle.residence import local_output
    calls = []
    point = SimpleNamespace(reference_boron=0., evaluate=lambda n, b: calls.append(float(b)) or {"keff": n.sum()})
    local_output(point, torch.ones(2, dtype=DTYPE))
    assert calls == [0.]


def test_equilibrium_compares_positions_and_equal_phases_not_fuel_ids():
    """Verify equilibrium compares positions and equal phases not fuel ids."""
    from copy import deepcopy
    from coupling.history import compare_cycles
    old = {"cycle": 2, "cycle_days": 80., "bol": "old_bol", "segments": [{"mid": "old_mid", "end": "old_end"}],
           "end_state": {"inventory": np.full((2, 2), 1e20), "energy_j": np.array([2e9, 3e9]),
                         "initial_hm_kg": np.ones(2), "ids": ["old/a", "old/b"]}}
    new = deepcopy(old)
    new.update(cycle=3, bol="new_bol", segments=[{"mid": "new_mid", "end": "new_end"}])
    new["end_state"]["ids"] = ["new/a", "new/b"]
    observations = {f"{cycle}_{stage}": {"power_W": [2., 3.], "keff": [1.01, .01]}
                    for cycle in ("old", "new") for stage in ("bol", "mid", "end")}
    result = compare_cycles(old, new, observations)
    assert result["inventory_relative_rms"] == result["burnup_relative_rms"] == 0.
    assert result["assembly_power_history_relative_rms"] == 0.
    new["end_state"]["inventory"] *= 1.1
    observations["new_mid"]["power_W"] = [2.2, 2.8]
    result = compare_cycles(old, new, observations)
    assert result["inventory_relative_rms"] == pytest.approx(1 / 11)
    assert result["assembly_power_history_relative_rms"] > 0.
    new["cycle_days"] = 90.
    with pytest.raises(ValueError, match="same refuelling period"):
        compare_cycles(old, new, observations)


def test_shared_coefficient_roundtrip_preserves_fitted_values_near_bounds():
    """Verify shared coefficient roundtrip preserves fitted values near bounds."""
    from coupling.calibration import SharedCoefficients
    values = torch.tensor([-2.29999, -1., .2, 2.29999], dtype=DTYPE)
    coefficients = SharedCoefficients(len(values), 2.3)
    coefficients.initialize(values)
    torch.testing.assert_close(coefficients(), values, atol=1e-14, rtol=1e-14)


def test_small_shared_spatial_bases_keep_the_two_outer_half_channels_equivalent():
    """Verify small shared spatial bases keep the two outer half channels equivalent."""
    from fixtures.small_case import make_case
    from examples.wr_1.model.calibration import equivalence_bases
    case = make_case(ROOT)
    regions = case["regions"]
    lower = next(i for i, r in enumerate(regions) if r["label"] == "outer_lower/axial/2")
    upper = next(i for i, r in enumerate(regions) if r["label"] == "outer_upper/axial/2")
    centre = next(i for i, r in enumerate(regions) if r["label"] == "center/axial/2")
    void = torch.tensor([r["void"] for r in regions])
    for layout, count in (("material", 8), ("material_axial", 12), ("ring_material", 14),
                           ("ring_material_axial", 18), ("fuel_levels", 18), ("region", 60)):
        names, bases, equivalence = equivalence_bases(case, layout=layout, include_void=False)
        assert len(names) == count
        assert equivalence.shape == (count, len(regions), 2)
        if layout != "region":
            torch.testing.assert_close(equivalence[:, lower], equivalence[:, upper])
        assert torch.count_nonzero(equivalence[:, void]) == 0
        physical_support = (bases["diffusion"][:, ~void].abs() + equivalence[:, ~void].abs()).sum((1, 2))
        assert (physical_support > 0).all()
        torch.testing.assert_close(bases["diffusion"][0, :, 0], torch.ones(len(regions), dtype=DTYPE))
        if layout == "fuel_levels":
            assert not torch.equal(equivalence[:, lower], equivalence[:, centre])
    # Keep the original WR-1 basis available for its retained historical fits.
    assert len(equivalence_bases(case, layout="material")[0]) == 10


def test_basis_transfer_preserves_diffusion_solution_and_rejects_incompatible_fields():
    """Verify basis transfer preserves diffusion solution and rejects incompatible fields."""
    from fixtures.small_case import SmallDesign, definition
    from coupling.calibration import transfer_coefficients
    from examples.wr_1.model.calibration import make_surrogate as WR1Surrogate, equivalence_bases
    from geometry.diffusion import discretize
    from examples.wr_1.build_geometry import parameterized_geometry
    from runtime.io import load_state
    state = load_state(FIXTURES / "small_cycle_states/fresh.json")
    case = state["case"]
    values = torch.linspace(-.15, .2, 8, dtype=DTYPE)

    def solve(layout, factors):
        """Solve a retained small case with a selected correction basis.

        Parameters
        ----------
        layout : str
            Spatial equivalence-basis layout.
        factors : torch.Tensor
            Log correction coefficients in that basis.

        Returns
        -------
        dict
            Diffusion outputs evaluated without retaining a gradient graph.
        """
        design = SmallDesign(ROOT)
        model = WR1Surrogate(definition(ROOT), design, case, state["result"], mesh=discretize(lambda: parameterized_geometry(design, case), subdivisions=8),
                             equivalence_layout=layout, equivalence_void_flux=False)
        with torch.no_grad():
            model.log_factors.copy_(factors)
            return model()

    reference = solve("material", values)
    for layout in ("material_axial", "ring_material", "ring_material_axial", "fuel_levels", "region"):
        projected = transfer_coefficients(case, "material", values, layout, 2.3, basis_factory=equivalence_bases)
        assert projected is not None
        output = solve(layout, projected)
        for key in ("keff", "fuel_power_W", "region_flux_per_source", "leakage_per_source"):
            torch.testing.assert_close(output[key], reference[key], rtol=1e-10, atol=1e-10)
    names = equivalence_bases(case, layout="region", include_void=False)[0]
    local = torch.zeros(len(names), dtype=DTYPE)
    local[names.index("equivalence:outer_upper/axial/2:g0")] = .2
    assert transfer_coefficients(case, "region", local, "material", 2.3, basis_factory=equivalence_bases) is None
    assert transfer_coefficients(case, "material", values, "region", .1, basis_factory=equivalence_bases) is None


def test_three_cycle_residence_preserves_earlier_depletion_and_loading_derivatives():
    """Verify three cycle residence preserves earlier depletion and loading derivatives."""
    from fuel_cycle.residence import replay_history
    labels = [f"{c}/axial/{j}" for c in ("center", "lower", "upper") for j in range(3)]
    design = SimpleNamespace(cycle_days=torch.tensor([4.], dtype=DTYPE, requires_grad=True),
                             enrichment=torch.tensor([.2], dtype=DTYPE, requires_grad=True),
                             element_radius_mm=torch.tensor([1., 1.], dtype=DTYPE))
    physical = SimpleNamespace(labels=labels, symmetry_multiplier=6,
        multiplicity=np.repeat([1, 3, 3], 3), design_values={"element_radius_mm": [1., 1.]},
        regions=[{"label": name, "ring": 0 if name.startswith("center") else 1} for name in labels],
        live_recipe=lambda d: ((2. + d.enrichment).expand(9, 1), torch.ones(9, dtype=DTYPE), torch.ones(9, dtype=DTYPE)))
    mapping = FuelMap(physical)
    config = {"initial_inventory": "fresh", "objective_cycle": 3, "differentiated_cycles": [1, 2, 3],
              "step_fractions": [0., .2, .4, .6, .8, 1.], "refueling": {"rings": [0, 1]},
              "objective": {"reference_discharge_burnup_MWd_tHM": 7500., "peak_burnup_MWd_tHM": 10800.,
                            "assembly_power_weight": 1., "keff_deficit_scale": .01}}
    cycles, points = [], {}
    for cycle in (1, 2, 3):
        start = f"cycle{cycle}/bol"
        record = {"cycle": cycle, "bol": start, "segments": []}
        for i in range(5):
            middle, end = f"cycle{cycle}/s{i}_mid", f"cycle{cycle}/s{i}_end"
            record["segments"].append({"start": start, "mid": middle, "end": end,
                                        "end_key": f"step/{i}", "reload_rings": []})
            for key in (start, middle, end):
                points[key] = SimpleNamespace(reference_boron=0., evaluate=lambda n, b: {
                    "rates": torch.full((9, 1, 1), .1 / DAY, dtype=DTYPE),
                    "zone_power_w": torch.ones(9, dtype=DTYPE), "keff": torch.tensor(1.1, dtype=DTYPE),
                    "boron_ppm": b})
            start = end
        cycles.append(record)
    chain = lambda n, rates, seconds: n * torch.exp(-rates[..., 0] * seconds)
    _, final, _, _ = replay_history(mapping, chain, design, {"cycles": cycles}, points, config)
    bottom = final.inventory[2, 0]
    expected = (2. + design.enrichment[0]) * torch.exp(-.1 * 3 * design.cycle_days[0])
    torch.testing.assert_close(bottom, expected)
    actual = torch.autograd.grad(bottom, (design.enrichment, design.cycle_days), retain_graph=True)
    wanted = torch.autograd.grad(expected, (design.enrichment, design.cycle_days))
    for got, want in zip(actual, wanted):
        torch.testing.assert_close(got, want)
    assert final.ids[2] == "cycle1/center/axial/0"
    torch.testing.assert_close(final.energy_j[2], 3 * design.cycle_days[0] * DAY)
