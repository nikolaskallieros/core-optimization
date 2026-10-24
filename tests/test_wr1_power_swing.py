"""Physical power-range and exact autograd checks; no finite differences."""
import unittest

import torch

from optimization.objectives import temporal_power
from optimization.objectives import joint_merit

DTYPE = torch.float64


class PowerSwingTests(unittest.TestCase):
    def test_full_sampled_range_and_absolute_units(self):
        """Verify full sampled range and absolute units."""
        # The interior peak matters; a high, constant channel has no swing.
        times = [0., .01, 1.]
        power = [[1e6, 3e6], [1.3e6, 3e6], [1.2e6, 3e6]]
        value = temporal_power(times, power, [1., 6.])
        self.assertAlmostEqual(float(value["maximum_power_swing_W"])/1e6, .3)
        self.assertEqual(int(value["limiting_channel_index"]), 0)
        self.assertEqual(value["maximum_time_days"].tolist(), [.01, 0.])
        swapped = temporal_power(times, power, [6., 1.])
        self.assertEqual(float(value["maximum_power_swing_W"]), float(swapped["maximum_power_swing_W"]))

    def test_limiting_position_switches_and_derivative_follows(self):
        """Verify limiting position switches and derivative follows."""
        for first_swing, limiting in ((.2, 0), (.1, 1)):
            power = torch.tensor([[1e6, 2e6], [(1+first_swing)*1e6, 2.15e6]],
                                 dtype=DTYPE, requires_grad=True)
            value = temporal_power([0., 1.], power, [6., 1.])
            self.assertEqual(int(value["limiting_channel_index"]), limiting)
            gradient, = torch.autograd.grad(value["maximum_power_swing_W"] / 1e6, power)
            expected = torch.zeros_like(power)
            expected[0, limiting], expected[1, limiting] = -1e-6, 1e-6
            torch.testing.assert_close(gradient, expected, rtol=0., atol=1e-15)

    def test_tied_limiting_positions_have_balanced_subgradient(self):
        """Verify tied limiting positions have balanced subgradient."""
        power = torch.tensor([[1e6, 2e6], [1.2e6, 2.2e6]], dtype=DTYPE, requires_grad=True)
        swing = temporal_power([0., 1.], power, [6., 1.])["maximum_power_swing_W"]
        gradient, = torch.autograd.grad(swing, power)
        torch.testing.assert_close(gradient, torch.tensor([[-.5, -.5], [.5, .5]], dtype=DTYPE))

    def test_joint_objective_uses_swing_not_rms_and_has_exact_derivatives(self):
        """Verify joint objective uses swing not rms and has exact derivatives."""
        bu = torch.tensor(25000., dtype=DTYPE, requires_grad=True)
        variance = torch.tensor(.0009, dtype=DTYPE, requires_grad=True)
        swing = torch.tensor(.2, dtype=DTYPE, requires_grad=True)
        bol = torch.tensor(1.2, dtype=DTYPE, requires_grad=True)
        reference = {"burnup_MWd_tHM": 25000., "maximum_channel_power_swing_MW": .2,
                     "power_metric": "maximum_channel_power_swing_MW", "bol_reactivity": 1-1/1.2,
                     "weights": {"burnup": 1., "maximum_channel_power_swing_MW": 1., "bol_reactivity": 1.}}
        score = joint_merit(bu, variance, bol, reference, power_swing_MW=swing)
        self.assertAlmostEqual(float(score.detach()), 0.)
        gradients = torch.autograd.grad(score, (bu, variance, swing, bol), allow_unused=True)
        self.assertAlmostEqual(float(gradients[0]), -1/25000.)
        self.assertIsNone(gradients[1])
        self.assertAlmostEqual(float(gradients[2]), 5.)
        self.assertAlmostEqual(float(gradients[3]), 1/(1.2**2*reference["bol_reactivity"]))


if __name__ == "__main__":
    unittest.main()
