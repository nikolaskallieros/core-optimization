"""Pure-BU and hard mass/EOL gates against the integrated optimizer; no solves."""
from runtime.paths import PROJECT_ROOT
from designs.wr_1 import WR1_ROOT
from copy import deepcopy
import sys
import unittest

import torch

ROOT = WR1_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from coupling import optimization
from designs.wr_1.model.case import study
from optimization.fuel_constraints import fuel_mass_check


class BurnupMassOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load scenario-six cycle and fuel-mass optimization settings."""
        workflow = study(6)
        cls.options = workflow.settings("optimization")
        cls.settings = workflow.settings()

    def prediction(self):
        """Build a low-burnup prediction that isolates fuel-mass policy.

        Returns
        -------
        dict
            Synthetic predicted objective, EOL and mass diagnostics.
        """
        return {"objective": -1., "metrics": {"discharge_burnup_MWd_tHM": 1.}, "minimum_keff": 1.002,
                "eol_corrected_keff": 1.002, "eol_standard_error": .003,
                "peak_constraint_passed": True,
                "fresh_fuel_mass_check": fuel_mass_check(self.settings["design"], self.options["fresh_fuel_mass"])}

    def native(self):
        """Build synthetic native fields independent of transport noise.

        Returns
        -------
        dict
            Prescribed low-burnup, high-power-variance acceptance-policy fixture.
        """
        # Prescribed values isolate acceptance policy from transport and sampling.
        return {"design": deepcopy(self.settings["design"]), "feasible": True, "physics_passed": True,
                "eol_keff_by_cycle": {"3": [1.002, .003]}, "merit": -1./5000.,
                "metrics": {"power_variance": 1e6, "discharge_burnup_MWd_tHM": 1.}}

    def test_only_burnup_has_objective_weight(self):
        """Verify only burnup has objective weight."""
        burnup = torch.tensor(12345., dtype=torch.float64, requires_grad=True)
        power = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        score = optimization.merit(burnup, power, self.settings["objective"], self.options)
        gradients = torch.autograd.grad(score, (burnup, power))
        self.assertEqual(float(gradients[0]), -1./5000.)
        self.assertEqual(float(gradients[1]), 0.)

    def test_absent_burnup_floor_and_power_limits(self):
        """Verify absent burnup floor and power limits."""
        self.assertNotIn("minimum_discharge_burnup_MWd_tHM", self.options)
        self.assertEqual(optimization.prediction_rank(self.prediction(), self.options), (0, -1.))
        value = self.native()
        optimization.assess_native(value, self.options)
        self.assertTrue(value["acceptable"])

    def test_native_burnup_floor_is_explicit_and_legacy_default_is_preserved(self):
        """Verify native burnup floor is explicit and legacy default is preserved."""
        # A negative sentinel isolates the policy gate; physical acceptance stays false.
        value = self.native()
        value["metrics"]["discharge_burnup_MWd_tHM"] = -1.
        value["physics_passed"] = False
        optimization.assess_native(value, self.options)
        self.assertTrue(value["burnup_passed"])
        self.assertFalse(value["acceptable"])
        legacy = deepcopy(self.options)
        legacy.pop("fresh_fuel_mass")
        optimization.assess_native(value, legacy)
        self.assertFalse(value["burnup_passed"])
        explicit = {**self.options, "minimum_discharge_burnup_MWd_tHM": 2.}
        value = self.native()
        optimization.assess_native(value, explicit)
        self.assertFalse(value["burnup_passed"])
        self.assertFalse(value["acceptable"])

    def test_predicted_mass_and_nonlinear_eol_are_both_required(self):
        """Verify predicted mass and nonlinear EOL are both required."""
        value = self.prediction()
        value["fresh_fuel_mass_check"]["passed"] = False
        self.assertEqual(optimization.prediction_rank(value, self.options)[0], 1)
        value = self.prediction()
        value["eol_corrected_keff"] = 1.12
        self.assertEqual(optimization.prediction_rank(value, self.options)[0], 1)

    def test_native_mass_and_eol_are_both_required(self):
        """Verify native mass and EOL are both required."""
        value = self.native()
        value["design"]["element_radius_mm"][0] *= 1.01
        optimization.assess_native(value, self.options)
        self.assertFalse(value["acceptable"])
        self.assertFalse(value["fuel_mass_passed"])
        value = self.native()
        value["eol_keff_by_cycle"]["3"] = [1.12, .003]
        optimization.assess_native(value, self.options)
        self.assertFalse(value["acceptable"])


if __name__ == "__main__":
    unittest.main()
