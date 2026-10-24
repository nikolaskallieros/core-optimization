"""Analytical checks of geometry-independent bounded quadratic projections."""
import numpy as np

from optimization.solvers import project_quadratic_bound


def test_signed_projection_onto_circle():
    """Recover the radial projection of a signed point onto a circle."""
    projected = project_quadratic_bound([-3., 4.], [-10., -10.], [10., 10.], [1., 1.], 4.)
    np.testing.assert_allclose(projected, [-1.2, 1.6], rtol=1e-14)


def test_signed_projection_with_active_box_face():
    """Recover the intersection of a circle with a binding negative coordinate."""
    projected = project_quadratic_bound([-4., 4.], [-3., -10.], [-1.5, 10.], [1., 1.], 4.)
    np.testing.assert_allclose(projected, [-1.5, np.sqrt(1.75)], rtol=1e-14)
