import unittest

from bio_grns.generators.generate_dynamics import (
    _random_walk,
    _cyclic_walk,
    _monotonic_walk,
    _updown_walk,
    dynamic_value_generator
)
import numpy as np
import numpy.testing as npt


class TestDynamics(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_cyclic_walk_down_up(self):
        dyn_vals = _cyclic_walk(
            500,
            0.5,
            0.1,
            0.5,
            self.rng
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[250:]) > 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[0:250]) < 0)
        )

        npt.assert_almost_equal(
            dyn_vals[249],
            0.
        )

        npt.assert_almost_equal(
            dyn_vals[-1],
            1.
        )

    def test_cyclic_walk_up_down(self):
        dyn_vals = _cyclic_walk(
            500,
            0.,
            0.1,
            0.5,
            self.rng
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[250:]) < 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[0:250]) > 0)
        )

        npt.assert_almost_equal(
            dyn_vals[249],
            1.
        )

        npt.assert_almost_equal(
            dyn_vals[-1],
            0.
        )

    def test_random_walk(self):
        dyn_vals = _random_walk(
            500,
            1,
            0.001,
            0.1,
            self.rng
        )

        self.assertTrue(
            np.all(dyn_vals > 0)
        )

    def test_monotonic_walk_increasing(self):
        dyn_vals = _monotonic_walk(
            500,
            0.1,
            0.0,
            0.005,
            self.rng
        )

        print(dyn_vals)

        self.assertTrue(
            np.all(dyn_vals > 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals) >= 0)
        )

    def test_monotonic_walk_decreasing(self):
        dyn_vals = _monotonic_walk(
            500,
            2.5,
            0.0,
            -0.005,
            self.rng
        )

        self.assertTrue(
            np.all(dyn_vals > 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals) <= 0)
        )

    def test_full_stack(self):
        dyn_vals = dynamic_value_generator(
            3,
            500,
            [
                ('random', 1, 0.05, 0.2),
                ('cyclic', 0, 0.1, 0.5),
                ('monotonic', 0.01, 0.01, 0.075)
            ],
            self.rng,
            force_positive=True
        )

        self.assertTrue(
            np.all(dyn_vals >= 0)
        )

        self.assertTrue(
            dyn_vals[0, 0] == 1.
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[250:, 1]) < 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[0:250, 1]) > 0)
        )

        npt.assert_almost_equal(
            dyn_vals[249, 1],
            1.
        )

        npt.assert_almost_equal(
            dyn_vals[-1, 1],
            0.
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[:, 2]) >= 0)
        )

    def test_updown_walk(self):

        dyn_vals = _updown_walk(
            500,
            .5,
            .5,
            (0.1, 1),
            self.rng
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[250:]) < 0)
        )

        self.assertTrue(
            np.all(np.diff(dyn_vals[0:250]) > 0)
        )

        npt.assert_almost_equal(
            dyn_vals[249],
            1.
        )

        npt.assert_almost_equal(
            dyn_vals[-1],
            0.
        )

        npt.assert_almost_equal(
            dyn_vals[0],
            0.5
        )
