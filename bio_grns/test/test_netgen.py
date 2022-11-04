import unittest

from bio_grns.generators.generate_network import network_generator
import numpy as np

class TestNetworkGenerator(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_all_positive(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=1.
        )

        self.assertTrue(
            np.all(net >= 0.)
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertEqual(
            np.max(np.sum(net != 0, axis=1)),
            1
        )

    def test_positive_ratios(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=0.5
        )

        self.assertFalse(
            np.all(net >= 0.)
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertGreaterEqual(
            np.sum(net > 0.),
            25
        )

        self.assertGreaterEqual(
            np.sum(net < 0.),
            25
        )

        self.assertEqual(
            np.max(np.sum(net != 0, axis=1)),
            1
        )

    def test_all_negative(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=0.
        )

        self.assertTrue(
            np.all(net <= 0.)
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertEqual(
            np.max(np.sum(net != 0, axis=1)),
            1
        )

    def test_abs_limits_min(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=0.5,
            minimum_abs_value=0.5
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertEqual(
            np.sum(np.abs(net) >= 0.5),
            60
        )

        self.assertEqual(
            np.max(np.sum(net != 0, axis=1)),
            1
        )

    def test_abs_limits_max(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=0.5,
            minimum_abs_value=1,
            maximum_abs_value=100
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertEqual(
            np.sum(np.abs(net) >= 1),
            60
        )

        self.assertGreater(
            np.max(np.abs(net)),
            75
        )

        self.assertEqual(
            np.max(np.sum(net != 0, axis=1)),
            1
        )

    def test_release_row_restriction(self):

        net = network_generator(
            3,
            200,
            [0.1, 0.15, 0.25],
            0.1,
            self.rng,
            positive_ratio=0.5,
            minimum_abs_value=1,
            maximum_abs_value=100,
            row_one_entry=False
        )

        self.assertEqual(
            np.sum(net != 0.),
            60
        )

        self.assertEqual(
            np.sum(np.abs(net) >= 1),
            60
        )

        self.assertGreater(
            np.max(np.abs(net)),
            75
        )

        self.assertGreaterEqual(
            np.max(np.sum(net != 0, axis=1)),
            2
        )

    def test_argument_checking(self):

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, 0.25, 1000],
                0.1,
                self.rng,
                positive_ratio=1.
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                10,
                0.1,
                self.rng,
                positive_ratio=1.
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, 0.25],
                0.1,
                self.rng,
                positive_ratio=100.
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, 0.25],
                0.1,
                self.rng,
                positive_ratio=-10.
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, -0.25],
                0.1,
                self.rng
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, -0.25],
                0.1,
                self.rng,
                minimum_abs_value=10,
                maximum_abs_value=1
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, -0.25],
                0.1,
                self.rng,
                minimum_abs_value=-10,
                maximum_abs_value=1
            )

        with self.assertRaises(ValueError):
            net = network_generator(
                3,
                200,
                [0.1, 0.15, -0.25],
                0.1,
                self.rng,
                minimum_abs_value=0,
                maximum_abs_value=-1
            )

            del net
