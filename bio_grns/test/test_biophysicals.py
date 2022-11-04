import unittest

from bio_grns.generators.generate_biophysicals import (
    generate_decays,
    generate_tf_indices,
    generate_transcription_rates
)

import numpy as np

class TestDecayGenerator(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_good_decays(self):

        lambdas = generate_decays(
            100,
            self.rng,
            decay_limits=(0.1, 1)
        )

        self.assertTrue(
            np.all(lambdas >= 0.1)
        )

        self.assertTrue(
            np.all(lambdas <= 1.)
        )

        self.assertEqual(
            lambdas.size,
            100
        )

        lambdas = generate_decays(
            100,
            self.rng,
            halflife_limits=(15, 60)
        )

        self.assertTrue(
            np.all(lambdas >= np.log(2) / 60)
        )

        self.assertTrue(
            np.all(lambdas <= np.log(2) / 15)
        )

        self.assertEqual(
            lambdas.size,
            100
        )

    def test_bad_decays(self):

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng
            )

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                halflife_limits=(-10, 10)
            )

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                halflife_limits=(0, 10)
            )

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                halflife_limits=(100, 10)
            )

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                decay_limits=(-1, 1)
            )

        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                decay_limits=(0, 1)
            )


        with self.assertRaises(ValueError):

            lambdas = generate_decays(
                100,
                self.rng,
                decay_limits=(10, 1)
            )

            del lambdas

class TestAlphaGenerator(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_alphas(self):

        alphas = generate_transcription_rates(
            100,
            self.rng
        )

        self.assertTrue(
            np.all(alphas >= 0)
        )

        self.assertTrue(
            np.all(alphas <= 10.)
        )

        self.assertEqual(
            alphas.size,
            100
        )

class TestTFIndexGenerator(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_alphas(self):

        idx = generate_tf_indices(
            10000,
            200,
            self.rng
        )

        self.assertTrue(
            np.all(idx >= 0)
        )

        self.assertTrue(
            np.all(idx <= 10000)
        )

        self.assertEqual(
            idx.size,
            200
        )

        self.assertEqual(
            np.unique(idx).size,
            200
        )

        self.assertTrue(
            np.all(np.diff(idx) > 0),
        )
