import unittest

from bio_grns.generators import count_generator
import numpy as np

class TestNetworkGenerator(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        return super().setUp()

    def test_countgen_equal_prob(self):

        counts = count_generator(
            np.ones(100, dtype=float) / 100,
            1000,
            self.rng
        )

        self.assertEqual(
            np.sum(counts),
            1000
        )

        self.assertEqual(
            counts.size,
            100
        )

    def test_countgen_unequal_prob(self):

        p = np.zeros(100, dtype=float)
        p[0] = 1.

        counts = count_generator(
            p,
            1000,
            self.rng
        )

        self.assertEqual(
            np.sum(counts),
            1000
        )

        self.assertEqual(
            counts.size,
            100
        )

        self.assertEqual(
            counts[0],
            1000
        )

        self.assertEqual(
            np.sum(counts[1:]),
            0
        )
