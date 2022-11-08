import unittest
import tempfile
import os

from bio_grns.calculators.downsample_network import network_holdouts
from bio_grns import GRNSimulator

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

class TestDownsample(unittest.TestCase):

    def setUp(self) -> None:

        self.rng = np.random.default_rng(100)

        net = self.rng.uniform(
            0, 1, 200
        ).reshape(50, 4)

        net[net < 0.75] = 0

        self.testnet = pd.DataFrame(
            net
        )

        self.nnz = (self.testnet != 0).sum().sum()

    def test_no_downsample(self):

        before = self.testnet.copy()
        after = network_holdouts(
            self.testnet,
            1.0,
            self.rng
        )

        pdt.assert_frame_equal(before, after)

    def test_no_remaining(self):

        before = self.testnet.copy()
        after = network_holdouts(
            before,
            0.0,
            self.rng
        )

        self.assertEqual(
            (after != 0).sum().sum(),
            0
        )

        self.assertEqual(
            (before != 0).sum().sum(),
            self.nnz
        )

    def test_half_remaining(self):

        after = network_holdouts(
            self.testnet,
            0.5,
            self.rng
        )

        self.assertEqual(
            (after != 0).sum().sum(),
            int(self.nnz / 2)
        )

    def test_bad_ratio(self):

        with self.assertRaises(ValueError):
            after = network_holdouts(
                self.testnet,
                1.5,
                self.rng
            )

        with self.assertRaises(ValueError):
            after = network_holdouts(
                self.testnet,
                -1.5,
                self.rng
            )

            del after

class TestSimNetwork(unittest.TestCase):

    def setUp(self) -> None:

        self.sim = GRNSimulator(
            50,
            4,
            100
        )

        self.sim._generate_tf_indices()
        self.sim._generate_regulatory_network()

        self.nnz = np.sum(self.sim._reg_network != 0)

    def test_make_network_files(self):

        with tempfile.TemporaryDirectory() as td:

            fn = os.path.join(td, "test.tsv")

            self.assertFalse(
                os.path.exists(fn)
            )

            self.sim.save_network(fn)

            self.assertTrue(
                os.path.exists(fn)
            )

            npt.assert_array_almost_equal(
                pd.read_csv(fn, sep="\t", index_col=0).values,
                self.sim._reg_network
            )

    def test_make_holdout_files(self):

        with tempfile.TemporaryDirectory() as td:

            fn = os.path.join(td, "test.tsv")

            self.assertFalse(
                os.path.exists(fn)
            )

            self.sim.save_network(fn, edges_to_include=0.5)

            self.assertTrue(
                os.path.exists(fn)
            )

            infile = pd.read_csv(fn, sep="\t", index_col=0).values

            self.assertEqual(
                (infile != 0).sum().sum(),
                int(self.nnz / 2)
            )
