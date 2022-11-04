import numpy as np
import pandas as pd

from .trajectories import Trajectory

class GRNSimulator:

    _trajectories = None
    _rng = None

    n_genes = None
    n_tfs = None
    n_samples = None

    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        n_samples: int,
        random_seed: int
    ) -> None:

        self._trajectories = []
        self._rng = np.random.default_rng(random_seed)

        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.n_samples = n_samples

    def add_trajectory(
        self,
        trajectory: Trajectory
    ) -> None:

        self._trajectories.append(trajectory)
