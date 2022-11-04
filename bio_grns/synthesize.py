import numpy as np
import anndata as ad
import pandas as pd
import tqdm

from .trajectories import Trajectory
from .generators import (
    network_generator,
    generate_transcription_rates,
    generate_decays,
    generate_tf_indices,
    count_generator
)
from .utils import logger


class GRNSimulator:

    _trajectories = None
    _rng = None

    n_genes = None
    n_tfs = None
    n_samples = None
    n_time_steps = None
    counts_per_sample = None

    _latent_networks = None
    _transcriptional_output = None
    _decay_constants = None

    ### Simulation Parameters ###
    _halflife_limits = (10, 100)
    _reg_network_sparsity = 0.03
    _reg_network_positive = 0.8

    _data = None
    _reg_network = None

    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        n_samples: int,
        counts_per_sample: int,
        random_seed: int,
        debug: bool = False
    ) -> None:

        self._trajectories = []
        self._latent_networks = []

        self._rng = np.random.default_rng(random_seed)

        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.n_samples = n_samples
        self.counts_per_sample = counts_per_sample

        if debug:
            logger.setLevel('DEBUG')
        else:
            logger.setLevel('INFO')

    def add_trajectory(
        self,
        trajectory: Trajectory
    ) -> None:

        self._trajectories.append(trajectory)

    def simulate(self) -> None:

        self._tf_indices = generate_tf_indices(
            self.n_genes,
            self.n_tfs,
            self._rng
        )

        self._generate_latent_networks()
        self._generate_biophysical_params()
        self._generate_regulatory_network()
        self._generate_dynamic_expression()

    def set_network_parameters(
        self,
        regulatory_network_sparsity=None,
        regulatory_network_activator_ratio=None
    ) -> None:

        if regulatory_network_sparsity is not None:
            self._reg_network_sparsity = regulatory_network_sparsity

        if regulatory_network_activator_ratio is not None:
            self._reg_network_positive = regulatory_network_activator_ratio

    def set_biophysical_parameters(
        self,
        halflife_limits=None
    ) -> None:

        if halflife_limits is not None:
            self._halflife_limits = halflife_limits

    def _generate_latent_networks(self):

        logger.info(
            "Generating trajectory -> regulator networks"
        )

        for i, traj in enumerate(self._trajectories):

            if traj.rng is None:
                traj.rng = self._rng

            if traj.name is None:
                traj.name = f"Trajectory{i}"

            _traj_network = network_generator(
                traj.n_patterns,
                self.n_tfs,
                traj.target_ratios,
                traj.latent_network_sparsity / traj.n_patterns,
                self._rng,
                row_one_entry=False
            )

            traj.set_network(_traj_network)

            logger.debug(
                f"Generated network for {traj.name}"
            )

    def _generate_biophysical_params(self):

        logger.info(
            "Simulating gene biophysical parameters"
        )

        self._transcriptional_output = generate_transcription_rates(
            self.n_genes,
            self._rng
        )

        self._decay_constants = generate_decays(
            self.n_genes,
            self._rng,
            halflife_limits=self._halflife_limits
        )

    def _generate_regulatory_network(self):

        logger.info(
            "Simulating regulatory network"
        )

        # Randomly generate target ratios
        # So some TFs have many targets and some have few targets
        _regulator_target_ratios = self._rng.negative_binomial(
            1, .1, self.n_tfs
        ).astype(float) * self._rng.uniform(0, 1, self.n_tfs)

        self._reg_network = network_generator(
            self.n_tfs,
            self.n_genes,
            _regulator_target_ratios,
            self._reg_network_sparsity,
            self._rng,
            positive_ratio=0.8,
            row_one_entry=False
        )

    def _generate_dynamic_expression(self):

        logger.info(
            "Simulating gene expression over time"
        )

        for traj in self._trajectories:

            initial_vector = np.maximum(
                0,
                self._rng.normal(
                    100,
                    100,
                    self.n_genes
                )
            )

            # Zero out the initial vector for targets
            # That are not in this trajectory
            if not traj.primary_trajectory:

                _relevant_genes = np.any(
                    traj._latent_network != 0,
                    axis=1
                )

                _relevant_genes = np.any(
                    self._reg_network[:, _relevant_genes] != 0,
                    axis=1
                )

                initial_vector[~_relevant_genes] = 0.

            traj.calculate_expression(
                self._reg_network,
                self._decay_constants,
                self._transcriptional_output,
                initial_vector,
                self._tf_indices,
            )

    def generate_count_data(
        self,
        n_counts_per_sample: int = None,
        random_seed: int = None
    ):

        if n_counts_per_sample is None:
            n_counts_per_sample = self.counts_per_sample

        if random_seed is None:
            rng = self._rng
        else:
            rng = np.random.default_rng(random_seed)

        logger.info(
            "Simulating count data from trajectories"
        )

        data = ad.AnnData(
            np.zeros(
                (self.n_samples, self.n_genes),
                dtype=int
            ),
            dtype=int
        )

        _select_times = []

        for i in tqdm.trange(self.n_samples):

            # Randomly select a point on each trajectory
            # Build a joint expression vector
            # And convert it to a probability vector to select counts
            select_expression = [
                traj.random_expression_time(rng=rng)
                for traj in self._trajectories
            ]

            joint_expression = sum([x[0] for x in select_expression])
            joint_expression /= np.sum(joint_expression)

            data.X[i, :] = count_generator(
                joint_expression,
                n_counts_per_sample,
                rng
            )

            _select_times.append([x[1] for x in select_expression])

        ### PACK UP THE UNDERLYING SIMULATED DATA INTO AN ANNDATA OBJECT ###

        data.obs[[x.name for x in self._trajectories]] = _select_times

        data.var['decay_constant'] = self._decay_constants
        data.var['transcriptional_output'] = self._transcriptional_output

        data.uns['network'] = pd.DataFrame(
            self._reg_network,
            index=data.var_names,
            columns=data.var_names.values[self._tf_indices]
        )

        for traj in self._trajectories:

            _pattern_names = [
                f"{n[0]}_{i}"
                for i, n in enumerate(traj.pattern)
            ]

            data.uns[f"{traj.name}"] = pd.DataFrame(
                traj._latent_network,
                index=data.var_names.values[self._tf_indices],
                columns=_pattern_names
            )

        return data
