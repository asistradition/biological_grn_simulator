import numpy as np
import anndata as ad
import pandas as pd
import tqdm
import scipy.sparse as spsparse

from .trajectories import Trajectory
from .generators import (
    network_generator,
    generate_transcription_rates,
    generate_decays,
    generate_tf_indices,
    count_generator
)
from .calculators import network_holdouts

from .utils import logger


class GRNSimulator:

    _trajectories = None
    _rng = None

    n_genes = None
    n_tfs = None
    n_samples = None

    n_time_steps = None
    counts_per_sample = None

    activation_function = 'relu_onemax'

    _latent_networks = None
    _tf_indices = None
    _transcriptional_output = None
    _decay_constants = None

    # Simulation Parameters #
    _halflife_limits = (10, 100)
    _reg_network_sparsity = 0.03
    _reg_network_positive = 0.8

    _data = None
    _reg_network = None

    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        random_seed: int,
        n_samples: int = 1000,
        counts_per_sample: int = 5000,
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

        self._generate_tf_indices()
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

    def _generate_tf_indices(self):

        self._tf_indices = generate_tf_indices(
            self.n_genes,
            self.n_tfs,
            self._rng
        )

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
                positive_ratio=1.0,
                row_one_entry=False,
                level_network=True
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

            _initial_probs = self._rng.uniform(
                size=self.n_genes
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

                _initial_probs[~_relevant_genes] = 0.

            _initial_probs /= _initial_probs.sum()

            initial_vector = np.bincount(
                self._rng.choice(
                    np.arange(self.n_genes),
                    size=self.counts_per_sample,
                    p=_initial_probs
                ),
                minlength=self.n_genes
            )

            traj.calculate_expression(
                self._reg_network,
                self._decay_constants,
                self._transcriptional_output,
                initial_vector,
                self._tf_indices,
                activation_function=self.activation_function
            )

    def generate_count_data(
        self,
        n_samples: int = None,
        n_counts_per_sample: int = None,
        random_seed: int = None,
        sparse: bool = False,
        no_metadata: bool = False
    ):

        # Use defaults from workflow
        # If not provided to this function
        if n_counts_per_sample is None:
            n_counts_per_sample = self.counts_per_sample

        if n_samples is None:
            n_samples = self.n_samples

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

        data.layers['velocity'] = np.zeros(data.X.shape)
        data.layers['transcription'] = np.zeros(data.X.shape)
        data.layers['decay'] = np.zeros(data.X.shape)

        _select_times = []

        for i in tqdm.trange(self.n_samples):

            # Randomly select a point on each trajectory
            # Build a joint expression vector
            # And convert it to a probability vector to select counts
            _traj_data, _time = _combine_trajectories([
                traj.random_expression_time(rng=rng)
                for traj in self._trajectories
            ])

            _count_depth = np.sum(_traj_data[..., 0])

            _traj_data[..., 0] /= _count_depth

            data.X[i, :] = count_generator(
                _traj_data[..., 0],
                n_counts_per_sample,
                rng
            )

            # Modify biophyscal layers to match count depth
            _mod = n_counts_per_sample / _count_depth

            data.layers['velocity'][i, :] = _traj_data[..., 1] * _mod
            data.layers['transcription'][i, :] = _traj_data[..., 2] * _mod
            data.layers['decay'][i, :] = _traj_data[..., 3] * _mod

            _select_times.append(_time)

        # PACK UP THE UNDERLYING SIMULATED DATA INTO AN ANNDATA OBJECT #

        if sparse:
            data.X = spsparse.csr_matrix(data.X)

        if not no_metadata:
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

                data.uns[f"{traj.name}_network"] = pd.DataFrame(
                    traj._latent_network,
                    index=data.var_names.values[self._tf_indices],
                    columns=_pattern_names
                )

                data.uns[f"{traj.name}_dynamic_activity"] = pd.DataFrame(
                    traj._dynamic_values,
                    columns=_pattern_names
                )

        return data

    def save_network(
        self,
        output_file,
        edges_to_include=1.0,
        sep="\t",
        **kwargs
    ) -> None:

        network = pd.DataFrame(
            self._reg_network
        )
        network.index = network.index.astype('str')
        network.columns = network.index.values[self._tf_indices]

        if edges_to_include != 1.0:
            network = network_holdouts(
                network,
                edges_to_include,
                self._rng
            )

        network.to_csv(
            output_file,
            sep=sep,
            float_format='%.5f',
            **kwargs
        )


def _combine_trajectories(
    trajectory_data
):

    times = [x[1] for x in trajectory_data]
    data = sum([x[0] for x in trajectory_data])

    return data, times
