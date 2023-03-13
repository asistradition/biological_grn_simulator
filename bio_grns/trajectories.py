from typing import (
    Tuple
)

from .generators.generate_dynamics import dynamic_value_generator
from .calculators import values_from_network_latent
from .calculators import values_from_dynamic_network
from .utils import logger

import numpy as np


class Trajectory:

    pattern = None
    name = None

    target_ratios = None
    latent_network_sparsity = None

    rng = None

    _activity = None
    expression = None

    _dynamic_values = None
    _latent_network = None

    @property
    def activity(self):
        if self._activity is None:
            self._calculate_activity()

        return self._activity

    @property
    def dynamics(self):
        if self._dynamic_values is None:
            self._calculate_dynamics()

        return self._dynamic_values

    @property
    def n_patterns(self):
        return len(self.pattern)

    def __init__(
        self,
        n_time_steps: int,
        rng: np.random.Generator = None,
        force_positive: bool = True,
        primary_trajectory: bool = True,
        latent_network_sparsity: float = 0.1,
        offset_expression_activity: int = 15,
        name: str = None
    ) -> None:

        self.pattern = []
        self.target_ratios = []

        self.n_time_steps = n_time_steps
        self.time_offset_expression_activity = offset_expression_activity

        self.force_positive = force_positive
        self.primary_trajectory = primary_trajectory
        self.rng = rng
        self.latent_network_sparsity = latent_network_sparsity
        self.name = name

    def add_pattern(
        self,
        pattern_name: str,
        *args: float,
        target_edge_ratio: float = 1.
    ) -> None:
        """
        Add an activity pattern to this trajectory.

        'random' is a random walk in positive and negative directions
        magnitude ~ U(step_minimum, step_maximum).
        Also requires initial_value

        'cyclic' is a rotation between zero and one with random step
        magnitude ~ U(step_minimum, step_maximum).
        Also requires the location to place the zero point (0, 1)

        'monotonic' is a random walk in only positive or only negative
        directions where magnitude ~ U(step_minimum, step_maximum).
        Also requires initial_value

        'random': (initial_value, step_minimum, step_maximum)
        'cyclic': (zero_point, step_minimum, step_maximum)
        'monotonic': (initial_value, step_minimum, step_maximum)

        :param pattern_name: Pattern name
        :type pattern_name: str
        :param *args: Pattern arguments
        :type *args: floats
        :param target_edge_ratio: Ratio of TFs that are influenced
            by this pattern
        :type target_edge_ratio: float
        """

        self.pattern.append(
            (pattern_name,) + args
        )

        self.target_ratios.append(
            target_edge_ratio
        )

    def set_network(
        self,
        network
    ) -> None:
        self._latent_network = network

    def _calculate_activity(self) -> None:
        if self._latent_network is None:
            raise RuntimeError(
                "Unable to calculate activity without a network"
            )

        logger.debug(
            f"Generating pattern regulatory activity for {self.name}"
        )

        self._activity = values_from_network_latent(
            self._latent_network,
            self.dynamics
        )

    def _calculate_dynamics(self) -> None:

        logger.debug(
            f"Generating pattern dynamic states for {self.name}"
        )

        self._dynamic_values = dynamic_value_generator(
            len(self.pattern),
            self.n_time_steps,
            self.pattern,
            self.rng,
            force_positive=self.force_positive
        )

    def calculate_expression(
        self,
        regulatory_matrix: np.ndarray,
        decay_vector: np.ndarray,
        transcription_vector: np.ndarray,
        initial_value_vector: np.ndarray,
        tf_indices: np.ndarray,
        **kwargs
    ) -> None:

        if self.expression is not None:
            return

        logger.debug(
            f"Generating pattern gene expression for {self.name}"
        )

        self.expression = values_from_dynamic_network(
            self.n_time_steps,
            self.activity,
            regulatory_matrix,
            decay_vector,
            transcription_vector,
            initial_value_vector,
            tf_indices,
            include_non_activity_tfs=self.primary_trajectory,
            offset_expression_activity=self.time_offset_expression_activity,
            **kwargs
        )

        self._times = np.arange(self.n_time_steps)
        self._times -= self.time_offset_expression_activity

    def random_expression_time(
        self,
        rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, int]:

        if rng is None:
            rng = self.rng

        _idx = rng.choice(
            np.arange(
                self.time_offset_expression_activity,
                self.n_time_steps
            ),
            1
        )[0]

        return self.expression[_idx, :], self._times[_idx]
