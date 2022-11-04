from .generators.generate_dynamics import dynamic_value_generator
from .values_from_network import values_from_network_latent
from .values_from_dynamics import values_from_dynamic_network

import numpy as np

class Trajectory:

    pattern = None
    target_ratios = None

    rng = None

    _activity = None
    _expression = None

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

    def __init__(
        self,
        n_tfs: int,
        rng: np.random.Generator,
        force_positive: bool = True,
        primary_trajectory: bool = True
    ) -> None:

        self.pattern = []
        self.target_ratios = []

        self.n_tfs = n_tfs
        self.force_positive = force_positive
        self.primary_trajectory = primary_trajectory
        self.rng = rng

    def add_pattern(
        self,
        pattern_name,
        *args,
        target_edge_ratio = 1.
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
        :param target_edge_ratio: Ratio of TFs that are influenced by this pattern
        :type target_edge_ratio: float
        """

        self.pattern.append(
            tuple(
                pattern_name,
                *args
            )
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

        self._activity = values_from_network_latent(
            self._latent_network,
            self._dynamic_values
        )

    def _calculate_dynamics(self) -> None:

        self._dynamic_values = dynamic_value_generator(
            len(self.pattern),
            self.n_tfs,
            self.pattern,
            self.rng,
            force_positive=self.force_positive
        )

    def expression(
        self,
        n_steps: int,
        regulatory_matrix: np.ndarray,
        decay_vector: np.ndarray,
        transcription_vector: np.ndarray,
        initial_value_vector: np.ndarray,
        tf_indices: np.ndarray,
        **kwargs
    ) -> np.ndarray:

        if self._expression is not None:
            return self._expression

        self._expression = values_from_dynamic_network(
            n_steps,
            self.activity,
            regulatory_matrix,
            decay_vector,
            transcription_vector,
            initial_value_vector,
            tf_indices,
            include_non_activity_tfs=self.primary_trajectory,
            **kwargs
        )

        return self._expression
