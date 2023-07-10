from typing import Callable

import numpy as np
from scipy.special import (
    softmax,
    expit
)
from sklearn.preprocessing import MinMaxScaler

from ..utils import logger


def relu(x):
    return np.maximum(
        0,
        x
    )


def relu_onemax(x):
    return np.minimum(
        1,
        relu(x)
    )


_activation_funcs = {
    'relu': relu,
    'relu_onemax': relu_onemax,
    'softmax': softmax,
    'sigmoid': expit,
    'linear': lambda x: x
}


def values_from_dynamic_network(
    m: int,
    activity_matrix: np.ndarray,
    regulatory_matrix: np.ndarray,
    decay_vector: np.ndarray,
    transcription_vector: np.ndarray,
    initial_value_vector: np.ndarray,
    tf_indices: np.ndarray,
    include_non_activity_tfs: bool = True,
    delta_time: float = 1.0,
    offset_expression_activity: int = 15,
    activation_function: str = "relu_onemax",
    balance_transcription_decay: bool = True,
    return_layers: bool = False
) -> np.ndarray:
    """
    Generate expression values based on ODE

    :param m: Number of time output steps
    :type m: int
    :param activity_matrix: TF activity matrix (m x k)
    :type activity_matrix: np.ndarray
    :param regulatory_matrix: TF to gene regulation matrix [g x k]
    :type regulatory_matrix: np.ndarray
    :param decay_vector: Vector of decay constants [g]
    :type decay_vector: np.ndarray
    :param transcription_vector: Vector of maximum transcriptional rates [g]
    :type transcription_vector: np.ndarray
    :param initial_value_vector: Initial expression vector at t0
    :type initial_value_vector: np.ndarray
    :param tf_indices: TF indices for TFs where activity == expression
    :type tf_indices: np.ndarray
    :param include_non_activity_tfs: Include TFs where activity == expression,
        defaults to True
    :type include_non_activity_tfs: bool, optional
    :param delta_time: Magnitude of dt, defaults to 1.0
    :type delta_time: float, optional
    :param offset_expression_activity: Temporal offset for TF activity derived
        from expression, defaults to 15
    :type offset_expression_activity: int, optional
    :param activation_function: TF activity activation function,
        defaults to "sigmoid"
    :type activation_function: str, optional
    :return: Expression array [m x g]
    :rtype: np.ndarray
    """

    n = initial_value_vector.shape[0]

    # Regulators which do not have any activity in the
    # activity matrix
    _no_activity = np.sum(activity_matrix != 0, axis=0) == 0

    out_expression = np.zeros(
        (m, n),
        float
    )

    out_transcription = np.zeros(
        (m, n),
        float
    )

    out_velocity = np.zeros(
        (m, n),
        float
    )

    out_decay = np.zeros(
        (m, n),
        float
    )

    out_expression[0, :] = initial_value_vector

    logger.debug(
        f"Generating dynamic expression ({m} x {n}) "
        f"with activation function {activation_function}"
    )

    for m_row in range(1, m):

        # Get the expression of prior TFs
        if include_non_activity_tfs:

            _activity_offset_row = max(
                0,
                m_row - offset_expression_activity
            )

            _row_activity = activity_matrix[m_row, :].copy()

            # Standardize regulator expression to the same range as activity
            _offset_activity = MinMaxScaler(
                feature_range=(0, _row_activity.max())
            ).fit_transform(
                out_expression[_activity_offset_row, tf_indices].reshape(-1, 1)
            ).ravel()

            # Assign standardized expression as activity
            _row_activity[_no_activity] = _offset_activity[_no_activity]

        else:
            _row_activity = activity_matrix[m_row, :]

        out_transcription[m_row, :] = _transcription_step(
            _row_activity,
            regulatory_matrix,
            transcription_vector,
            _activation_funcs[activation_function]
        )

        out_decay[m_row, :] = _decay_step(
            out_expression[m_row - 1, :],
            decay_vector
        )

        if balance_transcription_decay:
            _td_ratio = out_transcription[m_row, :].sum()
            _td_ratio /= out_decay[m_row, :].sum()

            out_transcription[m_row, :] /= np.abs(_td_ratio)

        out_velocity[m_row, :] = out_transcription[m_row, :]
        out_velocity[m_row, :] += out_decay[m_row, :]
        out_velocity[m_row, :] *= delta_time

        out_expression[m_row, :] = out_velocity[m_row, :]
        out_expression[m_row, :] += out_expression[m_row - 1, :]

    if return_layers:
        return out_expression, out_velocity, out_transcription, out_decay
    else:
        return out_expression


def _transcription_step(
    activity: np.ndarray,
    regulatory_matrix: np.ndarray,
    transcription_vector: np.ndarray,
    activation_function: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:

    return activation_function(
        np.dot(
            regulatory_matrix,
            activity
        )
    ) * transcription_vector


def _decay_step(
    x: np.ndarray,
    lamb: np.ndarray
) -> np.ndarray:

    return -1 * lamb * x
