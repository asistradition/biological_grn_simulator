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
    activation_function: str = "sigmoid"
) -> np.ndarray:

    n = initial_value_vector.shape[0]

    # Regulators which do not have any activity in the
    # activity matrix
    _no_activity = np.sum(activity_matrix != 0, axis=0) == 0

    out_values = np.zeros(
        (m, n),
        float
    )

    out_values[0, :] = initial_value_vector

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
                out_values[_activity_offset_row, tf_indices].reshape(-1, 1)
            ).ravel()

            # Assign standardized expression as activity
            _row_activity[_no_activity] = _offset_activity[_no_activity]

        else:
            _row_activity = activity_matrix[m_row, :]

        out_values[m_row, :] = _time_step(
            out_values[m_row - 1, :],
            _activation_funcs[activation_function](
                np.dot(
                    regulatory_matrix,
                    _row_activity
                )
            ) * transcription_vector,
            decay_vector
        ) * delta_time + out_values[m_row - 1, :]

    return out_values

def _time_step(
    x: np.ndarray,
    alpha: np.ndarray,
    lamb: np.ndarray
) -> np.ndarray:

    return -1 * lamb * x + alpha
