import numpy as np

def values_from_network_latent(
    network: np.ndarray,
    regulator_activity: np.ndarray
) -> np.ndarray:
    """
    Propagate latent values through a network

    :param network: Network array
    :type network: np.ndarray
    :param regulator_activity: Latent array
    :type regulator_activity: np.ndarray
    :return: Output array
    :rtype: np.ndarray
    """

    return np.dot(
        regulator_activity,
        network.T
    )
