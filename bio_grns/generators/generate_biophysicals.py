from typing import (
    Tuple
)

import numpy as np

def generate_decays(
    n: int,
    random: np.random.Generator,
    decay_limits: Tuple[float, float] = None,
    halflife_limits: Tuple[float, float] = None
) -> np.ndarray:
    """
    Generate a set of decay constants
    lambda ~ U(np.log(2) / max_halflife, np.log(2) / min_halflife)

    :param n: Number of decay constants to generate
    :type n: int
    :param decay_limits: Decay constant limits (min, max)
    :type decay_limits: float, float
    :param halflife_limits: Halflife limits (min, max)
        (in time units)
    :type halflife_limits: float, float
    :param random: Random number generator
    :type random: np.random.Generator
    :return: Array of decay constant values
    :rtype: np.ndarray
    """

    if halflife_limits is not None:

        if min(halflife_limits) <= 0:
            raise ValueError(
                "Half life may not be <= 0"
            )

        decay_low = np.log(2) / halflife_limits[1]
        decay_high = np.log(2) / halflife_limits[0]

    elif decay_limits is not None:
        decay_low = decay_limits[0]
        decay_high = decay_limits[1]

    else:
        raise ValueError(
            "Pass halflife_limits or decay_limits"
        )

    if min(decay_low, decay_high) <= 0:
        raise ValueError(
            "decay constant limits must be in (0, inf)"
        )

    elif decay_low > decay_high:
        raise ValueError(
            "decay constants must be passed as (min, max) "
            f"({decay_low} - {decay_high} provided)"
        )

    return random.uniform(
        decay_low,
        decay_high,
        n
    )

def generate_transcription_rates(
    n: int,
    random: np.random.Generator,
    scale: float = 1.,
    shape: float = 1.
) -> np.ndarray:
    """
    Generate a set of maximum transcription outputs
    alpha ~ Gamma(1, 1)

    :param n: Number of transcriptional outputs to generate
    :type n: int
    :param random: Random number generator
    :type random: np.random.Generator
    :param scale: Gamma scale parameter, defaults to 1
    :type scale: float, optional
    :param shape: Gamma shape parameter, defaults to 1
    :type shape: float, optional
    :return: Array of transcriptional output values
    :rtype: np.ndarray
    """

    return random.gamma(
        shape,
        scale,
        n
    )

def generate_tf_indices(
    n_genes: int,
    n_tfs: int,
    random: np.random.Generator
) -> np.ndarray:
    """
    Randomly assign a set of genes as TFs

    :param n_genes: Number of genes
    :type n_genes: int
    :param n_tfs: Number of TFs
    :type n_tfs: int
    :param random: Random number generator
    :type random: np.random.Generator
    :return: Array of sorted gene indices for TFs
    :rtype: np.ndarray
    """

    idx = random.choice(
        n_genes,
        n_tfs,
        replace=False
    )

    idx.sort()

    return idx
