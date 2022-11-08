import numpy as np
import pandas as pd

from ..utils import logger

def network_holdouts(
    network: pd.DataFrame,
    keep_edges_ratio: float,
    random: np.random.Generator
) -> pd.DataFrame:
    """
    Remove a subset of edges from a network
    dataframe

    :param network: Dataframe where connected edges
        are nonzero values
    :type network: pd.DataFrame
    :param keep_edges_ratio: Ratio of edges to keep (0, 1)
    :type keep_edges_ratio: float
    :param random: Random number generator
    :type random: np.random.Generator
    :return: Downsampled network dataframe
    :rtype: pd.DataFrame
    """

    if keep_edges_ratio < 0. or keep_edges_ratio > 1.:
        raise ValueError("keep_edges_ratio must be between 0 and 1")

    nnz = (network != 0).sum().sum()
    n_to_drop = nnz - int(nnz * keep_edges_ratio)

    logger.debug(
        f"Downsampling network with {nnz} edges "
        f"by removing {n_to_drop} edges ({1 - keep_edges_ratio})"
    )

    idx_to_drop = random.choice(
        nnz,
        n_to_drop,
        replace=False
    )

    network_vals = network.values.copy()

    edge_weights = network_vals[network_vals != 0]
    edge_weights[idx_to_drop] = 0

    network_vals[network_vals != 0] = edge_weights

    return pd.DataFrame(
        network_vals,
        index=network.index,
        columns=network.columns
    )
