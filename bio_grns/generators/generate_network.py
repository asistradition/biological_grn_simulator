from typing import List
import numpy as np

from ..utils import logger


def network_generator(
    n: int,
    m: int,
    n_ratio: List[float],
    n_sparsity: float,
    random: np.random.Generator,
    minimum_abs_value: float = 0.25,
    maximum_abs_value: float = 1.0,
    positive_ratio: float = 0.75,
    row_one_entry: bool = True,
    level_network: bool = False
) -> np.ndarray:
    """
    Simulate a regulatory network between n regulatory
    features and the m downstream targets, with a certain amount of
    sparsity.

    :param n: Number of columns (regulators)
    :type n: int
    :param m_genes: Number of rows (targets)
    :type m_genes: int
    :param n_ratio: Ratio of nonzero values to regulator columns
    :type n_ratio: _type_
    :param n_sparsity: Overall network sparsity
    :type n_sparsity: float
    :param random: Random number generator
    :type random: np.random.Generator
    :param minimum_abs_value: Minimum nonzero absolute value
    :type minimum_abs_value: float
    :param maximum_abs_value: Maximum nonzero absolute value
    :type maximum_abs_value: float
    :param positive_ratio: Ratio of positive to negative nonzero values,
        defaults to 0.75
    :type positive_ratio: float, optional
    :param row_one_entry: Each row may have at most one nonzero value
    :type row_one_entry: bool
    :param level_network: If row has more than one non-zero value,
        relevel so that sum of all values is within the value range
    :type level_network: bool
    """

    # Verify arguments are valid
    if positive_ratio < 0. or positive_ratio > 1.:
        raise ValueError(
            "positive_ratio must be between 0 and 1 "
            f"({positive_ratio} provided)"
        )

    if minimum_abs_value < 0.:
        raise ValueError(
            "minimum_abs_value must be positive "
            f"({minimum_abs_value} provided)"
        )

    if maximum_abs_value < 0.:
        raise ValueError(
            "maximum_abs_value must be positive "
            f"({maximum_abs_value} provided)"
        )

    if maximum_abs_value < minimum_abs_value:
        raise ValueError(
            "maximum_abs_value must be larger than minimum_abs_value "
            f"({minimum_abs_value} - {maximum_abs_value} provided)"
        )

    try:
        if len(n_ratio) != n:
            raise ValueError(
                "n_ratio list must have the same length "
                f"as n ({len(n_ratio)} sparsity "
                f"values provided for {n} features)"
            )

        n_ratio = np.asarray(n_ratio)

        if np.any(n_ratio < 0.):
            raise ValueError(
                "n_ratio may not have negative values"
            )

    except TypeError:
        raise ValueError(
            "n_ratio must be a list "
            f"{type(n_ratio)} provided)"
        )

    tll_network = np.zeros(
        (m, n),
        dtype=float
    )

    n_ratio /= np.sum(n_ratio)

    for n_col in range(n):

        n_connections = m * n_sparsity * n
        n_connections *= n_ratio[n_col]
        n_connections = int(n_connections)

        if n_connections == 0:
            continue

        if row_one_entry:
            unused_m = np.all(
                tll_network == 0.,
                axis=1
            ).astype(float)

            if np.sum(unused_m) == 0:
                unused_m = np.ones_like(unused_m)

            unused_m /= np.sum(unused_m)

        else:
            unused_m = None

        connections = random.uniform(
            minimum_abs_value,
            maximum_abs_value,
            m
        ) * random.choice(
            [1, -1],
            m,
            p=[positive_ratio, 1-positive_ratio]
        )

        m_connections = random.choice(
            m,
            n_connections,
            replace=False,
            p=unused_m
        )

        tll_network[m_connections, n_col] = connections[m_connections]

    if level_network:
        _row_sum = np.sum(tll_network, axis=1)
        _row_sum[_row_sum == 0] = 1.
        _row_mod = np.max(tll_network, axis=1) / _row_sum

        tll_network = np.multiply(tll_network, _row_mod[:, None])

    logger.debug(
        f"Generated {tll_network.shape} network "
        f"with {np.sum(tll_network != 0)} nonzero edges"
    )

    return tll_network
