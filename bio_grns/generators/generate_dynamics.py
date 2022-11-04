from typing import (
    List,
    Tuple
)

from sklearn.preprocessing import MinMaxScaler
import numpy as np


def dynamic_value_generator(
    n: int,
    m: int,
    n_pattern: List[Tuple[str, float, float, float]],
    random: np.random.Generator,
    force_positive: bool = False
) -> np.ndarray:
    """
    Generate m stepwise values for n features based on one
    of three patterns ('random', 'monotonic', and 'cyclic')

    :param n: Number of columns (regulators)
    :type n: int
    :param m_genes: Number of rows (targets)
    :type m_genes: int
    :param n_pattern: List of tuples that indicate which pattern
        to assign to each feature
        'random': (initial_value, step_minimum, step_maximum)
        'cyclic': (zero_point, step_minimum, step_maximum)
        'monotonic': (initial_value, step_minimum, step_maximum)
    :type n_pattern: List[Tuple[str, float, float, float]]
    :param random: Random number generator
    :type random: np.random.Generator
    :param force_positive: Return abs to eliminate negatives,
        defaults to False
    :type force_positive: bool, optional
    :return: Dynamic values in a m x n array
    :rtype: np.ndarray
    """

    try:
        if len(n_pattern) != n:
            raise ValueError(
                "n_pattern list must have the same length "
                f"as n ({len(n_pattern)} patterns "
                f"provided for {n} features)"
            )

    except TypeError:
        raise ValueError(
            "n_pattern must be a list "
            f"{type(n_pattern)} provided)"
        )

    dyn_vals = np.zeros(
        (m, n),
        float
    )

    pattern_func = {
        "random": _random_walk,
        "cyclic": _cyclic_walk,
        "monotonic": _monotonic_walk,
        "updown": _updown_walk
    }

    for n_col, col_pattern in zip(range(n), n_pattern):

        # Unpack pattern tuple
        pattern_name, v1, v2, v3 = col_pattern

        dyn_vals[:, n_col] = pattern_func[pattern_name](
            m,
            v1,
            v2,
            v3,
            random
        )

    if force_positive:
        dyn_vals = np.abs(dyn_vals)

    return dyn_vals


def _random_walk(
    m: int,
    iv: float,
    step_min: float,
    step_max: float,
    rng: np.random.Generator
) -> np.ndarray:

    steps = rng.uniform(
        step_min,
        step_max,
        m
    ) * rng.choice(
        [1, -1],
        m
    )

    steps[0] = iv

    return np.cumsum(steps)


def _cyclic_walk(
    m: int,
    zero_point: float,
    step_min: float,
    step_max: float,
    rng: np.random.Generator
) -> np.ndarray:

    one_point = (0.5 + zero_point) % 1
    one_point = int(one_point * m)

    zero_point = int(zero_point * m)

    steps = rng.uniform(
        step_min,
        step_max,
        m
    )

    if one_point > zero_point:
        lft, rght, dmod, cmod = zero_point, one_point, -1, 1
    else:
        rght, lft, dmod, cmod = zero_point, one_point, 1, -1

    _discont_sum = np.sum(steps[0:lft]) + np.sum(steps[rght:])

    steps[lft:rght] /= np.sum(steps[lft:rght]) * cmod
    steps[0:lft] /= _discont_sum * dmod
    steps[rght:] /= _discont_sum * dmod

    if one_point > zero_point:
        steps[lft:rght] = np.cumsum(steps[lft:rght])
        steps[rght - 1:] = np.cumsum(steps[rght - 1:])
        steps[0:lft] = np.cumsum(steps[0:lft]) + steps[-1]

    else:
        steps[rght:] = np.cumsum(steps[rght:])
        steps[0:rght] = np.cumsum(steps[0:rght]) + steps[-1]

    return steps


def _updown_walk(
    m: int,
    iv: float,
    one_point: float,
    step_limits: Tuple[float, float],
    rng: np.random.Generator
) -> np.ndarray:

    steps = rng.uniform(
        step_limits[0],
        step_limits[1],
        m
    )

    one_point = int(one_point * m)

    steps[0] = 0.
    steps[0:one_point] /= np.sum(steps[0:one_point])
    steps[0:one_point] = np.cumsum(steps[0:one_point])

    # Rescale to fit with the initial value
    steps[0:one_point] = MinMaxScaler(
        feature_range=(iv, 1.)
    ).fit_transform(steps[0:one_point].reshape(-1, 1)).ravel()

    steps[one_point:] /= np.sum(steps[one_point:]) * -1
    steps[one_point - 1:] = np.cumsum(steps[one_point - 1:])

    return steps


def _monotonic_walk(
    m: int,
    iv: float,
    step_min: float,
    step_max: float,
    rng: np.random.Generator
) -> np.ndarray:

    if step_min > step_max:
        step_max, step_min = step_min, step_max

    steps = rng.uniform(
        step_min,
        step_max,
        m
    )

    steps[0] = iv

    return np.cumsum(steps)
