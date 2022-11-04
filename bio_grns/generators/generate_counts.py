import numpy as np


def count_generator(
    target_probs: np.ndarray,
    n_counts: int,
    random: np.random.Generator
) -> np.ndarray:

    n = target_probs.shape[0]

    return np.bincount(
        random.choice(
            n,
            size=n_counts,
            p=target_probs
        ),
        minlength=n
    )
