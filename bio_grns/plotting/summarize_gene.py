from typing import Union

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt


def plot_gene_summary(
    adata: ad.AnnData,
    _gene: int,
    trajectory_for_time: Union[int, str],
    ax: np.ndarray = None
):

    _time = adata.obs[trajectory_for_time]

    if ax is None:

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(4, 4),
            dpi=300,
            gridspec_kw={'wspace': 0.4, 'hspace': 0.4}
        )

    ax[0, 0].scatter(
        _time,
        adata.X[:, _gene],
        color='black',
        s=2,
        alpha=0.1
    )

    ax[0, 0].plot(
        np.arange(22),
        np.array(
            [
                np.median(adata.X[adata.obs[trajectory_for_time] == i, _gene])
                for i in range(22)
            ]
        )
    )

    ax[0, 0].set_title("Counts")
    ax[0, 0].set_ylim(0, None)

    ax[0, 1].scatter(
        _time,
        adata.layers['velocity'][:, _gene],
        color='black',
        s=2,
        alpha=0.1
    )

    ax[0, 1].set_title("Velocity")
    _vmax = np.max(np.abs(adata.layers['velocity'][:, _gene]))
    ax[0, 1].set_ylim(-1 * _vmax, _vmax)

    ax[1, 0].scatter(
        _time,
        adata.layers['transcription'][:, _gene],
        color='black',
        s=2,
        alpha=0.1
    )

    ax[1, 0].set_title("Transcription")
    ax[1, 0].set_ylim(0, None)

    ax[1, 1].scatter(
        _time,
        adata.layers['decay'][:, _gene].ravel(),
        color='black',
        s=2,
        alpha=0.1
    )

    ax[1, 1].set_title("Decay")
    ax[1, 1].set_ylim(None, 0)

    return ax
