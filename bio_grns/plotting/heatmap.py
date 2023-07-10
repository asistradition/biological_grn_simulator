from typing import Union

import matplotlib.pyplot as plt
import matplotlib.colors as cm

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

import numpy as np

from ..synthesize import GRNSimulator
from .utils import _lookup_trajectory


def expression_heatmap(
    model: GRNSimulator,
    trajectory: Union[int, str] = 0,
    ax: plt.Axes = None,
    cbar_ax: plt.Axes = None,
    cmap: Union[str, cm.Colormap] = 'viridis',
    log: bool = False,
    tpm: bool = False
):

    if ax is None:
        ax, cbar_ax = _draw_fig()

    # Convert to TPM for plotting
    tpm_expression = _lookup_trajectory(
        model,
        trajectory
    ).expression.copy()

    if tpm:
        tpm_expression /= np.sum(tpm_expression, axis=1)[:, None]
        tpm_expression *= 1e6

    if log:
        tpm_expression = np.log1p(tpm_expression)

    _tpm_idx = _hclust_order(
        tpm_expression.T,
        metric='euclidean'
    )

    _lab = "log({v} + 1)" if log else "{v}"
    _lab = _lab.format(v="TPM" if tpm else "Count")

    _heatmap(
        ax,
        cbar_ax,
        tpm_expression[:, _tpm_idx].T,
        cmap=cmap,
        cbar_label=_lab
    )

    ax.set_ylabel("Genes")
    ax.set_xlabel("Time")
    ax.set_title(f"Trajectory {trajectory} Expression")

    return ax


def activity_heatmap(
    model: GRNSimulator,
    trajectory: Union[int, str] = 0,
    ax: plt.Axes = None,
    cbar_ax: plt.Axes = None,
    cmap: Union[str, cm.Colormap] = 'viridis'
):

    if ax is None:
        ax, cbar_ax = _draw_fig()

    # Convert to TPM for plotting
    activity = _lookup_trajectory(
        model,
        trajectory
    ).activity.copy()

    _act_idx = _hclust_order(
        activity.T,
        metric='euclidean'
    )

    _heatmap(
        ax,
        cbar_ax,
        activity[:, _act_idx].T,
        cmap=cmap,
        cbar_label="Activity"
    )

    ax.set_ylabel("TFs")
    ax.set_xlabel("Time")
    ax.set_title(f"Trajectory {trajectory} Activity")

    return ax


def latent_heatmap(
    model: GRNSimulator,
    trajectory: Union[int, str] = 0,
    ax: plt.Axes = None,
    cbar_ax: plt.Axes = None,
    cmap: Union[str, cm.Colormap] = 'viridis'
):

    if ax is None:
        ax, cbar_ax = _draw_fig()

    # Convert to TPM for plotting
    latent = _lookup_trajectory(
        model,
        trajectory
    )._dynamic_values.copy()

    _lat_idx = _hclust_order(
        latent.T,
        metric='euclidean'
    )

    _ylabs = np.array(
        [
            n[0]
            for n in _lookup_trajectory(
                model,
                trajectory
            ).pattern
        ]
    )

    _heatmap(
        ax,
        cbar_ax,
        latent[:, _lat_idx].T,
        cmap=cmap,
        cbar_label="Latent Activity"
    )

    ax.set_xlabel("Time")
    ax.set_yticks(
        np.arange(len(_ylabs)) + 0.5,
        _ylabs[_lat_idx]
    )
    ax.set_title(f"Trajectory {trajectory} Latent Pattern")

    return ax


def _draw_fig():

    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes([0.075, 0.075, 0.8, 0.8])
    cbar_ax = fig.add_axes([0.9, 0.075, 0.02, 0.8])

    return ax, cbar_ax


def _hclust_order(
    data: np.ndarray,
    metric: str = 'euclidean'

):
    """
    Generate an index to reorder data

    :param data: Data
    :type data: np.ndarray
    :return: Ordering index integer array
    :rtype: np.ndarray
    """

    # Fill NaNs
    _dist = pdist(
        data,
        metric=metric
    )

    _dist[np.isnan(_dist)] = 0.

    # Hclust for ordering
    return dendrogram(
        linkage(
            _dist
        ),
        no_plot=True
    )['leaves']


def _heatmap(
    ax: plt.Axes,
    cbar_ax: plt.Axes,
    data: np.ndarray,
    cmap: Union[str, cm.Colormap] = 'viridis',
    cbar_label: str = None
):

    # Plot
    matrix_ref = ax.pcolormesh(
        data,
        cmap=cmap,
        vmin=0,
        vmax=data.max()
    )

    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if cbar_ax is not None:
        cbar_ref = cbar_ax.get_figure().colorbar(
            matrix_ref,
            cax=cbar_ax,
            orientation='vertical',
            ticks=[0, np.nanmax(data)],
            format='{x:0.2f}'
        )

        cbar_ax.yaxis.set_tick_params(labelsize=8)

        if cbar_label is not None:
            cbar_ref.set_label(
                cbar_label,
                labelpad=-10,
                size=8,
                rotation=270
            )
