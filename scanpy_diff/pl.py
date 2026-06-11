"""
Visualization functions for differential expression results.

All functions return matplotlib Axes objects for further customization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import markers_to_dict, top_markers

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def volcano(
    markers: pd.DataFrame,
    log2fc_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
    n_label: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color_up: str = "#E74C3C",
    color_down: str = "#3498DB",
    color_ns: str = "#BDC3C7",
    label_fontsize: int = 8,
    point_size: int = 15,
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Volcano plot of differential expression results.

    Plots -log10(padj) vs log2(fold change), with significant genes
    colored and top genes labeled.

    Parameters
    ----------
    markers : pd.DataFrame
        Output from ``find_markers()``.
    log2fc_cutoff : float
        Threshold for log2 fold change (vertical dashed lines).
    padj_cutoff : float
        Threshold for adjusted p-value (horizontal dashed line).
    n_label : int
        Number of top genes to label on each side (up/down).
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size (width, height).
    color_up : str
        Color for upregulated genes.
    color_down : str
        Color for downregulated genes.
    color_ns : str
        Color for non-significant genes.
    label_fontsize : int
        Font size for gene labels.
    point_size : int
        Scatter plot point size.
    ax : plt.Axes, optional
        Existing axes to plot on.
    save : str, optional
        File path to save the figure.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes

    Examples
    --------
    >>> import scanpy_diff as sd
    >>> markers = sd.find_markers(adata, groupby='leiden', group='0')
    >>> sd.pl.volcano(markers, log2fc_cutoff=1.0, padj_cutoff=0.05)
    """
    df = markers.copy()

    # Compute -log10(padj), replacing zeros to avoid inf
    df["_neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    df["_neg_log10_padj"] = df["_neg_log10_padj"].clip(upper=300)

    # Classification
    sig_threshold = -np.log10(padj_cutoff)
    up_mask = (df["log2fc"] >= log2fc_cutoff) & (df["_neg_log10_padj"] >= sig_threshold)
    down_mask = (df["log2fc"] <= -log2fc_cutoff) & (df["_neg_log10_padj"] >= sig_threshold)
    ns_mask = ~(up_mask | down_mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot non-significant
    ax.scatter(
        df.loc[ns_mask, "log2fc"],
        df.loc[ns_mask, "_neg_log10_padj"],
        c=color_ns,
        s=point_size,
        alpha=0.5,
        linewidths=0,
        label=f"NS ({ns_mask.sum()})",
        rasterized=True,
    )

    # Plot upregulated
    ax.scatter(
        df.loc[up_mask, "log2fc"],
        df.loc[up_mask, "_neg_log10_padj"],
        c=color_up,
        s=point_size,
        alpha=0.8,
        linewidths=0,
        label=f"Up ({up_mask.sum()})",
        rasterized=True,
    )

    # Plot downregulated
    ax.scatter(
        df.loc[down_mask, "log2fc"],
        df.loc[down_mask, "_neg_log10_padj"],
        c=color_down,
        s=point_size,
        alpha=0.8,
        linewidths=0,
        label=f"Down ({down_mask.sum()})",
        rasterized=True,
    )

    # Threshold lines
    ax.axhline(
        y=sig_threshold,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.axvline(
        x=log2fc_cutoff,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.axvline(
        x=-log2fc_cutoff,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    # Label top genes
    if n_label > 0:
        _label_genes_volcano(
            ax=ax,
            df=df,
            up_mask=up_mask,
            down_mask=down_mask,
            n_label=n_label,
            fontsize=label_fontsize,
            color_up=color_up,
            color_down=color_down,
        )

    ax.set_xlabel("log$_2$ Fold Change", fontsize=12)
    ax.set_ylabel("-log$_{10}$(adj. p-value)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    if title is None:
        group = markers.attrs.get("group", "")
        ref = markers.attrs.get("reference", "rest")
        method = markers.attrs.get("method", "")
        title = f"Volcano: {group} vs {ref} ({method})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def _label_genes_volcano(
    ax: plt.Axes,
    df: pd.DataFrame,
    up_mask: pd.Series,
    down_mask: pd.Series,
    n_label: int,
    fontsize: int,
    color_up: str,
    color_down: str,
) -> None:
    """Label top genes on a volcano plot."""
    try:
        from adjustText import adjust_text
        has_adjusttext = True
    except ImportError:
        has_adjusttext = False

    texts = []

    # Top upregulated
    top_up = df[up_mask].nlargest(n_label, "_neg_log10_padj")
    for _, row in top_up.iterrows():
        t = ax.text(
            row["log2fc"],
            row["_neg_log10_padj"],
            row["gene"],
            fontsize=fontsize,
            color=color_up,
            ha="left",
        )
        texts.append(t)

    # Top downregulated
    top_down = df[down_mask].nlargest(n_label, "_neg_log10_padj")
    for _, row in top_down.iterrows():
        t = ax.text(
            row["log2fc"],
            row["_neg_log10_padj"],
            row["gene"],
            fontsize=fontsize,
            color=color_down,
            ha="right",
        )
        texts.append(t)

    if has_adjusttext and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )


def marker_heatmap(
    adata: AnnData,
    all_markers: pd.DataFrame,
    groupby: str,
    n_genes: int = 5,
    layer: Optional[str] = None,
    use_raw: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdYlBu_r",
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Heatmap of top marker genes per cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    groupby : str
        Column in ``adata.obs`` used for grouping.
    n_genes : int
        Number of top marker genes per cluster to show.
    layer : str, optional
        Expression layer to use.
    use_raw : bool
        Use ``adata.raw`` if True.
    figsize : tuple, optional
        Figure size. Auto-computed if None.
    cmap : str
        Colormap for expression values.
    save : str, optional
        File path to save the figure.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes
    """
    import scanpy as sc

    top = top_markers(all_markers, n=n_genes)
    genes = top["gene"].tolist()

    # Remove duplicate genes (a gene may be top marker for multiple clusters)
    genes_unique = list(dict.fromkeys(genes))

    if figsize is None:
        w = max(8, len(genes_unique) * 0.3)
        h = max(5, adata.obs[groupby].nunique() * 0.4 + 2)
        figsize = (w, h)

    ax = sc.pl.heatmap(
        adata,
        var_names=genes_unique,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        cmap=cmap,
        figsize=figsize,
        show=False,
        save=False,
    )

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def dotplot(
    adata: AnnData,
    all_markers: pd.DataFrame,
    groupby: str,
    n_genes: int = 5,
    layer: Optional[str] = None,
    use_raw: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Dot plot of top marker genes per cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    groupby : str
        Column in ``adata.obs`` used for grouping.
    n_genes : int
        Number of top marker genes per cluster to show.
    layer : str, optional
        Expression layer to use.
    use_raw : bool
        Use ``adata.raw`` if True.
    figsize : tuple, optional
        Figure size.
    save : str, optional
        File path to save.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes
    """
    import scanpy as sc

    marker_dict = markers_to_dict(all_markers, n=n_genes)

    ax = sc.pl.dotplot(
        adata,
        var_names=marker_dict,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        figsize=figsize,
        show=False,
        save=False,
    )

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def violin(
    adata: AnnData,
    genes: Union[str, List[str]],
    groupby: str,
    layer: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Violin plot of gene expression across groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : str or list of str
        Gene(s) to plot.
    groupby : str
        Column in ``adata.obs`` for grouping.
    layer : str, optional
        Expression layer.
    figsize : tuple, optional
        Figure size.
    save : str, optional
        File path to save.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes
    """
    import scanpy as sc

    if isinstance(genes, str):
        genes = [genes]

    # If figsize given, set figure size before calling scanpy
    if figsize is not None:
        plt.figure(figsize=figsize)

    ax = sc.pl.violin(
        adata,
        keys=genes,
        groupby=groupby,
        layer=layer,
        show=False,
        save=False,
    )

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def summary_barplot(
    all_markers: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 4),
    color: str = "#3498DB",
    title: str = "Number of Markers per Cluster",
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Bar chart showing the number of marker genes per cluster.

    Parameters
    ----------
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
    title : str
        Plot title.
    save : str, optional
        File path to save.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes
    """
    if "cluster" not in all_markers.columns:
        raise ValueError("DataFrame must have a 'cluster' column.")

    counts = all_markers.groupby("cluster").size().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.8)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Markers", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def log2fc_heatmap(
    all_markers: pd.DataFrame,
    n_genes: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    vmax: float = 3.0,
    save: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Heatmap of log2 fold changes for top markers across all clusters.

    Rows are genes, columns are clusters. Values are log2FC of that gene
    in each cluster (from find_all_markers results).

    Parameters
    ----------
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    n_genes : int
        Number of top markers per cluster.
    figsize : tuple, optional
        Figure size.
    cmap : str
        Colormap.
    vmax : float
        Maximum value for color scale (vmin = -vmax).
    save : str, optional
        File path to save.
    show : bool
        Whether to show the figure.

    Returns
    -------
    plt.Axes
    """
    top = top_markers(all_markers, n=n_genes)
    clusters = sorted(all_markers["cluster"].unique().tolist())
    genes_ordered = top["gene"].tolist()

    # Build pivot table: rows=genes, cols=clusters
    pivot = all_markers.pivot_table(
        index="gene", columns="cluster", values="log2fc", aggfunc="max"
    )

    # Reindex to selected genes (preserve order)
    genes_in_pivot = [g for g in genes_ordered if g in pivot.index]
    pivot = pivot.reindex(index=genes_in_pivot, columns=clusters)

    if figsize is None:
        w = max(6, len(clusters) * 0.7 + 2)
        h = max(5, len(genes_in_pivot) * 0.25 + 2)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            pivot,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            center=0,
            ax=ax,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "log$_2$FC", "shrink": 0.7},
        )
    else:
        im = ax.imshow(
            pivot.values,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, rotation=45, ha="right")
        ax.set_yticks(range(len(genes_in_pivot)))
        ax.set_yticklabels(genes_in_pivot, fontsize=7)
        plt.colorbar(im, ax=ax, label="log$_2$FC", shrink=0.7)

    ax.set_title(
        f"Top {n_genes} Markers per Cluster (log₂FC)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax
