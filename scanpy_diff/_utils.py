"""
Utility functions for working with differential expression results.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from anndata import AnnData


def filter_markers(
    markers: pd.DataFrame,
    min_log2fc: float = 0.0,
    max_padj: float = 0.05,
    min_pct: float = 0.0,
    max_pct_reference: float = 1.0,
    only_positive: bool = False,
    only_negative: bool = False,
) -> pd.DataFrame:
    """
    Filter a markers DataFrame by various criteria.

    Parameters
    ----------
    markers : pd.DataFrame
        Output from ``find_markers()`` or ``find_all_markers()``.
    min_log2fc : float
        Minimum absolute log2 fold change.
    max_padj : float
        Maximum adjusted p-value.
    min_pct : float
        Minimum pct_1 (fraction of cells in group expressing gene).
    max_pct_reference : float
        Maximum pct_2 (fraction of cells in reference expressing gene).
    only_positive : bool
        If True, keep only upregulated genes (log2fc > 0).
    only_negative : bool
        If True, keep only downregulated genes (log2fc < 0).

    Returns
    -------
    pd.DataFrame
        Filtered markers DataFrame.

    Examples
    --------
    >>> markers_filtered = sd.filter_markers(
    ...     markers, min_log2fc=1.0, max_padj=0.01, min_pct=0.25
    ... )
    """
    df = markers.copy()

    if min_log2fc > 0:
        df = df[np.abs(df["log2fc"]) >= min_log2fc]
    if max_padj < 1.0:
        df = df[df["padj"] <= max_padj]
    if min_pct > 0:
        df = df[df["pct_1"] >= min_pct]
    if max_pct_reference < 1.0:
        df = df[df["pct_2"] <= max_pct_reference]
    if only_positive:
        df = df[df["log2fc"] > 0]
    if only_negative:
        df = df[df["log2fc"] < 0]

    return df.reset_index(drop=True)


def rank_markers(
    markers: pd.DataFrame,
    by: Literal["log2fc", "padj", "pval", "scores", "combined"] = "combined",
    n_top: Optional[int] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Re-rank markers by a given criterion.

    Parameters
    ----------
    markers : pd.DataFrame
        Output from ``find_markers()`` or ``find_all_markers()``.
    by : str
        Ranking criterion:

        - ``'log2fc'``   : Rank by log2 fold change.
        - ``'padj'``     : Rank by adjusted p-value.
        - ``'pval'``     : Rank by raw p-value.
        - ``'scores'``   : Rank by test statistic.
        - ``'combined'`` : Combined score = log2fc * -log10(padj + 1e-300).
    n_top : int, optional
        Return only the top N markers.
    ascending : bool
        Sort in ascending order. Default is False (best markers first).

    Returns
    -------
    pd.DataFrame
        Re-ranked markers DataFrame.
    """
    df = markers.copy()

    if by == "combined":
        df["_rank_score"] = df["log2fc"] * (-np.log10(df["padj"] + 1e-300))
        df = df.sort_values("_rank_score", ascending=ascending).drop(
            columns=["_rank_score"]
        )
    elif by == "padj":
        df = df.sort_values("padj", ascending=not ascending)
    else:
        df = df.sort_values(by, ascending=ascending)

    df = df.reset_index(drop=True)

    if n_top is not None:
        df = df.head(n_top)

    return df


def top_markers(
    all_markers: pd.DataFrame,
    n: int = 10,
    by: Literal["log2fc", "padj", "combined"] = "combined",
) -> pd.DataFrame:
    """
    Get top N markers per cluster from find_all_markers() output.

    Parameters
    ----------
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    n : int
        Number of top markers per cluster.
    by : str
        Ranking criterion. See ``rank_markers()``.

    Returns
    -------
    pd.DataFrame
        Top N markers per cluster.

    Examples
    --------
    >>> top = sd.top_markers(all_markers, n=5)
    >>> print(top.groupby('cluster')['gene'].apply(list))
    """
    if "cluster" not in all_markers.columns:
        raise ValueError(
            "DataFrame must have a 'cluster' column. "
            "Use find_all_markers() to get per-cluster results."
        )

    groups = []
    for cluster, df_grp in all_markers.groupby("cluster", sort=True):
        ranked = rank_markers(df_grp, by=by, n_top=n)
        groups.append(ranked)

    return pd.concat(groups, ignore_index=True)


def markers_to_dict(
    all_markers: pd.DataFrame,
    n: Optional[int] = None,
) -> dict:
    """
    Convert find_all_markers() output to a dict of {cluster: [genes]}.

    Parameters
    ----------
    all_markers : pd.DataFrame
        Output from ``find_all_markers()``.
    n : int, optional
        Take top N genes per cluster. If None, take all.

    Returns
    -------
    dict
        Mapping from cluster name to list of marker gene names.

    Examples
    --------
    >>> marker_dict = sd.markers_to_dict(all_markers, n=20)
    >>> sc.pl.dotplot(adata, marker_dict, groupby='leiden')
    """
    if "cluster" not in all_markers.columns:
        raise ValueError("DataFrame must have a 'cluster' column.")

    result = {}
    for cluster, df_grp in all_markers.groupby("cluster", sort=True):
        genes = df_grp["gene"].tolist()
        if n is not None:
            genes = genes[:n]
        result[str(cluster)] = genes

    return result


def store_in_adata(
    adata: AnnData,
    markers: pd.DataFrame,
    groupby: str,
    key: str = "diff_markers",
) -> None:
    """
    Store find_all_markers() results in adata.uns.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix to store results in.
    markers : pd.DataFrame
        Output from ``find_all_markers()``.
    groupby : str
        The groupby column used to compute markers.
    key : str
        Key to store results under in ``adata.uns``.

    Examples
    --------
    >>> sd.store_in_adata(adata, all_markers, groupby='leiden')
    >>> adata.uns['diff_markers']  # Access later
    """
    if "cluster" not in markers.columns:
        # Single-group result from find_markers
        adata.uns[key] = markers
    else:
        # Multi-group result from find_all_markers
        adata.uns[key] = {
            "markers": markers,
            "groupby": groupby,
            "params": {
                "method": markers.attrs.get("method", "unknown"),
            },
        }

    print(f"[scanpy_diff] Results stored in adata.uns['{key}'].")
