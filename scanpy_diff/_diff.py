"""
Core differential expression analysis functions.

This module provides find_markers() and find_all_markers() as the primary
user-facing API, mirroring Seurat's FindMarkers() and FindAllMarkers().
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._stats import (
    adjust_pvalues,
    compute_log2fc,
    compute_pct,
    logistic_regression_test,
    roc_test,
    ttest,
    wilcoxon_test,
)

logger = logging.getLogger(__name__)

# Type alias for test methods
TestMethod = Literal["wilcoxon", "t-test", "logreg", "roc", "deseq2"]


def _get_expression_matrix(
    adata: AnnData,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Extract the expression matrix, ensuring it is dense."""
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    return np.asarray(X, dtype=np.float64)


def _validate_group(
    adata: AnnData,
    groupby: str,
    group: Union[str, int],
) -> str:
    """Validate that a group exists in adata.obs[groupby]."""
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"'{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    categories = adata.obs[groupby].astype(str).unique().tolist()
    group_str = str(group)
    if group_str not in categories:
        raise ValueError(
            f"Group '{group}' not found in adata.obs['{groupby}']. "
            f"Available groups: {categories}"
        )
    return group_str


def find_markers(
    adata: AnnData,
    groupby: str,
    group: Union[str, int],
    reference: Union[str, int, Literal["rest"]] = "rest",
    method: TestMethod = "wilcoxon",
    layer: Optional[str] = None,
    n_genes: Optional[int] = None,
    min_pct: float = 0.1,
    min_pct_reference: float = 0.0,
    logfc_threshold: float = 0.25,
    pval_cutoff: float = 1.0,
    padj_cutoff: float = 1.0,
    only_positive: bool = False,
    correction_method: Literal["fdr_bh", "bonferroni", "fdr_by", "holm"] = "fdr_bh",
    tie_correct: bool = True,
    use_raw: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Find markers for a group vs. a reference group.

    This is the primary differential expression function, analogous to
    Seurat's ``FindMarkers()``. It compares expression in ``group`` against
    ``reference`` (or all other cells if ``reference='rest'``).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in ``adata.obs`` containing cell group labels.
    group : str or int
        The group (cluster) to find markers for.
    reference : str, int, or 'rest'
        Reference group to compare against. Use ``'rest'`` (default) to
        compare against all other cells.
    method : {'wilcoxon', 't-test', 'logreg', 'roc', 'deseq2'}
        Statistical test to use:

        - ``'wilcoxon'`` : Wilcoxon rank-sum test (default, most robust).
        - ``'t-test'``   : Welch's t-test (faster, assumes normality).
        - ``'logreg'``   : Logistic regression likelihood-ratio test.
        - ``'roc'``      : ROC AUC analysis.
        - ``'deseq2'``   : DESeq2 negative binomial (requires pydeseq2,
                           use with raw counts).
    layer : str, optional
        Layer to use for expression values. Defaults to ``adata.X``.
    n_genes : int, optional
        Maximum number of top genes to return. Returns all if None.
    min_pct : float
        Minimum fraction of cells in ``group`` that must express a gene
        for it to be tested. Genes with pct.1 < min_pct are skipped.
    min_pct_reference : float
        Minimum fraction of cells in reference that must express a gene.
        Genes with pct.2 < min_pct_reference are skipped.
    logfc_threshold : float
        Minimum absolute log2 fold change threshold. Genes below this
        threshold are not tested (pre-filter for speed).
    pval_cutoff : float
        Filter result to genes with raw p-value <= pval_cutoff.
    padj_cutoff : float
        Filter result to genes with adjusted p-value <= padj_cutoff.
    only_positive : bool
        If True, only return upregulated markers (log2FC > 0).
    correction_method : str
        Multiple testing correction method. One of 'fdr_bh' (Benjamini-
        Hochberg, default), 'bonferroni', 'fdr_by', 'holm'.
    tie_correct : bool
        Not used (kept for API compatibility).
    use_raw : bool
        If True, use ``adata.raw`` for expression values.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:

        - ``gene``        : Gene name.
        - ``scores``      : Test statistic or AUC.
        - ``log2fc``      : Average log2 fold change.
        - ``pct_1``       : Fraction of cells expressing gene in group.
        - ``pct_2``       : Fraction of cells expressing gene in reference.
        - ``pval``        : Raw p-value.
        - ``padj``        : Adjusted p-value.

        Sorted by ``log2fc`` (descending) then ``padj`` (ascending).

    Examples
    --------
    >>> import scanpy as sc
    >>> import scanpy_diff as sd
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> markers = sd.find_markers(adata, groupby='louvain', group='0')
    >>> markers.head(10)

    >>> # Compare cluster 0 vs cluster 1
    >>> markers = sd.find_markers(
    ...     adata, groupby='louvain', group='0', reference='1'
    ... )

    >>> # Use t-test with stricter thresholds
    >>> markers = sd.find_markers(
    ...     adata, groupby='louvain', group='0',
    ...     method='t-test', min_pct=0.25, logfc_threshold=0.5
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    group_str = _validate_group(adata, groupby, group)

    if use_raw and adata.raw is None:
        warnings.warn(
            "use_raw=True but adata.raw is None. Using adata.X instead.",
            UserWarning,
        )
        use_raw = False

    # Get labels as string
    labels = adata.obs[groupby].astype(str).values

    # ------------------------------------------------------------------
    # 2. Subset cells
    # ------------------------------------------------------------------
    mask_group = labels == group_str

    if reference == "rest":
        mask_ref = ~mask_group
        ref_name = "rest"
    else:
        ref_str = _validate_group(adata, groupby, reference)
        mask_ref = labels == ref_str
        ref_name = ref_str

    n_cells_group = mask_group.sum()
    n_cells_ref = mask_ref.sum()

    if n_cells_group == 0:
        raise ValueError(f"No cells found for group '{group}'.")
    if n_cells_ref == 0:
        raise ValueError(f"No cells found for reference '{reference}'.")

    if verbose:
        logger.info(
            f"Testing group '{group_str}' (n={n_cells_group}) vs "
            f"'{ref_name}' (n={n_cells_ref}) using method='{method}'"
        )
        print(
            f"[scanpy_diff] '{group_str}' (n={n_cells_group}) vs "
            f"'{ref_name}' (n={n_cells_ref}) | method={method}"
        )

    # ------------------------------------------------------------------
    # 3. Get expression matrices
    # ------------------------------------------------------------------
    if use_raw:
        adata_sub = adata.raw.to_adata()
        X_full = _get_expression_matrix(adata_sub, layer=None)
    else:
        X_full = _get_expression_matrix(adata, layer=layer)

    X_group = X_full[mask_group, :]
    X_rest = X_full[mask_ref, :]

    gene_names = (
        adata.raw.var_names.tolist() if use_raw else adata.var_names.tolist()
    )
    n_total_genes = len(gene_names)

    # ------------------------------------------------------------------
    # 4. Pre-filtering: pct and logFC
    # ------------------------------------------------------------------
    pct1 = compute_pct(X_group)
    pct2 = compute_pct(X_rest)
    log2fc_all = compute_log2fc(X_group, X_rest)

    # Gene filter mask
    gene_mask = np.ones(n_total_genes, dtype=bool)

    if min_pct > 0:
        gene_mask &= (pct1 >= min_pct) | (pct2 >= min_pct)

    if min_pct_reference > 0:
        gene_mask &= pct2 >= min_pct_reference

    if logfc_threshold > 0:
        gene_mask &= np.abs(log2fc_all) >= logfc_threshold

    n_tested = gene_mask.sum()
    if verbose:
        print(
            f"[scanpy_diff] Testing {n_tested}/{n_total_genes} genes "
            f"(after pct/logFC pre-filtering)"
        )

    if n_tested == 0:
        warnings.warn(
            "No genes passed pre-filtering thresholds. "
            "Consider relaxing min_pct or logfc_threshold.",
            UserWarning,
        )
        return pd.DataFrame(
            columns=["gene", "scores", "log2fc", "pct_1", "pct_2", "pval", "padj"]
        )

    # Subset to tested genes
    gene_indices = np.where(gene_mask)[0]
    X_group_sub = X_group[:, gene_indices]
    X_rest_sub = X_rest[:, gene_indices]

    # ------------------------------------------------------------------
    # 5. Run statistical test
    # ------------------------------------------------------------------
    if method == "wilcoxon":
        scores, pvals = wilcoxon_test(X_group_sub, X_rest_sub)
    elif method == "t-test":
        scores, pvals = ttest(X_group_sub, X_rest_sub)
    elif method == "logreg":
        scores, pvals = logistic_regression_test(X_group_sub, X_rest_sub)
    elif method == "roc":
        scores, pvals = roc_test(X_group_sub, X_rest_sub)
    elif method == "deseq2":
        from ._stats import deseq2_test

        scores, pvals = deseq2_test(X_group_sub, X_rest_sub)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'wilcoxon', 't-test', 'logreg', 'roc', 'deseq2'."
        )

    # ------------------------------------------------------------------
    # 6. Multiple testing correction
    # ------------------------------------------------------------------
    padj = adjust_pvalues(pvals, method=correction_method)

    # ------------------------------------------------------------------
    # 7. Build result DataFrame
    # ------------------------------------------------------------------
    result = pd.DataFrame(
        {
            "gene": [gene_names[i] for i in gene_indices],
            "scores": scores,
            "log2fc": log2fc_all[gene_indices],
            "pct_1": np.round(pct1[gene_indices], 3),
            "pct_2": np.round(pct2[gene_indices], 3),
            "pval": pvals,
            "padj": padj,
        }
    )

    # ------------------------------------------------------------------
    # 8. Post-filtering
    # ------------------------------------------------------------------
    if pval_cutoff < 1.0:
        result = result[result["pval"] <= pval_cutoff]
    if padj_cutoff < 1.0:
        result = result[result["padj"] <= padj_cutoff]
    if only_positive:
        result = result[result["log2fc"] > 0]

    # Sort: primary by log2fc descending, secondary by padj ascending
    result = result.sort_values(
        ["log2fc", "padj"], ascending=[False, True]
    ).reset_index(drop=True)

    # Optionally limit to top N genes
    if n_genes is not None:
        result = result.head(n_genes)

    # Metadata
    result.attrs["group"] = group_str
    result.attrs["reference"] = ref_name
    result.attrs["method"] = method
    result.attrs["n_cells_group"] = int(n_cells_group)
    result.attrs["n_cells_reference"] = int(n_cells_ref)

    if verbose:
        print(f"[scanpy_diff] Found {len(result)} significant markers.")

    return result


def find_all_markers(
    adata: AnnData,
    groupby: str,
    reference: Union[str, int, Literal["rest"]] = "rest",
    method: TestMethod = "wilcoxon",
    layer: Optional[str] = None,
    n_genes_per_group: Optional[int] = None,
    min_pct: float = 0.1,
    min_pct_reference: float = 0.0,
    logfc_threshold: float = 0.25,
    pval_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
    only_positive: bool = True,
    correction_method: Literal["fdr_bh", "bonferroni", "fdr_by", "holm"] = "fdr_bh",
    use_raw: bool = False,
    groups: Optional[List[Union[str, int]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Find markers for all groups against a reference.

    Analogous to Seurat's ``FindAllMarkers()``. Runs ``find_markers()``
    for each group and combines the results.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in ``adata.obs`` containing cell group labels.
    reference : str, int, or 'rest'
        Reference group. Default is ``'rest'`` (all other cells).
    method : TestMethod
        Statistical test. See ``find_markers()`` for options.
    layer : str, optional
        Expression layer to use. Defaults to ``adata.X``.
    n_genes_per_group : int, optional
        Maximum number of top marker genes to return per group.
    min_pct : float
        Minimum expression fraction threshold.
    min_pct_reference : float
        Minimum expression fraction in reference threshold.
    logfc_threshold : float
        Minimum log2 fold change threshold for pre-filtering.
    pval_cutoff : float
        Maximum raw p-value cutoff.
    padj_cutoff : float
        Maximum adjusted p-value cutoff.
    only_positive : bool
        If True (default), only return upregulated markers.
    correction_method : str
        Multiple testing correction method.
    use_raw : bool
        Use ``adata.raw`` if True.
    groups : list, optional
        Subset of groups to test. Defaults to all groups.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an extra ``'cluster'`` column indicating
        the source group. Sorted by cluster, then log2fc descending.

    Examples
    --------
    >>> import scanpy as sc
    >>> import scanpy_diff as sd
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> all_markers = sd.find_all_markers(adata, groupby='louvain')
    >>> all_markers.groupby('cluster').head(5)

    >>> # Top 10 markers per cluster
    >>> all_markers = sd.find_all_markers(
    ...     adata, groupby='louvain', n_genes_per_group=10
    ... )
    """
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"'{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    all_groups = adata.obs[groupby].astype(str).unique().tolist()
    all_groups = sorted(all_groups)

    if groups is not None:
        groups_str = [str(g) for g in groups]
        invalid = [g for g in groups_str if g not in all_groups]
        if invalid:
            raise ValueError(
                f"Groups {invalid} not found in adata.obs['{groupby}']. "
                f"Available: {all_groups}"
            )
        test_groups = groups_str
    else:
        test_groups = all_groups

    if verbose:
        print(
            f"[scanpy_diff] Running find_all_markers for {len(test_groups)} groups "
            f"using '{groupby}' | method={method}"
        )

    results_list = []

    for i, grp in enumerate(test_groups):
        if verbose:
            print(
                f"[scanpy_diff] [{i+1}/{len(test_groups)}] Testing group '{grp}' ..."
            )

        # Skip if this group IS the reference
        if reference != "rest" and str(reference) == grp:
            if verbose:
                print(f"  Skipping group '{grp}' (same as reference).")
            continue

        try:
            df = find_markers(
                adata=adata,
                groupby=groupby,
                group=grp,
                reference=reference,
                method=method,
                layer=layer,
                n_genes=n_genes_per_group,
                min_pct=min_pct,
                min_pct_reference=min_pct_reference,
                logfc_threshold=logfc_threshold,
                pval_cutoff=pval_cutoff,
                padj_cutoff=padj_cutoff,
                only_positive=only_positive,
                correction_method=correction_method,
                use_raw=use_raw,
                verbose=False,  # Suppress per-group verbosity
            )
            df.insert(0, "cluster", grp)
            results_list.append(df)

            if verbose:
                print(f"  → {len(df)} markers found.")

        except Exception as e:
            warnings.warn(
                f"Error testing group '{grp}': {e}. Skipping.",
                UserWarning,
            )
            continue

    if not results_list:
        warnings.warn("No markers found for any group.", UserWarning)
        return pd.DataFrame(
            columns=[
                "cluster",
                "gene",
                "scores",
                "log2fc",
                "pct_1",
                "pct_2",
                "pval",
                "padj",
            ]
        )

    result = pd.concat(results_list, ignore_index=True)

    if verbose:
        total = len(result)
        n_groups = result["cluster"].nunique()
        print(
            f"[scanpy_diff] Done. Found {total} markers across {n_groups} groups."
        )

    return result
