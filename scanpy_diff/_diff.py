"""
Core differential expression analysis functions.

This module provides find_markers() and find_all_markers() as the primary
user-facing API, mirroring Seurat's FindMarkers() and FindAllMarkers().
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._stats import (
    _to_linear_scale,
    adjust_pvalues,
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
) -> Union[np.ndarray, sparse.spmatrix]:
    """Extract the expression matrix, keeping it sparse if it is sparse."""
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
    else:
        X = adata.X

    return X


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
    replicate_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
    min_pct: float = 0.1,
    min_pct_reference: float = 0.0,
    logfc_threshold: float = 0.25,
    pval_cutoff: float = 1.0,
    padj_cutoff: float = 1.0,
    only_positive: bool = False,
    correction_method: Literal["fdr_bh", "bonferroni", "fdr_by", "holm"] = "fdr_bh",
    correction_scope: Literal["tested", "all_genes"] = "all_genes",
    expression_scale: Literal["log", "raw", "linear"] = "log",
    log_base: Optional[float] = None,
    tie_correct: bool = True,
    use_raw: bool = False,
    verbose: bool = True,
    _precomputed_stats: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Find markers for a group vs. a reference group.

    This is the primary differential expression function, analogous to
    Seurat's ``FindMarkers()``. It compares expression in ``group`` against
    ``reference`` (or all other cells if ``reference='rest'``).
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    group_str = _validate_group(adata, groupby, group)

    if not tie_correct:
        warnings.warn(
            "tie_correct=False is not yet implemented and has no effect. "
            "The parameter is reserved for future use.",
            FutureWarning,
            stacklevel=2,
        )

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

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 3. Get expression matrices
    # ------------------------------------------------------------------
    if _precomputed_stats is not None:
        X_full = _precomputed_stats["X_full"]
    else:
        if use_raw:
            adata_sub = adata.raw.to_adata()
            X_full = _get_expression_matrix(adata_sub, layer=None)
        else:
            X_full = _get_expression_matrix(adata, layer=layer)

    gene_names = (
        adata.raw.var_names.tolist() if use_raw else adata.var_names.tolist()
    )
    n_total_genes = len(gene_names)

    # ------------------------------------------------------------------
    # 4. Pre-filtering: pct and logFC
    # ------------------------------------------------------------------
    if _precomputed_stats is not None:
        pct1 = _precomputed_stats["pcts"][group_str]
        all_groups = list(_precomputed_stats["n_cells"].keys())

        if reference == "rest":
            other_groups = [g for g in all_groups if g != group_str]
            total_ref_cells = sum(_precomputed_stats["n_cells"][g] for g in other_groups)
            if total_ref_cells > 0:
                sum_pct = sum(_precomputed_stats["n_cells"][g] * _precomputed_stats["pcts"][g] for g in other_groups)
                pct2 = sum_pct / total_ref_cells
                sum_mean = sum(_precomputed_stats["n_cells"][g] * _precomputed_stats["means"][g] for g in other_groups)
                mean2 = sum_mean / total_ref_cells
            else:
                pct2 = np.zeros(n_total_genes)
                mean2 = np.zeros(n_total_genes)
        else:
            pct2 = _precomputed_stats["pcts"][ref_name]
            mean2 = _precomputed_stats["means"][ref_name]

        mean1 = _precomputed_stats["means"][group_str]
        log2fc_all = np.log2(mean1 + 1.0) - np.log2(mean2 + 1.0)
    else:
        X_group = X_full[mask_group, :]
        X_rest = X_full[mask_ref, :]

        pct1 = compute_pct(X_group)
        pct2 = compute_pct(X_rest)

        X_group_lin = _to_linear_scale(X_group, scale=expression_scale, log_base=log_base)
        X_rest_lin = _to_linear_scale(X_rest, scale=expression_scale, log_base=log_base)
        if sparse.issparse(X_group_lin):
            mean1 = np.asarray(X_group_lin.mean(axis=0)).flatten()
        else:
            mean1 = X_group_lin.mean(axis=0)

        if sparse.issparse(X_rest_lin):
            mean2 = np.asarray(X_rest_lin.mean(axis=0)).flatten()
        else:
            mean2 = X_rest_lin.mean(axis=0)

        log2fc_all = np.log2(mean1 + 1.0) - np.log2(mean2 + 1.0)

    # Gene filter mask
    n_before = n_total_genes
    gene_mask = np.ones(n_total_genes, dtype=bool)

    if min_pct > 0:
        gene_mask &= (pct1 >= min_pct) | (pct2 >= min_pct)
    n_after_pct = gene_mask.sum()

    if min_pct_reference > 0:
        gene_mask &= pct2 >= min_pct_reference
    n_after_ref_pct = gene_mask.sum()

    if logfc_threshold > 0:
        gene_mask &= np.abs(log2fc_all) >= logfc_threshold
    n_tested = gene_mask.sum()

    if verbose:
        excluded_pct = n_before - n_after_pct
        excluded_ref = n_after_pct - n_after_ref_pct
        excluded_lfc = n_after_ref_pct - n_tested
        parts = []
        if excluded_pct:
            parts.append(f"{excluded_pct} by min_pct<{min_pct}")
        if excluded_ref:
            parts.append(f"{excluded_ref} by min_pct_reference<{min_pct_reference}")
        if excluded_lfc:
            parts.append(f"{excluded_lfc} by |log2fc|<{logfc_threshold}")
        if parts:
            print(f"[scanpy_diff] Pre-filter: excluded {', '.join(parts)}")
        print(
            f"[scanpy_diff] Testing {n_tested}/{n_before} genes "
            f"(after pre-filtering)"
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

    # ------------------------------------------------------------------
    # 5. Run statistical test
    # ------------------------------------------------------------------
    if verbose:
        print(f"[scanpy_diff] Running {method} on {n_tested} genes ...")

    if method == "deseq2":
        from ._stats import deseq2_test

        if verbose:
            print("  [deseq2] fitting pseudo-bulk model ...")

        if replicate_col is None:
            raise ValueError(
                "replicate_col must be provided for DESeq2 to aggregate cells into pseudo-bulk samples."
            )

        scores, pvals = deseq2_test(
            X_group_or_adata=adata,
            groupby=groupby,
            group=group_str,
            reference=ref_name,
            replicate_col=replicate_col,
            covariates=covariates,
            layer=layer,
            use_raw=use_raw,
            gene_indices=gene_indices,
        )
    else:
        X_group_sub = X_full[mask_group, :][:, gene_indices]
        X_rest_sub = X_full[mask_ref, :][:, gene_indices]

        if method == "wilcoxon":
            scores, pvals = wilcoxon_test(X_group_sub, X_rest_sub, verbose=verbose)
        elif method == "t-test":
            scores, pvals = ttest(X_group_sub, X_rest_sub)
        elif method == "logreg":
            scores, pvals = logistic_regression_test(X_group_sub, X_rest_sub, verbose=verbose)
        elif method == "roc":
            scores, pvals = roc_test(X_group_sub, X_rest_sub, verbose=verbose)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: 'wilcoxon', 't-test', 'logreg', 'roc', 'deseq2'."
            )

    # ------------------------------------------------------------------
    # 6. Multiple testing correction
    # ------------------------------------------------------------------
    if correction_scope == "all_genes":
        full_pvals = np.ones(n_total_genes)
        full_pvals[gene_indices] = pvals
        full_padj = adjust_pvalues(full_pvals, method=correction_method)
        padj = full_padj[gene_indices]
    else:
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
    result.attrs["correction_scope"] = correction_scope
    result.attrs["min_pct"] = min_pct
    result.attrs["min_pct_reference"] = min_pct_reference
    result.attrs["logfc_threshold"] = logfc_threshold

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(
            f"[scanpy_diff] Found {len(result)} significant markers "
            f"({elapsed:.1f}s)"
        )

    return result


def find_all_markers(
    adata: AnnData,
    groupby: str,
    reference: Union[str, int, Literal["rest"]] = "rest",
    method: TestMethod = "wilcoxon",
    layer: Optional[str] = None,
    replicate_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    n_genes_per_group: Optional[int] = None,
    min_pct: float = 0.1,
    min_pct_reference: float = 0.0,
    logfc_threshold: float = 0.25,
    pval_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
    only_positive: bool = True,
    correction_method: Literal["fdr_bh", "bonferroni", "fdr_by", "holm"] = "fdr_bh",
    correction_scope: Literal["tested", "all_genes"] = "all_genes",
    expression_scale: Literal["log", "raw", "linear"] = "log",
    log_base: Optional[float] = None,
    use_raw: bool = False,
    groups: Optional[List[Union[str, int]]] = None,
    ignore_failures: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Find markers for all groups against a reference.

    Analogous to Seurat's ``FindAllMarkers()``. Runs ``find_markers()``
    for each group and combines the results.
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

    t_start_all = time.perf_counter()

    if verbose:
        print(
            f"[scanpy_diff] Running find_all_markers for {len(test_groups)} groups "
            f"using '{groupby}' | method={method}"
        )

    # ------------------------------------------------------------------
    # Precompute statistics for all groups if method is NOT deseq2 (as deseq2 needs raw counts and design)
    # ------------------------------------------------------------------
    precomputed_stats = None
    if method != "deseq2":
        if verbose:
            print("[scanpy_diff] Precomputing group statistics for acceleration ...")
        labels = adata.obs[groupby].astype(str).values
        if use_raw:
            adata_sub = adata.raw.to_adata()
            X_full = _get_expression_matrix(adata_sub, layer=None)
        else:
            X_full = _get_expression_matrix(adata, layer=layer)

        means = {}
        pcts = {}
        n_cells = {}

        for grp in all_groups:
            mask = labels == grp
            n_cells[grp] = int(mask.sum())
            if n_cells[grp] > 0:
                X_grp = X_full[mask, :]
                pcts[grp] = compute_pct(X_grp, threshold=0.0)

                X_grp_lin = _to_linear_scale(X_grp, scale=expression_scale, log_base=log_base)
                if sparse.issparse(X_grp_lin):
                    means[grp] = np.asarray(X_grp_lin.mean(axis=0)).flatten()
                else:
                    means[grp] = X_grp_lin.mean(axis=0)
            else:
                n_genes = X_full.shape[1]
                pcts[grp] = np.zeros(n_genes)
                means[grp] = np.zeros(n_genes)

        precomputed_stats = {
            "means": means,
            "pcts": pcts,
            "n_cells": n_cells,
            "X_full": X_full,
        }

    results_list = []
    failures = {}

    for i, grp in enumerate(test_groups):
        t_group = time.perf_counter()

        if verbose:
            print(
                f"[scanpy_diff] [{i+1}/{len(test_groups)}] Testing group '{grp}' ..."
            )

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
                replicate_col=replicate_col,
                covariates=covariates,
                n_genes=n_genes_per_group,
                min_pct=min_pct,
                min_pct_reference=min_pct_reference,
                logfc_threshold=logfc_threshold,
                pval_cutoff=pval_cutoff,
                padj_cutoff=padj_cutoff,
                only_positive=only_positive,
                correction_method=correction_method,
                correction_scope=correction_scope,
                expression_scale=expression_scale,
                log_base=log_base,
                use_raw=use_raw,
                verbose=False,  # Suppress per-group verbosity
                _precomputed_stats=precomputed_stats,
            )
            df.insert(0, "cluster", grp)
            results_list.append(df)

            if verbose:
                elapsed = time.perf_counter() - t_group
                print(f"  → {len(df)} markers found ({elapsed:.1f}s)")

        except Exception as e:
            if not ignore_failures:
                raise e
            failures[grp] = str(e)
            if verbose:
                elapsed = time.perf_counter() - t_group
                print(f"  → error after {elapsed:.1f}s: {e}")
            warnings.warn(
                f"Error testing group '{grp}': {e}. Skipping.",
                UserWarning,
            )
            continue

    if failures:
        print(f"[scanpy_diff] Warning: differential expression failed for {len(failures)} groups:")
        for failed_grp, err in failures.items():
            print(f"  - Group '{failed_grp}': {err}")

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
        elapsed = time.perf_counter() - t_start_all
        print(
            f"[scanpy_diff] Done. Found {total} markers across {n_groups} groups "
            f"({elapsed:.1f}s total)"
        )

    return result
