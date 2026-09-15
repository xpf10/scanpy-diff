"""
Statistical test implementations for differential expression analysis.

Each test function takes:
    X_group : np.ndarray  - expression matrix for group of interest (cells x genes)
    X_rest  : np.ndarray  - expression matrix for reference group (cells x genes)

And returns:
    scores  : np.ndarray  - test statistic or score per gene
    pvals   : np.ndarray  - raw p-values per gene
"""

from __future__ import annotations

import inspect
import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse, stats
from scipy.special import xlogy

# ---------------------------------------------------------------------------
# Wilcoxon rank-sum test (default, equivalent to Seurat's "wilcox")
# ---------------------------------------------------------------------------


def wilcoxon_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
    block_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wilcoxon rank-sum (Mann-Whitney U) test for each gene.

    Parameters
    ----------
    X_group : np.ndarray | sparse.spmatrix
        Expression matrix for the group of interest (n_cells_group x n_genes).
    X_rest : np.ndarray | sparse.spmatrix
        Expression matrix for the reference group (n_cells_rest x n_genes).
    verbose : bool
        Print progress.
    block_size : int
        Number of genes to process in a block to prevent memory issues.

    Returns
    -------
    scores : np.ndarray
        U-statistics normalized to [0,1] (AUC-like, 1 = perfect separation).
    pvals : np.ndarray
        Two-sided p-values.
    """
    n_genes = X_group.shape[1]
    scores = np.zeros(n_genes)
    pvals = np.ones(n_genes)

    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]

    for start in range(0, n_genes, block_size):
        end = min(start + block_size, n_genes)
        g_block = X_group[:, start:end]
        r_block = X_rest[:, start:end]
        if sparse.issparse(g_block):
            g_block = g_block.toarray()
        if sparse.issparse(r_block):
            r_block = r_block.toarray()

        if g_block.shape[1] == 0:
            continue

        # Use scipy.stats.mannwhitneyu vectorized over axis 0
        stat, pval = stats.mannwhitneyu(g_block, r_block, alternative="two-sided", axis=0)

        # Identify constant columns (where all values in g_block are equal,
        # all values in r_block are equal, and the values are equal)
        is_g_const = np.all(g_block == g_block[0:1, :], axis=0)
        is_r_const = np.all(r_block == r_block[0:1, :], axis=0)
        is_both_const_and_equal = is_g_const & is_r_const & (g_block[0, :] == r_block[0, :])

        stat = np.where(is_both_const_and_equal, n1 * n2 / 2, stat)
        pval = np.where(is_both_const_and_equal, 1.0, pval)

        scores[start:end] = stat / (n1 * n2)
        pvals[start:end] = pval

        if verbose:
            pct = end / n_genes * 100
            print(f"  [wilcoxon] processed {end}/{n_genes} genes ({pct:.0f}%)")

    return scores, pvals


# ---------------------------------------------------------------------------
# Student's t-test (equivalent to Seurat's "t")
# ---------------------------------------------------------------------------


def ttest(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch's two-sample t-test for each gene.

    Parameters
    ----------
    X_group : np.ndarray | sparse.spmatrix
        Expression matrix for the group of interest.
    X_rest : np.ndarray | sparse.spmatrix
        Expression matrix for the reference group.

    Returns
    -------
    scores : np.ndarray
        t-statistics.
    pvals : np.ndarray
        Two-sided p-values.
    """
    n_genes = X_group.shape[1]
    scores = np.zeros(n_genes)
    pvals = np.ones(n_genes)

    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]

    mean1, var1 = _mean_var(X_group, axis=0, ddof=1)
    mean2, var2 = _mean_var(X_rest, axis=0, ddof=1)

    var1 = var1 + 1e-9
    var2 = var2 + 1e-9

    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / (df_den + 1e-15)

    pvals = 2 * stats.t.sf(np.abs(t_stat), df=df)
    scores = t_stat

    return scores, pvals


# ---------------------------------------------------------------------------
# Logistic regression test (equivalent to Seurat's "LR")
# ---------------------------------------------------------------------------


def logistic_regression_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logistic regression likelihood-ratio test for each gene.

    Uses a logistic regression model to predict group membership from gene
    expression. The LR test statistic compares the model with the gene vs
    a null model.

    Parameters
    ----------
    X_group : np.ndarray | sparse.spmatrix
        Expression matrix for the group of interest.
    X_rest : np.ndarray | sparse.spmatrix
        Expression matrix for the reference group.
    max_iter : int
        Maximum iterations for logistic regression solver.
    verbose : bool
        Print progress.

    Returns
    -------
    scores : np.ndarray
        Log-likelihood ratio test statistics (chi-squared distributed).
    pvals : np.ndarray
        p-values from chi-squared distribution with 1 degree of freedom.
    """
    from sklearn.linear_model import LogisticRegression

    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n_genes = X_group.shape[1]

    if sparse.issparse(X_group) or sparse.issparse(X_rest):
        X = sparse.vstack([X_group, X_rest])
    else:
        X = np.vstack([X_group, X_rest])
    y = np.array([1] * n1 + [0] * n2)

    scores = np.zeros(n_genes)
    pvals = np.ones(n_genes)

    # Null model log-likelihood (intercept only)
    p_null = n1 / (n1 + n2)
    ll_null = n1 * np.log(p_null + 1e-15) + n2 * np.log(1 - p_null + 1e-15)

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        random_state=0,
    )

    report_step = max(1, min(n_genes // 10, 500))

    for i in range(n_genes):
        xi = X[:, i : i + 1]
        if sparse.issparse(xi):
            xi = xi.toarray()
        try:
            lr.fit(xi, y)
            probs = lr.predict_proba(xi)[:, 1]
            ll_full = np.sum(
                xlogy(y, probs + 1e-15) + xlogy(1 - y, 1 - probs + 1e-15)
            )
            lr_stat = 2 * (ll_full - ll_null)
            scores[i] = lr_stat
            pvals[i] = stats.chi2.sf(lr_stat, df=1)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            scores[i] = 0.0
            pvals[i] = 1.0

        if verbose and (i + 1) % report_step == 0:
            pct = (i + 1) / n_genes * 100
            print(f"  [logreg] {i+1}/{n_genes} genes ({pct:.0f}%)")

    return scores, pvals


# ---------------------------------------------------------------------------
# DESeq2-style negative binomial test (pseudo-bulk, requires pydeseq2)
# ---------------------------------------------------------------------------


def deseq2_test(
    adata: AnnData,
    groupby: str,
    group: str,
    reference: str,
    replicate_col: str,
    covariates: Optional[List[str]] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    gene_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DESeq2 negative binomial test on pseudo-bulk samples aggregated from cells.

    Cells are summed into one pseudo-bulk sample per
    ``(condition, replicate_col, *covariates)`` combination. Testing replicates
    rather than cells is what makes the p-values calibrated: a per-cell
    negative binomial fit treats cells from the same donor as independent
    observations and massively overstates significance.

    Parameters
    ----------
    adata : AnnData
        Unsubset object; the cells to use are selected here from ``groupby``.
    groupby : str
        Column in ``adata.obs`` holding the group labels.
    group : str
        Label of the group of interest.
    reference : str
        Label of the reference group, or ``"rest"`` to pool every cell outside
        ``group``.
    replicate_col : str
        Column in ``adata.obs`` identifying biological replicates.
    covariates : list of str, optional
        Extra ``adata.obs`` columns added to the design formula. They also take
        part in the pseudo-bulk grouping, so a covariate that varies within a
        replicate splits that replicate instead of being averaged away.
    layer : str, optional
        Layer holding raw counts. Defaults to ``adata.X``.
    use_raw : bool
        Read counts from ``adata.raw`` instead.
    gene_indices : np.ndarray, optional
        Positions of the genes to test; ``None`` tests every gene.
    verbose : bool
        Print progress.

    Returns
    -------
    scores : np.ndarray
        Log2 fold change (group vs reference), one entry per requested gene.
    pvals : np.ndarray
        Wald test p-value, one entry per requested gene. Genes that could not
        be fitted (e.g. zero counts everywhere) get 0.0 and 1.0.

    Notes
    -----
    pydeseq2's independent filtering is disabled: ``find_markers`` applies its
    own multiple-testing correction afterwards, and independent filtering would
    rewrite the p-values of low-mean genes to NaN (here 1.0), making that
    correction needlessly conservative. Cooks outlier filtering is kept.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for the 'deseq2' method. "
            "Install it with: pip install scanpy-diff[pydeseq2]"
        )

    covariates = list(covariates) if covariates else []
    for col in [replicate_col, *covariates]:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # The factor holding group-vs-reference must not clash with a user column
    # that is itself used as the replicate key or a covariate.
    condition_factor = "condition"
    while condition_factor == replicate_col or condition_factor in covariates:
        condition_factor = "_" + condition_factor

    # ------------------------------------------------------------------
    # Select the cells belonging to the two conditions
    # ------------------------------------------------------------------
    labels = adata.obs[groupby].astype(str).values
    mask_group = labels == group
    mask_ref = ~mask_group if reference == "rest" else labels == reference
    cell_mask = mask_group | mask_ref

    if mask_group.sum() == 0:
        raise ValueError(f"No cells found for group '{group}'.")
    if mask_ref.sum() == 0:
        raise ValueError(f"No cells found for reference '{reference}'.")

    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        X = adata.raw.X
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
    else:
        X = adata.X

    n_total_genes = X.shape[1]
    if gene_indices is None:
        gene_indices = np.arange(n_total_genes)
    gene_indices = np.asarray(gene_indices)

    X_sub = X[cell_mask][:, gene_indices]
    X_sub = X_sub.tocsr() if sparse.issparse(X_sub) else sparse.csr_matrix(X_sub)

    # ------------------------------------------------------------------
    # DESeq2 models counts, so reject anything already transformed
    # ------------------------------------------------------------------
    data = X_sub.data
    if data.size and (
        not np.all(np.isfinite(data))
        or np.any(data < 0)
        or np.any(data % 1 != 0)
    ):
        raise ValueError(
            "method='deseq2' requires raw integer counts, but the expression "
            "matrix contains NaN, negative, or non-integer values. Pass "
            "unnormalized counts, e.g. layer='counts' or use_raw=True."
        )

    # ------------------------------------------------------------------
    # Aggregate cells into pseudo-bulk samples
    # ------------------------------------------------------------------
    condition = np.where(mask_group[cell_mask], group, reference)
    keys = adata.obs.loc[cell_mask, [replicate_col, *covariates]].astype(str)
    keys.insert(0, condition_factor, condition)
    key_cols = list(keys.columns)

    grouped = keys.groupby(key_cols, sort=True, observed=True)
    sample_codes = grouped.ngroup().to_numpy()
    sample_meta = grouped.size().rename("n_cells").reset_index()
    sample_meta.index = [f"sample_{i}" for i in range(len(sample_meta))]
    n_samples = len(sample_meta)

    n_per_condition = sample_meta[condition_factor].value_counts()
    for cond in (group, reference):
        if int(n_per_condition.get(cond, 0)) < 2:
            raise ValueError(
                f"DESeq2 needs at least 2 pseudo-bulk samples per condition to "
                f"estimate dispersions, but '{cond}' has "
                f"{int(n_per_condition.get(cond, 0))}. Check that "
                f"replicate_col='{replicate_col}' has enough distinct values."
            )

    if verbose:
        print(
            f"  [deseq2] {int(cell_mask.sum())} cells -> {n_samples} pseudo-bulk "
            f"samples ({int(n_per_condition[group])} vs {int(n_per_condition[reference])})"
        )

    aggregator = sparse.csr_matrix(
        (np.ones(int(cell_mask.sum())), (sample_codes, np.arange(int(cell_mask.sum())))),
        shape=(n_samples, int(cell_mask.sum())),
    )
    bulk = np.rint((aggregator @ X_sub).toarray()).astype(np.int64)

    # pydeseq2 cannot fit a gene that is zero in every sample
    expressed = bulk.sum(axis=0) > 0
    fitted_positions = np.where(expressed)[0]

    scores = np.zeros(len(gene_indices))
    pvals = np.ones(len(gene_indices))

    if fitted_positions.size == 0:
        warnings.warn(
            "All requested genes have zero counts; returning log2fc=0 and pval=1.",
            UserWarning,
        )
        return scores, pvals

    if verbose and fitted_positions.size < len(gene_indices):
        print(
            f"  [deseq2] skipping {len(gene_indices) - fitted_positions.size} "
            f"all-zero genes"
        )

    counts_df = pd.DataFrame(
        bulk[:, fitted_positions],
        index=sample_meta.index,
        columns=[f"g{j}" for j in fitted_positions],
    )
    design_factors = [condition_factor, *covariates]

    # pydeseq2 0.5 replaced design_factors with a formulaic design string, but
    # 0.4.x is the newest release installable on Python 3.9, which this package
    # still supports.
    if "design" in inspect.signature(DeseqDataSet).parameters:
        design_kwargs = {"design": "~" + " + ".join(design_factors)}
    else:
        design_kwargs = {"design_factors": design_factors}

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=sample_meta[design_factors],
        quiet=not verbose,
        **design_kwargs,
    )
    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=[condition_factor, group, reference],
        cooks_filter=True,
        independent_filter=False,
        quiet=not verbose,
    )
    stat_res.summary()

    results = stat_res.results_df.reindex(counts_df.columns)
    scores[fitted_positions] = np.nan_to_num(
        results["log2FoldChange"].to_numpy(dtype=float), nan=0.0
    )
    pvals[fitted_positions] = np.nan_to_num(
        results["pvalue"].to_numpy(dtype=float), nan=1.0
    )

    return scores, pvals


# ---------------------------------------------------------------------------
# ROC AUC analysis (equivalent to Seurat's "roc")
# ---------------------------------------------------------------------------


def roc_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
    block_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROC AUC analysis for each gene. Matches the Wilcoxon test U-statistic mathematically.
    """
    return wilcoxon_test(X_group, X_rest, verbose=verbose, block_size=block_size)


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------


def adjust_pvalues(
    pvals: np.ndarray,
    method: Literal["bonferroni", "fdr_bh", "fdr_by", "holm"] = "fdr_bh",
) -> np.ndarray:
    """
    Apply multiple testing correction to p-values.

    Parameters
    ----------
    pvals : np.ndarray
        Raw p-values.
    method : str
        Correction method. One of 'bonferroni', 'fdr_bh' (Benjamini-Hochberg),
        'fdr_by' (Benjamini-Yekutieli), 'holm'.

    Returns
    -------
    np.ndarray
        Adjusted p-values.
    """
    from statsmodels.stats.multitest import multipletests

    # Replace NaN with 1.0
    pvals_clean = np.where(np.isnan(pvals), 1.0, pvals)
    pvals_clean = np.clip(pvals_clean, 0, 1)

    _, padj, _, _ = multipletests(pvals_clean, alpha=0.05, method=method)
    return padj


# ---------------------------------------------------------------------------
# Helper: compute percent expressed
# ---------------------------------------------------------------------------


def compute_pct(X: np.ndarray | sparse.spmatrix, threshold: float = 0.0) -> np.ndarray:
    """
    Compute fraction of cells expressing each gene above threshold.

    Parameters
    ----------
    X : np.ndarray | sparse.spmatrix
        Expression matrix (cells x genes).
    threshold : float
        Expression threshold. A gene is considered expressed if > threshold.

    Returns
    -------
    np.ndarray
        Fraction of cells expressing each gene (length = n_genes).
    """
    if sparse.issparse(X):
        if threshold == 0.0:
            return np.asarray(X.getnnz(axis=0)).flatten() / X.shape[0]
        else:
            return np.asarray((X > threshold).mean(axis=0)).flatten()
    return (X > threshold).mean(axis=0)


# ---------------------------------------------------------------------------
# Helper: compute log2 fold change
# ---------------------------------------------------------------------------


def compute_log2fc(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    base: float = 2.0,
    pseudocount: float = 1.0,
    expression_scale: Literal["log", "raw", "linear"] = "log",
    log_base: Optional[float] = None,
    mode: Literal["seurat", "scanpy"] = "seurat",
) -> np.ndarray:
    """
    Compute average log fold change between two groups.

    Parameters
    ----------
    X_group : np.ndarray | sparse.spmatrix
        Expression matrix for the group of interest.
    X_rest : np.ndarray | sparse.spmatrix
        Expression matrix for the reference group.
    base : float
        Target log base for log2FC output (default 2.0).
    pseudocount : float
        Pseudocount added before log2 transformation in Seurat mode (default 1.0).
    expression_scale : Literal["log", "raw", "linear"]
        Input scale of expression data (default "log").
    log_base : float, optional
        Base of the log transformation if expression_scale="log" (default e).
    mode : Literal["seurat", "scanpy"]
        Formula mode: "seurat" (expm1 -> mean -> log2(mean+1)) or "scanpy" (mean_log1 - mean_log2).
    """
    if mode == "scanpy":
        if sparse.issparse(X_group):
            m1 = np.asarray(X_group.mean(axis=0)).flatten()
        else:
            m1 = X_group.mean(axis=0)

        if sparse.issparse(X_rest):
            m2 = np.asarray(X_rest.mean(axis=0)).flatten()
        else:
            m2 = X_rest.mean(axis=0)

        if expression_scale == "log":
            eff_base = np.e if log_base is None else log_base
            scale_factor = 1.0 / np.log(2) if eff_base == np.e else np.log(eff_base) / np.log(2)
            return (m1 - m2) * scale_factor
        else:
            return np.log2(m1 + pseudocount) - np.log2(m2 + pseudocount)

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

    log2fc = np.log2(mean1 + pseudocount) - np.log2(mean2 + pseudocount)
    return log2fc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_linear_scale(
    X: np.ndarray | sparse.spmatrix,
    scale: Literal["log", "raw", "linear"] = "log",
    log_base: Optional[float] = None,
) -> np.ndarray | sparse.spmatrix:
    if scale in ("raw", "linear"):
        return X
    elif scale == "log":
        base = np.e if log_base is None else log_base
        if sparse.issparse(X):
            X_lin = X.copy()
            if base == np.e:
                X_lin.data = np.expm1(X_lin.data)
            else:
                X_lin.data = (base ** X_lin.data) - 1
            return X_lin
        else:
            if base == np.e:
                return np.expm1(X)
            else:
                return (base ** X) - 1
    else:
        raise ValueError(f"Unknown expression_scale: {scale}")


def _mean_var(
    X: np.ndarray | sparse.spmatrix,
    axis: int = 0,
    ddof: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[axis]
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=axis)).flatten()
        mean_sq = np.asarray(X.power(2).mean(axis=axis)).flatten()
        factor = n / (n - ddof) if n > ddof else 1.0
        var = (mean_sq - mean**2) * factor
        var = np.clip(var, 0, None)
    else:
        mean = X.mean(axis=axis)
        var = X.var(axis=axis, ddof=ddof)
    return mean, var


# ---------------------------------------------------------------------------
# Bimodal Likelihood Ratio Test (equivalent to Seurat's "bimod")
# ---------------------------------------------------------------------------


def bimod_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    McDavid et al. (2013) likelihood ratio test for single-cell expression (bimod model).
    """
    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n0 = n1 + n2

    if sparse.issparse(X_group):
        k1 = np.asarray((X_group > 0).sum(axis=0)).flatten()
        k2 = np.asarray((X_rest > 0).sum(axis=0)).flatten()
    else:
        k1 = (X_group > 0).sum(axis=0)
        k2 = (X_rest > 0).sum(axis=0)

    k0 = k1 + k2

    p1 = k1 / n1
    p2 = k2 / n2
    p0 = k0 / n0

    ll_bin1 = xlogy(k1, p1 + 1e-15) + xlogy(n1 - k1, 1 - p1 + 1e-15)
    ll_bin2 = xlogy(k2, p2 + 1e-15) + xlogy(n2 - k2, 1 - p2 + 1e-15)
    ll_bin0 = xlogy(k0, p0 + 1e-15) + xlogy(n0 - k0, 1 - p0 + 1e-15)

    lrt_bin = 2 * (ll_bin1 + ll_bin2 - ll_bin0)
    lrt_bin = np.maximum(0, lrt_bin)

    m1, v1 = _mean_var(X_group, axis=0, ddof=1)
    m2, v2 = _mean_var(X_rest, axis=0, ddof=1)
    if sparse.issparse(X_group) or sparse.issparse(X_rest):
        X_pooled = sparse.vstack([X_group, X_rest])
    else:
        X_pooled = np.vstack([X_group, X_rest])
    m0, v0 = _mean_var(X_pooled, axis=0, ddof=1)

    ll_cont1 = -0.5 * k1 * np.log(v1 + 1e-9)
    ll_cont2 = -0.5 * k2 * np.log(v2 + 1e-9)
    ll_cont0 = -0.5 * k0 * np.log(v0 + 1e-9)

    lrt_cont = 2 * (ll_cont1 + ll_cont2 - ll_cont0)
    lrt_cont = np.maximum(0, lrt_cont)

    scores = lrt_bin + lrt_cont
    pvals = stats.chi2.sf(scores, df=2)

    return scores, pvals


# ---------------------------------------------------------------------------
# Poisson GLM Likelihood Ratio Test (equivalent to Seurat's "poisson")
# ---------------------------------------------------------------------------


def poisson_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poisson GLM likelihood ratio test for single-cell count data (equivalent to Seurat's 'poisson').
    """
    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n0 = n1 + n2

    if sparse.issparse(X_group):
        s1 = np.asarray(X_group.sum(axis=0)).flatten()
        s2 = np.asarray(X_rest.sum(axis=0)).flatten()
    else:
        s1 = X_group.sum(axis=0)
        s2 = X_rest.sum(axis=0)

    s0 = s1 + s2

    mu1 = s1 / n1
    mu2 = s2 / n2
    mu0 = s0 / n0

    ll1 = s1 * np.log(mu1 / (mu0 + 1e-15) + 1e-15)
    ll2 = s2 * np.log(mu2 / (mu0 + 1e-15) + 1e-15)

    lrt = 2 * (ll1 + ll2)
    scores = np.maximum(0, lrt)
    pvals = stats.chi2.sf(scores, df=1)

    return scores, pvals


# ---------------------------------------------------------------------------
# Negative Binomial Likelihood Ratio Test (equivalent to Seurat's "negbinom")
# ---------------------------------------------------------------------------


def negbinom_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Negative Binomial GLM likelihood ratio test (equivalent to Seurat's 'negbinom').
    """
    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n0 = n1 + n2

    m1, v1 = _mean_var(X_group, axis=0, ddof=1)
    m2, v2 = _mean_var(X_rest, axis=0, ddof=1)

    if sparse.issparse(X_group) or sparse.issparse(X_rest):
        X_pooled = sparse.vstack([X_group, X_rest])
    else:
        X_pooled = np.vstack([X_group, X_rest])
    m0, v0 = _mean_var(X_pooled, axis=0, ddof=1)

    alpha0 = np.maximum(1e-4, (v0 - m0) / (m0**2 + 1e-9))
    alpha1 = np.maximum(1e-4, (v1 - m1) / (m1**2 + 1e-9))
    alpha2 = np.maximum(1e-4, (v2 - m2) / (m2**2 + 1e-9))

    def nb_ll(n, m, a):
        return n * (xlogy(m, a * m + 1e-15) - (m + 1.0 / a) * np.log(1 + a * m + 1e-15))

    ll1 = nb_ll(n1, m1, alpha1)
    ll2 = nb_ll(n2, m2, alpha2)
    ll0 = nb_ll(n0, m0, alpha0)

    lrt = 2 * (ll1 + ll2 - ll0)
    scores = np.maximum(0, lrt)
    pvals = stats.chi2.sf(scores, df=1)

    return scores, pvals


# ---------------------------------------------------------------------------
# MAST Hurdle Model Test (equivalent to Seurat's "mast")
# ---------------------------------------------------------------------------


def _residual_ss(D: np.ndarray, z: np.ndarray) -> float:
    """Residual sum of squares of ``z`` regressed on design ``D``."""
    coef, *_ = np.linalg.lstsq(D, z, rcond=None)
    resid = z - D @ coef
    return float(resid @ resid)


def _detection_rate(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Fraction of genes detected in each cell."""
    if X.shape[1] == 0:
        return np.zeros(X.shape[0])
    if sparse.issparse(X):
        detected = np.asarray((X != 0).sum(axis=1)).flatten()
    else:
        detected = (np.asarray(X) != 0).sum(axis=1)
    return detected / X.shape[1]


def _logistic_loglik_block(
    Y: np.ndarray,
    D: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Maximum binomial log-likelihood of every column of ``Y`` on design ``D``.

    Fitted by Newton-Raphson. All response vectors share one design matrix,
    which is what makes it worth solving them as a batch instead of looping.
    """
    n_genes = Y.shape[1]
    p = D.shape[1]

    ybar = np.clip(Y.mean(axis=0), 1e-6, 1.0 - 1e-6)
    beta = np.zeros((n_genes, p))
    beta[:, 0] = np.log(ybar / (1.0 - ybar))

    ridge = np.eye(p) * 1e-8

    for _ in range(max_iter):
        eta = np.clip(D @ beta.T, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-10)

        grad = D.T @ (Y - mu)
        info = np.einsum("ip,ig,iq->gpq", D, w, D) + ridge
        try:
            delta = np.linalg.solve(info, grad.T[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break

        beta += delta
        if np.max(np.abs(delta)) < tol:
            break

    eta = np.clip(D @ beta.T, -30.0, 30.0)
    # log(1 + exp(eta)) evaluated so that large |eta| does not overflow
    log_norm = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    return np.sum(Y * eta - log_norm, axis=0)


def mast_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    cdr_group: Optional[np.ndarray] = None,
    cdr_rest: Optional[np.ndarray] = None,
    verbose: bool = False,
    block_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAST two-part hurdle model test (Finak et al., 2015; Seurat ``test="MAST"``).

    Each gene gets two regressions over cells that share one design matrix:

    - a **discrete** logistic regression of the detection indicator ``x > 0``
      on condition + cellular detection rate (CDR);
    - a **continuous** Gaussian regression of expression on the same design,
      fitted only over the cells where the gene is detected.

    The two likelihood-ratio statistics are summed into a chi-square statistic.
    Conditioning on detection is the point of the hurdle model: it separates
    "more cells express this gene" from "expressing cells express more of it".
    CDR is what absorbs library-level technical variation, and it is the reason
    MAST is not interchangeable with the closed-form ``bimod`` test.

    Parameters
    ----------
    X_group, X_rest : np.ndarray | sparse.spmatrix
        Expression matrices (cells x genes) for the group and reference. Values
        should be on a log scale, since the continuous component models them
        with a Gaussian.
    cdr_group, cdr_rest : np.ndarray, optional
        Per-cell detection rate (fraction of genes detected) computed over the
        **full, unfiltered** gene set. Pass these from the caller so the
        covariate is not distorted by upstream gene pre-filtering; when omitted
        they are computed from the matrices given here.
    verbose : bool
        Print progress.
    block_size : int
        Genes processed at a time, to bound memory.

    Returns
    -------
    scores : np.ndarray
        Combined chi-square likelihood-ratio statistic per gene.
    pvals : np.ndarray
        p-values from ``chi2`` with 2 degrees of freedom, reduced to 1 (or 0)
        for genes where one component is degenerate.
    """
    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n_cells = n1 + n2
    n_genes = X_group.shape[1]

    if sparse.issparse(X_group) or sparse.issparse(X_rest):
        X_all = sparse.vstack([X_group, X_rest]).tocsr()
    else:
        X_all = np.vstack([X_group, X_rest])

    if cdr_group is None or cdr_rest is None:
        cdr_group = _detection_rate(X_group)
        cdr_rest = _detection_rate(X_rest)
    cdr = np.concatenate([np.asarray(cdr_group, float), np.asarray(cdr_rest, float)])

    # Standardising CDR cannot change any fit (it spans the same column space)
    # but keeps the Newton step well conditioned.
    cdr_sd = cdr.std()
    if cdr_sd > 0:
        cdr = (cdr - cdr.mean()) / cdr_sd
    else:
        cdr = np.zeros(n_cells)

    condition = np.concatenate([np.ones(n1), np.zeros(n2)])
    D_full = np.column_stack([np.ones(n_cells), condition, cdr])
    D_reduced = np.column_stack([np.ones(n_cells), cdr])
    p_full = D_full.shape[1]

    scores = np.zeros(n_genes)
    pvals = np.ones(n_genes)

    report_step = max(1, min(n_genes // 10, 500))

    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)

        block = X_all[:, start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=float)

        detected = (block > 0).astype(float)

        # ---- discrete component ----
        # A gene detected in every cell or in none carries no detection signal,
        # so the logistic part is skipped rather than fitted to a constant.
        valid_bin = detected.sum(axis=0) > 0
        valid_bin &= detected.sum(axis=0) < n_cells

        chi2_bin = np.zeros(detected.shape[1])
        if valid_bin.any():
            ll_full = _logistic_loglik_block(detected[:, valid_bin], D_full)
            ll_red = _logistic_loglik_block(detected[:, valid_bin], D_reduced)
            chi2_bin[valid_bin] = np.maximum(0.0, 2.0 * (ll_full - ll_red))

        # ---- continuous component (detected cells only) ----
        chi2_cont = np.zeros(detected.shape[1])
        df_cont = np.zeros(detected.shape[1], dtype=int)

        for j in range(detected.shape[1]):
            rows = detected[:, j] > 0
            n_det = int(rows.sum())
            if n_det <= p_full:
                continue
            z = block[rows, j]
            Ds_full = D_full[rows]
            Ds_red = D_reduced[rows]

            rss_full = _residual_ss(Ds_full, z)
            rss_red = _residual_ss(Ds_red, z)
            if rss_full <= 0 or rss_red <= 0:
                continue

            chi2_cont[j] = max(0.0, n_det * np.log(rss_red / rss_full))
            df_cont[j] = 1

        df = valid_bin.astype(int) + df_cont
        total = chi2_bin + chi2_cont

        scores[start:stop] = total
        ok = df > 0
        pvals[start:stop] = np.where(
            ok, stats.chi2.sf(total, np.where(ok, df, 1)), 1.0
        )

        if verbose and (start + 1) % report_step == 0:
            pct = (start + 1) / n_genes * 100
            print(f"  [mast] {start+1}/{n_genes} genes ({pct:.0f}%)")

    return scores, pvals
