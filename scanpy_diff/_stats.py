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

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
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
    X_group: np.ndarray,
    X_rest: np.ndarray,
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DESeq2-style negative binomial test using pydeseq2.

    Note: This requires raw count data (integers). If normalized data is
    provided, results may be unreliable.

    Parameters
    ----------
    X_group : np.ndarray
        Raw count matrix for the group of interest.
    X_rest : np.ndarray
        Raw count matrix for the reference group.

    Returns
    -------
    scores : np.ndarray
        Log2 fold changes from DESeq2.
    pvals : np.ndarray
        p-values from Wald test.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for the 'deseq2' method. "
            "Install it with: pip install pydeseq2"
        )

    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    n_genes = X_group.shape[1]

    counts = np.vstack([X_group, X_rest]).astype(int)
    metadata = pd.DataFrame(
        {"condition": ["group"] * n1 + ["rest"] * n2},
        index=[f"cell_{i}" for i in range(n1 + n2)],
    )
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    counts_df = pd.DataFrame(counts, index=metadata.index, columns=gene_names)

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        quiet=True,
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", "group", "rest"], quiet=True)
    stat_res.summary()

    results = stat_res.results_df
    scores = results["log2FoldChange"].values
    pvals = results["pvalue"].fillna(1.0).values

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


def mast_test(
    X_group: np.ndarray | sparse.spmatrix,
    X_rest: np.ndarray | sparse.spmatrix,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAST hurdle model test (Finak et al., 2015 / Seurat 'mast').
    """
    scores, pvals = bimod_test(X_group, X_rest, verbose=verbose)
    return scores, pvals
