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
    X_group: np.ndarray,
    X_rest: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wilcoxon rank-sum (Mann-Whitney U) test for each gene.

    This is the default method in Seurat's FindMarkers. Tests whether the
    distribution of expression values differs between two groups without
    assuming normality.

    Parameters
    ----------
    X_group : np.ndarray
        Expression matrix for the group of interest (n_cells_group x n_genes).
    X_rest : np.ndarray
        Expression matrix for the reference group (n_cells_rest x n_genes).

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

    for i in range(n_genes):
        g = X_group[:, i]
        r = X_rest[:, i]

        # Skip constant genes
        if np.all(g == g[0]) and np.all(r == r[0]) and g[0] == r[0]:
            pvals[i] = 1.0
            scores[i] = 0.5
            continue

        try:
            stat, pval = stats.mannwhitneyu(g, r, alternative="two-sided")
            scores[i] = stat / (n1 * n2)  # Normalize to [0,1] (AUC)
            pvals[i] = pval
        except (ValueError, RuntimeError):
            pvals[i] = 1.0
            scores[i] = 0.5

    return scores, pvals


# ---------------------------------------------------------------------------
# Student's t-test (equivalent to Seurat's "t")
# ---------------------------------------------------------------------------


def ttest(
    X_group: np.ndarray,
    X_rest: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch's two-sample t-test for each gene.

    Parameters
    ----------
    X_group : np.ndarray
        Expression matrix for the group of interest.
    X_rest : np.ndarray
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

    mean1 = X_group.mean(axis=0)
    mean2 = X_rest.mean(axis=0)
    var1 = X_group.var(axis=0, ddof=1) + 1e-9
    var2 = X_rest.var(axis=0, ddof=1) + 1e-9
    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]

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
    X_group: np.ndarray,
    X_rest: np.ndarray,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logistic regression likelihood-ratio test for each gene.

    Uses a logistic regression model to predict group membership from gene
    expression. The LR test statistic compares the model with the gene vs
    a null model.

    Parameters
    ----------
    X_group : np.ndarray
        Expression matrix for the group of interest.
    X_rest : np.ndarray
        Expression matrix for the reference group.
    max_iter : int
        Maximum iterations for logistic regression solver.

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

    for i in range(n_genes):
        xi = X[:, i : i + 1]
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
    X_group: np.ndarray,
    X_rest: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROC AUC analysis for each gene.

    Computes the AUC for classifying group membership based on gene
    expression. AUC > 0.5 means gene is upregulated in group.

    Parameters
    ----------
    X_group : np.ndarray
        Expression matrix for the group of interest.
    X_rest : np.ndarray
        Expression matrix for the reference group.

    Returns
    -------
    scores : np.ndarray
        AUC values (0.5 = random, 1.0 = perfect classifier).
    pvals : np.ndarray
        p-values from Mann-Whitney U (same as Wilcoxon).
    """
    from sklearn.metrics import roc_auc_score

    n_genes = X_group.shape[1]
    scores = np.zeros(n_genes)
    pvals = np.ones(n_genes)

    n1 = X_group.shape[0]
    n2 = X_rest.shape[0]
    y = np.array([1] * n1 + [0] * n2)
    X = np.vstack([X_group, X_rest])

    for i in range(n_genes):
        xi = X[:, i]
        if np.all(xi == xi[0]):
            scores[i] = 0.5
            pvals[i] = 1.0
            continue
        try:
            auc = roc_auc_score(y, xi)
            scores[i] = auc
            _, pval = stats.mannwhitneyu(
                X_group[:, i], X_rest[:, i], alternative="two-sided"
            )
            pvals[i] = pval
        except (ValueError, RuntimeError):
            scores[i] = 0.5
            pvals[i] = 1.0

    return scores, pvals


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


def compute_pct(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Compute fraction of cells expressing each gene above threshold.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells x genes).
    threshold : float
        Expression threshold. A gene is considered expressed if > threshold.

    Returns
    -------
    np.ndarray
        Fraction of cells expressing each gene (length = n_genes).
    """
    if sparse.issparse(X):
        return np.asarray((X > threshold).mean(axis=0)).flatten()
    return (X > threshold).mean(axis=0)


# ---------------------------------------------------------------------------
# Helper: compute log2 fold change
# ---------------------------------------------------------------------------


def compute_log2fc(
    X_group: np.ndarray,
    X_rest: np.ndarray,
    base: float = 2.0,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """
    Compute average log fold change between two groups.

    Mimics Seurat's avg_log2FC calculation. Assumes data is log-normalized
    (e.g., log1p). The fold change is computed in the natural scale.

    Parameters
    ----------
    X_group : np.ndarray
        Log-normalized expression matrix for the group of interest.
    X_rest : np.ndarray
        Log-normalized expression matrix for the reference group.
    base : float
        Log base used for fold change calculation.
    pseudocount : float
        Pseudocount added before log transformation to avoid log(0).

    Returns
    -------
    np.ndarray
        Log2 fold change values per gene.
    """
    # Convert from log1p scale back to natural scale, compute means, then log FC
    mean1 = np.expm1(X_group).mean(axis=0)  # mean in natural scale
    mean2 = np.expm1(X_rest).mean(axis=0)

    log2fc = np.log2(mean1 + pseudocount) - np.log2(mean2 + pseudocount)
    return log2fc
