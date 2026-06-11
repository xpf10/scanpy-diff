"""
Tests for the core differential expression functions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scanpy_diff as sd
from scanpy_diff._diff import find_all_markers, find_markers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_adata():
    """
    Create a simple AnnData with 2 well-separated groups.

    Group 0: Genes 0-4 highly expressed
    Group 1: Genes 5-9 highly expressed
    """
    np.random.seed(42)
    n_cells = 200
    n_genes = 20

    X = np.zeros((n_cells, n_genes))

    # Group 0: cells 0-99, genes 0-4 highly expressed
    X[:100, :5] = np.random.lognormal(2, 0.5, size=(100, 5))

    # Group 1: cells 100-199, genes 5-9 highly expressed
    X[100:, 5:10] = np.random.lognormal(2, 0.5, size=(100, 5))

    # Background noise
    X += np.random.lognormal(-1, 0.3, size=(n_cells, n_genes))

    # Log-normalize (as scanpy would)
    X = np.log1p(X)

    obs = pd.DataFrame(
        {"cluster": ["0"] * 100 + ["1"] * 100},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def multi_group_adata():
    """AnnData with 4 clusters, each with distinct marker genes."""
    np.random.seed(123)
    n_per_cluster = 50
    n_clusters = 4
    n_cells = n_per_cluster * n_clusters
    n_genes = 40  # 10 genes per cluster

    X = np.zeros((n_cells, n_genes))
    labels = []

    for c in range(n_clusters):
        start = c * n_per_cluster
        end = start + n_per_cluster
        gene_start = c * 10
        gene_end = gene_start + 10
        X[start:end, gene_start:gene_end] = np.random.lognormal(
            2.5, 0.5, size=(n_per_cluster, 10)
        )
        labels.extend([str(c)] * n_per_cluster)

    # Background
    X += np.random.lognormal(-1, 0.3, size=(n_cells, n_genes))
    X = np.log1p(X)

    obs = pd.DataFrame(
        {"leiden": labels},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        index=[f"gene_{i}" for i in range(n_genes)]
    )

    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Tests for find_markers
# ---------------------------------------------------------------------------


class TestFindMarkers:
    """Tests for the find_markers() function."""

    def test_basic_wilcoxon(self, simple_adata):
        """Basic Wilcoxon test returns a DataFrame with expected columns."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="wilcoxon",
            verbose=False,
        )

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"gene", "scores", "log2fc", "pct_1", "pct_2", "pval", "padj"}
        assert expected_cols.issubset(set(result.columns))

    def test_detects_true_markers_wilcoxon(self, simple_adata):
        """Wilcoxon test should detect the planted marker genes."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="wilcoxon",
            padj_cutoff=0.05,
            only_positive=True,
            verbose=False,
        )

        # All top genes should be from the planted set (gene_0 to gene_4)
        top_genes = result.head(5)["gene"].tolist()
        expected_markers = {"gene_0", "gene_1", "gene_2", "gene_3", "gene_4"}
        found = set(top_genes) & expected_markers
        assert len(found) >= 3, f"Expected planted markers in top 5, got: {top_genes}"

    def test_ttest_method(self, simple_adata):
        """t-test method should work and return results."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="t-test",
            verbose=False,
        )
        assert len(result) > 0
        assert "log2fc" in result.columns

    def test_logreg_method(self, simple_adata):
        """Logistic regression method should work."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="logreg",
            verbose=False,
        )
        assert len(result) > 0
        assert result["scores"].min() >= 0  # LR statistic is non-negative

    def test_roc_method(self, simple_adata):
        """ROC method should return AUC scores in [0, 1]."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="roc",
            verbose=False,
        )
        assert len(result) > 0
        # AUC should be in [0, 1]
        assert result["scores"].between(0, 1).all(), \
            f"AUC out of range: {result['scores'].describe()}"

    def test_reference_group(self, simple_adata):
        """Test comparison against a specific reference group."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            reference="1",
            method="wilcoxon",
            verbose=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.attrs["reference"] == "1"

    def test_logfc_threshold_reduces_genes(self, simple_adata):
        """Higher logfc_threshold should reduce number of tested genes."""
        result_strict = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            logfc_threshold=2.0,
            verbose=False,
        )
        result_lenient = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            logfc_threshold=0.0,
            verbose=False,
        )
        assert len(result_strict) <= len(result_lenient)

    def test_only_positive_filter(self, simple_adata):
        """only_positive=True should return only upregulated genes."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            only_positive=True,
            verbose=False,
        )
        assert (result["log2fc"] > 0).all()

    def test_pct_columns_range(self, simple_adata):
        """pct_1 and pct_2 should be in [0, 1]."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            verbose=False,
        )
        assert result["pct_1"].between(0, 1).all()
        assert result["pct_2"].between(0, 1).all()

    def test_padj_leq_1(self, simple_adata):
        """Adjusted p-values should be <= 1."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            verbose=False,
        )
        assert (result["padj"] <= 1.0).all()
        assert (result["padj"] >= 0.0).all()

    def test_invalid_group_raises(self, simple_adata):
        """Invalid group name should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            find_markers(
                simple_adata,
                groupby="cluster",
                group="999",
                verbose=False,
            )

    def test_invalid_groupby_raises(self, simple_adata):
        """Invalid groupby column should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            find_markers(
                simple_adata,
                groupby="nonexistent_column",
                group="0",
                verbose=False,
            )

    def test_n_genes_limit(self, simple_adata):
        """n_genes parameter should limit the number of returned genes."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            n_genes=3,
            verbose=False,
        )
        assert len(result) <= 3

    def test_attrs_populated(self, simple_adata):
        """Result attrs should contain metadata."""
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            method="wilcoxon",
            verbose=False,
        )
        assert result.attrs["group"] == "0"
        assert result.attrs["reference"] == "rest"
        assert result.attrs["method"] == "wilcoxon"
        assert "n_cells_group" in result.attrs

    def test_bonferroni_correction(self, simple_adata):
        """Bonferroni correction should give stricter p-values than BH."""
        result_bh = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            correction_method="fdr_bh",
            verbose=False,
        )
        result_bf = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            correction_method="bonferroni",
            verbose=False,
        )
        # Bonferroni padj >= BH padj on average
        common_genes = set(result_bh["gene"]) & set(result_bf["gene"])
        if common_genes:
            bh_mean = result_bh[result_bh["gene"].isin(common_genes)]["padj"].mean()
            bf_mean = result_bf[result_bf["gene"].isin(common_genes)]["padj"].mean()
            assert bf_mean >= bh_mean, \
                "Expected Bonferroni to be more conservative than BH"

    def test_layer_parameter(self, simple_adata):
        """Using a custom layer should work correctly."""
        simple_adata.layers["counts"] = np.exp(simple_adata.X) - 1
        result = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            layer="counts",
            verbose=False,
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests for find_all_markers
# ---------------------------------------------------------------------------


class TestFindAllMarkers:
    """Tests for find_all_markers()."""

    def test_returns_all_clusters(self, multi_group_adata):
        """find_all_markers should return results for all clusters."""
        result = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            method="wilcoxon",
            padj_cutoff=1.0,  # No filtering
            verbose=False,
        )
        assert "cluster" in result.columns
        found_clusters = set(result["cluster"].unique())
        expected_clusters = {"0", "1", "2", "3"}
        assert expected_clusters.issubset(found_clusters)

    def test_detects_planted_markers(self, multi_group_adata):
        """find_all_markers should correctly identify cluster-specific markers."""
        result = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            method="wilcoxon",
            padj_cutoff=0.05,
            only_positive=True,
            verbose=False,
        )

        # For cluster 0, the top markers should be gene_0 to gene_9
        cluster0_markers = result[result["cluster"] == "0"]["gene"].head(5).tolist()
        expected_0 = {f"gene_{i}" for i in range(10)}
        found = set(cluster0_markers) & expected_0
        assert len(found) >= 3, \
            f"Expected cluster 0 markers in planted genes, got: {cluster0_markers}"

    def test_subset_groups(self, multi_group_adata):
        """groups parameter should restrict which clusters are tested."""
        result = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            groups=["0", "1"],
            verbose=False,
        )
        assert set(result["cluster"].unique()) == {"0", "1"}

    def test_n_genes_per_group(self, multi_group_adata):
        """n_genes_per_group limits result size per cluster."""
        result = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            n_genes_per_group=5,
            verbose=False,
        )
        for cluster, grp in result.groupby("cluster"):
            assert len(grp) <= 5, \
                f"Cluster {cluster} has {len(grp)} genes, expected <= 5"

    def test_result_has_cluster_column(self, multi_group_adata):
        """Result must have 'cluster' column."""
        result = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            verbose=False,
        )
        assert "cluster" in result.columns

    def test_invalid_groups_raises(self, multi_group_adata):
        """Requesting non-existent groups should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            find_all_markers(
                multi_group_adata,
                groupby="leiden",
                groups=["0", "999"],
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------


class TestUtils:
    """Tests for utility functions."""

    def test_filter_markers(self, simple_adata):
        """filter_markers should correctly apply filters."""
        markers = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            verbose=False,
        )
        filtered = sd.filter_markers(markers, min_log2fc=1.0, max_padj=0.05)
        assert (filtered["padj"] <= 0.05).all()
        assert (np.abs(filtered["log2fc"]) >= 1.0).all()

    def test_rank_markers_by_log2fc(self, simple_adata):
        """rank_markers should sort by log2fc correctly."""
        markers = find_markers(
            simple_adata,
            groupby="cluster",
            group="0",
            verbose=False,
        )
        ranked = sd.rank_markers(markers, by="log2fc", ascending=False)
        lfc_values = ranked["log2fc"].values
        assert np.all(lfc_values[:-1] >= lfc_values[1:]), \
            "Expected descending log2fc order"

    def test_top_markers(self, multi_group_adata):
        """top_markers should return at most n genes per cluster."""
        all_markers = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            verbose=False,
        )
        top = sd.top_markers(all_markers, n=3)
        for cluster, grp in top.groupby("cluster"):
            assert len(grp) <= 3

    def test_markers_to_dict(self, multi_group_adata):
        """markers_to_dict should return a proper dict."""
        all_markers = find_all_markers(
            multi_group_adata,
            groupby="leiden",
            verbose=False,
        )
        d = sd.markers_to_dict(all_markers, n=5)
        assert isinstance(d, dict)
        assert len(d) == 4  # 4 clusters
        for k, v in d.items():
            assert isinstance(v, list)
            assert len(v) <= 5


# ---------------------------------------------------------------------------
# Tests for statistical functions
# ---------------------------------------------------------------------------


class TestStatFunctions:
    """Tests for individual statistical functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(0)
        n = 50
        n_genes = 10
        # Group with high expression for first 5 genes
        self.X_group = np.hstack([
            np.random.lognormal(3, 0.5, (n, 5)),
            np.random.lognormal(0, 0.3, (n, 5)),
        ])
        # Rest with low expression
        self.X_rest = np.hstack([
            np.random.lognormal(0, 0.3, (n, 5)),
            np.random.lognormal(3, 0.5, (n, 5)),
        ])

    def test_wilcoxon_pval_range(self):
        from scanpy_diff._stats import wilcoxon_test
        scores, pvals = wilcoxon_test(self.X_group, self.X_rest)
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_wilcoxon_detects_upregulation(self):
        from scanpy_diff._stats import wilcoxon_test
        scores, pvals = wilcoxon_test(self.X_group, self.X_rest)
        # First 5 genes should be significant
        assert np.all(pvals[:5] < 0.05), f"Expected significant pvals, got: {pvals[:5]}"

    def test_ttest_pval_range(self):
        from scanpy_diff._stats import ttest
        scores, pvals = ttest(self.X_group, self.X_rest)
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_ttest_detects_upregulation(self):
        from scanpy_diff._stats import ttest
        scores, pvals = ttest(self.X_group, self.X_rest)
        assert np.all(pvals[:5] < 0.05), f"Expected significant pvals, got: {pvals[:5]}"

    def test_compute_pct(self):
        from scanpy_diff._stats import compute_pct
        X = np.array([[0, 1, 2], [0, 0, 3], [1, 1, 4]])
        pct = compute_pct(X, threshold=0)
        expected = np.array([1/3, 2/3, 1.0])
        np.testing.assert_allclose(pct, expected, rtol=1e-5)

    def test_compute_log2fc_direction(self):
        from scanpy_diff._stats import compute_log2fc
        # group has higher expression → positive log2fc
        X_high = np.full((10, 3), 5.0)
        X_low = np.full((10, 3), 0.5)
        lfc = compute_log2fc(X_high, X_low)
        assert np.all(lfc > 0), f"Expected positive log2fc, got: {lfc}"

    def test_adjust_pvalues_bh(self):
        from scanpy_diff._stats import adjust_pvalues
        pvals = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
        padj = adjust_pvalues(pvals, method="fdr_bh")
        assert np.all(padj >= pvals), "Adjusted p-values should be >= raw p-values"
        assert np.all(padj <= 1.0)

    def test_adjust_pvalues_bonferroni(self):
        from scanpy_diff._stats import adjust_pvalues
        pvals = np.array([0.001, 0.01, 0.1])
        padj = adjust_pvalues(pvals, method="bonferroni")
        # Bonferroni = pval * n (capped at 1)
        expected = np.minimum(pvals * len(pvals), 1.0)
        np.testing.assert_allclose(padj, expected, rtol=1e-5)

    def test_roc_auc_range(self):
        from scanpy_diff._stats import roc_test
        scores, pvals = roc_test(self.X_group, self.X_rest)
        assert np.all((scores >= 0) & (scores <= 1)), \
            f"AUC scores out of range: {scores}"
