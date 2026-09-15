# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

scanpy-diff is a scanpy plugin providing Seurat-style differential gene expression analysis (`FindMarkers`/`FindAllMarkers`). Built with hatchling, targeting Python ≥3.9. Core dependencies: scanpy, anndata, numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, igraph, leidenalg.

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a specific test class
pytest tests/test_diff.py::TestFindMarkers -v

# Run a single test
pytest tests/test_diff.py::TestFindMarkers::test_basic_wilcoxon -v

# Run with coverage
pytest tests/ --cov=scanpy_diff --cov-report=html

# Lint (ruff config in pyproject.toml)
ruff check scanpy_diff/ tests/
```

## Architecture

```
scanpy_diff/
├── __init__.py    # Public API — exports find_markers, find_all_markers, filter_markers,
│                  #   rank_markers, top_markers, markers_to_dict, pl
├── _diff.py       # Core DE pipeline: find_markers() and find_all_markers().
│                  #   Orchestrates validation → subset → pre-filter → test → correction → result
├── _stats.py      # Statistical engines and low-level helpers.
│                  #   Tests: wilcoxon_test, ttest, logistic_regression_test, roc_test,
│                  #          bimod_test, poisson_test, negbinom_test, mast_test, deseq2_test
│                  #   Helpers: compute_pct, compute_log2fc, adjust_pvalues
├── _utils.py      # Post-analysis result manipulation.
│                  #   filter_markers, rank_markers, top_markers, markers_to_dict, store_in_adata
└── pl.py          # Visualization (all return matplotlib Axes).
                    #   volcano, marker_heatmap, dotplot, violin, summary_barplot, log2fc_heatmap
```

### Data flow through `find_markers()` (the central function)

1. Validate inputs (group/reference exist in adata.obs)
2. Subset cells by group mask; extract expression matrix (dense) from `adata.X`, a specific `layer`, or `adata.raw`
3. Pre-filter genes by `min_pct`, `min_pct_reference`, and `|log2fc| >= logfc_threshold`
4. Dispatch to the selected statistical test (each takes `X_group, X_rest` → `scores, pvals`).
   Two exceptions:
   - `deseq2` receives the whole `adata` plus `replicate_col`/`covariates` and aggregates cells
     into pseudo-bulk samples itself, so it also skips the `find_all_markers()` precomputation step.
   - `mast` additionally receives `cdr_group`/`cdr_rest` — per-cell detection rates computed over
     the **full** gene set in step 3's input, before pre-filtering narrows the genes.
5. Apply multiple testing correction via statsmodels
6. Build and sort the result `pd.DataFrame`, attach metadata in `df.attrs`

`find_all_markers()` loops over groups, calling `find_markers()` for each, and concatenates results with a `cluster` column.

### Key design decisions

- **Expression data is converted to dense** before testing (handles sparse AnnData input).
- **Log2FC is computed in natural scale** from log1p-transformed data: `expm1` → mean → `log2(mean+pseudocount)` difference. This mirrors Seurat's approach. `method="deseq2"` forces `expression_scale="raw"` instead, because that path validates the input is raw integer counts.
- **MAST's CDR covariate is computed before gene pre-filtering.** `mast_test` is a genuine two-part hurdle model (logistic on the detection indicator + Gaussian on detected cells only, both including the per-cell detection rate), not an alias for `bimod_test`. CDR measures library-level technical noise, so it has to span the full gene set — deriving it from the `min_pct`/`logfc_threshold`-filtered subset would compress the cell-to-cell detection variance the covariate exists to absorb. `find_all_markers()` caches it in `_precomputed_stats["cdr"]` to avoid a full-matrix pass per group.
- **`compute_pct` and `compute_log2fc`** live in `_stats.py` (not `_utils.py`) because `_diff.py` imports them alongside the test functions for the pre-filtering step.
- **`pl.py` wraps scanpy's built-in plotting** for heatmap/dotplot/violin and adds custom volcano, summary barplot, and log2fc heatmap.
- The `comparison/` directory contains R scripts and Seurat outputs for validation against Seurat — it is not part of the Python package.
