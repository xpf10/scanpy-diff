"""
PBMC3k Differential Expression: Scanpy-diff vs Seurat-style Comparison
=======================================================================

This script:
1. Downloads PBMC3k dataset
2. Runs full preprocessing pipeline (normalize → log → HVG → PCA → neighbors → UMAP → clustering)
3. Runs DE analysis with:
   - scanpy built-in rank_genes_groups (Wilcoxon) — mirrors Seurat's algorithm
   - scanpy-diff find_markers (Wilcoxon / t-test / logreg / roc)
4. Compares log2FC, p-values, gene ranking between the two
5. Saves comparison figures and tables to results/
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy_diff as sd
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# Output directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("  PBMC3k DE Analysis: scanpy-diff vs Seurat-style")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────
# 1. Load & preprocess PBMC3k
# ─────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading PBMC3k dataset ...")
CACHE_H5AD = os.path.join(RESULTS_DIR, "pbmc3k_processed.h5ad")

if os.path.exists(CACHE_H5AD):
    print(f"  Loading from cache: {CACHE_H5AD}")
    adata = sc.read_h5ad(CACHE_H5AD)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes, "
          f"{adata.obs['leiden'].nunique()} clusters")
else:
    adata = sc.datasets.pbmc3k()
    print(f"  Raw data: {adata.n_obs} cells × {adata.n_vars} genes")

    print("\n[2/5] Preprocessing ...")

    # QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5].copy()
    print(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # Store raw counts for later
    adata.layers["counts"] = adata.X.copy()

    # Normalize & log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # Save log-normalized data as raw (for rank_genes_groups)

    # HVG, scale, PCA
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.pp.pca(adata_hvg, n_comps=40, svd_solver="arpack")

    # Copy PCA embedding back
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # Neighbors, UMAP, Leiden clustering
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    print(f"  Clusters: {adata.obs['leiden'].nunique()} groups")
    print(f"  Cluster sizes:\n{adata.obs['leiden'].value_counts().sort_index().to_string()}")

    # Save processed adata
    adata.write_h5ad(CACHE_H5AD)
    print(f"  Saved: {CACHE_H5AD}")

print("\n[2/5] Preprocessing ... (skipped, loaded from cache)")

# Plot UMAP
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(adata, color="leiden", ax=ax, show=False, title="PBMC3k — Leiden clusters")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_umap_clusters.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {RESULTS_DIR}/01_umap_clusters.png")


# ─────────────────────────────────────────────────────────────────────
# 3. Scanpy built-in rank_genes_groups (Seurat-style reference)
# ─────────────────────────────────────────────────────────────────────
print("\n[3/5] Running scanpy rank_genes_groups (Seurat-style reference) ...")

METHODS_SC = ["wilcoxon", "t-test"]
for method in METHODS_SC:
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method=method,
        use_raw=True,
        key_added=f"sc_{method}",
        pts=True,         # Compute pct.1 and pct.2 (like Seurat)
    )
    print(f"  Done: sc.tl.rank_genes_groups [{method}]")


def scanpy_rgg_to_df(adata, key, group):
    """Convert scanpy rank_genes_groups result to a clean DataFrame."""
    result = adata.uns[key]
    names = result["names"][group]
    scores = result["scores"][group]
    pvals = result["pvals"][group]
    padj = result["pvals_adj"][group]
    log2fc = result["logfoldchanges"][group]
    # pct (pts) may not always be present
    pts1 = result.get("pts", {}).get(group, pd.Series(dtype=float))
    pts2 = result.get("pts_rest", {}).get(group, pd.Series(dtype=float))
    
    df = pd.DataFrame({
        "gene": names,
        "scores": scores,
        "log2fc": log2fc,
        "pval": pvals,
        "padj": padj,
    })
    if len(pts1):
        df["pct_1"] = df["gene"].map(pts1)
        df["pct_2"] = df["gene"].map(pts2)
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. scanpy-diff analysis (all methods)
# ─────────────────────────────────────────────────────────────────────
print("\n[4/5] Running scanpy-diff (all methods) ...")

SD_METHODS = ["wilcoxon", "t-test", "logreg", "roc"]
sd_all_results = {}

for method in SD_METHODS:
    print(f"  Running find_all_markers [{method}] ...")
    result = sd.find_all_markers(
        adata,
        groupby="leiden",
        method=method,
        min_pct=0.1,
        logfc_threshold=0.25,
        padj_cutoff=1.0,
        only_positive=False,
        verbose=False,
    )
    sd_all_results[method] = result
    out_path = os.path.join(RESULTS_DIR, f"sd_{method}_all_markers.csv")
    result.to_csv(out_path, index=False)
    print(f"    → {len(result)} results saved to {out_path}")

# ─────────────────────────────────────────────────────────────────────
# 5. Comparison: scanpy-diff Wilcoxon vs scanpy rank_genes_groups
# ─────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating comparison plots ...")

# Pick cluster 0 for detailed comparison
FOCUS_CLUSTER = "0"

sc_df = scanpy_rgg_to_df(adata, "sc_wilcoxon", FOCUS_CLUSTER)
sd_df = sd_all_results["wilcoxon"][sd_all_results["wilcoxon"]["cluster"] == FOCUS_CLUSTER].copy()

# Merge on gene
merged = pd.merge(
    sc_df.rename(columns={"log2fc": "sc_log2fc", "padj": "sc_padj", "scores": "sc_scores"}),
    sd_df.rename(columns={"log2fc": "sd_log2fc", "padj": "sd_padj", "scores": "sd_scores"})[
        ["gene", "sd_log2fc", "sd_padj", "sd_scores"]
    ],
    on="gene",
    how="inner",
)
print(f"  Merged genes for cluster '{FOCUS_CLUSTER}': {len(merged)}")
merged.to_csv(os.path.join(RESULTS_DIR, f"comparison_cluster{FOCUS_CLUSTER}.csv"), index=False)

# ── Figure 1: Multi-panel comparison ──────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    f"Differential Expression Comparison: Cluster {FOCUS_CLUSTER} vs Rest\n"
    "scanpy rank_genes_groups (Seurat-style)  vs  scanpy-diff",
    fontsize=14, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

BLUE = "#2980B9"
RED = "#E74C3C"
GRAY = "#95A5A6"
GREEN = "#27AE60"

# ── Panel A: log2FC correlation ───────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
corr_lfc, pval_lfc = stats.pearsonr(merged["sc_log2fc"], merged["sd_log2fc"])
ax_a.scatter(
    merged["sc_log2fc"], merged["sd_log2fc"],
    alpha=0.35, s=8, c=BLUE, linewidths=0, rasterized=True,
)
lim = max(abs(merged["sc_log2fc"].max()), abs(merged["sd_log2fc"].max())) + 0.3
ax_a.plot([-lim, lim], [-lim, lim], "r--", lw=1, alpha=0.7, label="y=x")
ax_a.set_xlim(-lim, lim)
ax_a.set_ylim(-lim, lim)
ax_a.set_xlabel("scanpy log₂FC", fontsize=10)
ax_a.set_ylabel("scanpy-diff log₂FC", fontsize=10)
ax_a.set_title(f"log₂FC Correlation\n(r={corr_lfc:.4f})", fontsize=10, fontweight="bold")
ax_a.legend(fontsize=8)
ax_a.text(0.05, 0.92, f"r = {corr_lfc:.4f}", transform=ax_a.transAxes, fontsize=9,
          color="red", fontweight="bold")

# ── Panel B: -log10(padj) correlation ────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
sc_neglog = -np.log10(merged["sc_padj"].clip(1e-300))
sd_neglog = -np.log10(merged["sd_padj"].clip(1e-300))
corr_p, _ = stats.pearsonr(sc_neglog, sd_neglog)
ax_b.scatter(sc_neglog, sd_neglog, alpha=0.35, s=8, c=RED, linewidths=0, rasterized=True)
lim_p = max(sc_neglog.max(), sd_neglog.max()) + 1
ax_b.plot([0, lim_p], [0, lim_p], "b--", lw=1, alpha=0.7, label="y=x")
ax_b.set_xlim(0, lim_p)
ax_b.set_ylim(0, lim_p)
ax_b.set_xlabel("scanpy −log₁₀(padj)", fontsize=10)
ax_b.set_ylabel("scanpy-diff −log₁₀(padj)", fontsize=10)
ax_b.set_title(f"Significance Correlation\n(r={corr_p:.4f})", fontsize=10, fontweight="bold")
ax_b.text(0.05, 0.92, f"r = {corr_p:.4f}", transform=ax_b.transAxes, fontsize=9,
          color="blue", fontweight="bold")

# ── Panel C: Gene rank correlation ────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
sc_rank = merged["sc_log2fc"].rank(ascending=False)
sd_rank = merged["sd_log2fc"].rank(ascending=False)
rho, _ = stats.spearmanr(sc_rank, sd_rank)
ax_c.scatter(sc_rank, sd_rank, alpha=0.3, s=8, c=GREEN, linewidths=0, rasterized=True)
ax_c.plot([0, len(merged)], [0, len(merged)], "r--", lw=1, alpha=0.7)
ax_c.set_xlabel("scanpy gene rank", fontsize=10)
ax_c.set_ylabel("scanpy-diff gene rank", fontsize=10)
ax_c.set_title(f"Gene Rank Correlation\n(Spearman ρ={rho:.4f})", fontsize=10, fontweight="bold")
ax_c.text(0.05, 0.92, f"ρ = {rho:.4f}", transform=ax_c.transAxes, fontsize=9,
          color="darkgreen", fontweight="bold")

# ── Panel D: Volcano scanpy ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
sig = (merged["sc_padj"] < 0.05) & (np.abs(merged["sc_log2fc"]) > 1)
ax_d.scatter(
    merged.loc[~sig, "sc_log2fc"], -np.log10(merged.loc[~sig, "sc_padj"].clip(1e-300)),
    s=6, c=GRAY, alpha=0.4, linewidths=0, rasterized=True,
)
ax_d.scatter(
    merged.loc[sig, "sc_log2fc"], -np.log10(merged.loc[sig, "sc_padj"].clip(1e-300)),
    s=6, c=BLUE, alpha=0.7, linewidths=0, rasterized=True,
)
ax_d.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
ax_d.axvline(1, color="gray", ls="--", lw=0.8)
ax_d.axvline(-1, color="gray", ls="--", lw=0.8)
ax_d.set_xlabel("log₂FC", fontsize=10)
ax_d.set_ylabel("-log₁₀(padj)", fontsize=10)
ax_d.set_title(f"Volcano — scanpy\n({sig.sum()} sig. genes)", fontsize=10, fontweight="bold")

# ── Panel E: Volcano scanpy-diff ──────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
sig2 = (merged["sd_padj"] < 0.05) & (np.abs(merged["sd_log2fc"]) > 1)
ax_e.scatter(
    merged.loc[~sig2, "sd_log2fc"], -np.log10(merged.loc[~sig2, "sd_padj"].clip(1e-300)),
    s=6, c=GRAY, alpha=0.4, linewidths=0, rasterized=True,
)
ax_e.scatter(
    merged.loc[sig2, "sd_log2fc"], -np.log10(merged.loc[sig2, "sd_padj"].clip(1e-300)),
    s=6, c=RED, alpha=0.7, linewidths=0, rasterized=True,
)
ax_e.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
ax_e.axvline(1, color="gray", ls="--", lw=0.8)
ax_e.axvline(-1, color="gray", ls="--", lw=0.8)
ax_e.set_xlabel("log₂FC", fontsize=10)
ax_e.set_ylabel("-log₁₀(padj)", fontsize=10)
ax_e.set_title(f"Volcano — scanpy-diff\n({sig2.sum()} sig. genes)", fontsize=10, fontweight="bold")

# ── Panel F: Top gene overlap (Venn-like bar chart) ───────────────────
ax_f = fig.add_subplot(gs[1, 2])
top_ns = [10, 20, 50, 100, 200]
overlaps = []
for n in top_ns:
    sc_top = set(sc_df.head(n)["gene"])
    sd_top = set(sd_df.head(n)["gene"])
    overlap = len(sc_top & sd_top)
    overlaps.append(overlap / n * 100)

bars = ax_f.bar([str(n) for n in top_ns], overlaps, color="#8E44AD", alpha=0.8)
for bar, ov in zip(bars, overlaps):
    ax_f.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{ov:.0f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
ax_f.set_xlabel("Top N genes", fontsize=10)
ax_f.set_ylabel("Overlap (%)", fontsize=10)
ax_f.set_ylim(0, 110)
ax_f.set_title("Top Gene Overlap\nscanpy vs scanpy-diff (Wilcoxon)", fontsize=10, fontweight="bold")
ax_f.spines["top"].set_visible(False)
ax_f.spines["right"].set_visible(False)

# ── Panel G: log2FC comparison across ALL clusters ────────────────────
ax_g = fig.add_subplot(gs[2, :2])
clusters = sorted(adata.obs["leiden"].unique().tolist())
corr_per_cluster = []
n_sig_sc = []
n_sig_sd = []

for clust in clusters:
    sc_c = scanpy_rgg_to_df(adata, "sc_wilcoxon", clust)
    sd_c = sd_all_results["wilcoxon"][sd_all_results["wilcoxon"]["cluster"] == clust]
    mg = pd.merge(
        sc_c[["gene", "log2fc", "padj"]].rename(columns={"log2fc": "sc_lfc", "padj": "sc_padj"}),
        sd_c[["gene", "log2fc", "padj"]].rename(columns={"log2fc": "sd_lfc", "padj": "sd_padj"}),
        on="gene", how="inner",
    )
    if len(mg) > 10:
        r, _ = stats.pearsonr(mg["sc_lfc"], mg["sd_lfc"])
        corr_per_cluster.append(r)
    else:
        corr_per_cluster.append(np.nan)
    n_sig_sc.append((sc_c["padj"] < 0.05).sum())
    n_sig_sd.append((sd_c["padj"] < 0.05).sum())

x = np.arange(len(clusters))
w = 0.35
bars1 = ax_g.bar(x - w/2, n_sig_sc, w, label="scanpy (Seurat-style)", color=BLUE, alpha=0.8)
bars2 = ax_g.bar(x + w/2, n_sig_sd, w, label="scanpy-diff", color=RED, alpha=0.8)
ax_g.set_xticks(x)
ax_g.set_xticklabels([f"C{c}" for c in clusters])
ax_g.set_xlabel("Cluster", fontsize=10)
ax_g.set_ylabel("# Significant Genes (padj<0.05)", fontsize=10)
ax_g.set_title("Significant DE Genes per Cluster — Wilcoxon", fontsize=10, fontweight="bold")
ax_g.legend(fontsize=9)
ax_g.spines["top"].set_visible(False)
ax_g.spines["right"].set_visible(False)

# Add log2FC correlation as text above bars
for i, (c, r) in enumerate(zip(clusters, corr_per_cluster)):
    if not np.isnan(r):
        ax_g.text(i, max(n_sig_sc[i], n_sig_sd[i]) + 3, f"r={r:.2f}",
                  ha="center", va="bottom", fontsize=7, color="#2C3E50")

# ── Panel H: Method comparison summary table ──────────────────────────
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis("off")

# Build summary stats
summary_data = []
for method in SD_METHODS:
    sd_c0 = sd_all_results[method][sd_all_results[method]["cluster"] == FOCUS_CLUSTER]
    n_sig = (sd_c0["padj"] < 0.05).sum()
    n_up = ((sd_c0["padj"] < 0.05) & (sd_c0["log2fc"] > 0)).sum()
    summary_data.append([method.upper(), n_sig, n_up])

table_data = [["Method", "#Sig", "#Up"]] + summary_data
table = ax_h.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#ECF0F1")
    cell.set_edgecolor("white")
ax_h.set_title(f"Methods Summary\n(Cluster {FOCUS_CLUSTER} vs rest)", fontsize=10, fontweight="bold")

plt.savefig(
    os.path.join(RESULTS_DIR, "02_comparison_multipanel.png"),
    dpi=150, bbox_inches="tight",
)
plt.close()
print(f"  Saved: {RESULTS_DIR}/02_comparison_multipanel.png")

# ── Figure 2: Top markers heatmap ────────────────────────────────────
print("  Generating marker heatmap ...")
top10 = sd.top_markers(sd_all_results["wilcoxon"], n=5)
gene_list = top10["gene"].drop_duplicates().tolist()

# sc.pl.heatmap returns a dict of axes; don't pass ax= directly
sc.settings.figdir = RESULTS_DIR
sc.pl.heatmap(
    adata,
    var_names=gene_list,
    groupby="leiden",
    use_raw=True,
    cmap="RdYlBu_r",
    show=False,
    save="_marker_heatmap_tmp.png",
)
# Rename the auto-saved file
import shutil
tmp = os.path.join(RESULTS_DIR, "heatmap_marker_heatmap_tmp.png")
dst = os.path.join(RESULTS_DIR, "03_marker_heatmap.png")
if os.path.exists(tmp):
    shutil.move(tmp, dst)
print(f"  Saved: {RESULTS_DIR}/03_marker_heatmap.png")

# ── Figure 3: Multi-method volcano comparison ─────────────────────────
print("  Generating multi-method volcano ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
COLORS = [BLUE, RED, GREEN, "#8E44AD"]
METHOD_LABELS = {
    "wilcoxon": "Wilcoxon (default)",
    "t-test": "Welch t-test",
    "logreg": "Logistic Regression",
    "roc": "ROC AUC",
}

for ax, method, color in zip(axes, SD_METHODS, COLORS):
    df_c = sd_all_results[method][sd_all_results[method]["cluster"] == FOCUS_CLUSTER].copy()
    if method == "roc":
        # ROC: AUC > 0.7 as significance proxy, no padj
        df_c["_neg_log"] = -np.log10(df_c["padj"].clip(1e-300))
    else:
        df_c["_neg_log"] = -np.log10(df_c["padj"].clip(1e-300))

    sig_mask = (df_c["padj"] < 0.05) & (np.abs(df_c["log2fc"]) > 1)
    ax.scatter(
        df_c.loc[~sig_mask, "log2fc"], df_c.loc[~sig_mask, "_neg_log"],
        s=5, c=GRAY, alpha=0.3, linewidths=0, rasterized=True,
    )
    ax.scatter(
        df_c.loc[sig_mask, "log2fc"], df_c.loc[sig_mask, "_neg_log"],
        s=8, c=color, alpha=0.8, linewidths=0, rasterized=True,
    )
    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
    ax.axvline(1, color="gray", ls="--", lw=0.8)
    ax.axvline(-1, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("log₂FC", fontsize=10)
    ax.set_ylabel("-log₁₀(padj)", fontsize=10)
    ax.set_title(
        f"{METHOD_LABELS[method]}\n{sig_mask.sum()} sig. (padj<0.05, |log2FC|>1)",
        fontsize=10, fontweight="bold",
    )
    # Label top 5
    top5 = df_c[df_c["log2fc"] > 0].nlargest(5, "log2fc")
    for _, row in top5.iterrows():
        ax.annotate(
            row["gene"], (row["log2fc"], row["_neg_log"]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=6, color=color,
        )

fig.suptitle(
    f"Cluster {FOCUS_CLUSTER} vs Rest — Comparison Across Methods",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_multi_method_volcano.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {RESULTS_DIR}/04_multi_method_volcano.png")

# ─────────────────────────────────────────────────────────────────────
# 6. Print summary statistics
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPARISON SUMMARY -- Cluster 0 vs Rest (Wilcoxon)")
print("=" * 65)
print(f"\n  Genes tested (in common): {len(merged)}")
print(f"\n  log2FC correlation (Pearson r):  {corr_lfc:.4f}")
print(f"  p-value correlation (Pearson r): {corr_p:.4f}")
print(f"  Gene rank correlation (Spearman rho): {rho:.4f}")

sc_sig = (sc_df["padj"] < 0.05).sum()
sd_sig = (sd_df["padj"] < 0.05).sum()
sc_top10 = set(sc_df.head(10)["gene"])
sd_top10 = set(sd_df.head(10)["gene"])
print(f"\n  Significant genes (padj<0.05):")
print(f"    scanpy (Seurat-style): {sc_sig}")
print(f"    scanpy-diff:           {sd_sig}")
print(f"\n  Top-10 gene overlap: {len(sc_top10 & sd_top10)}/10")
print(f"    scanpy top-10:      {sorted(sc_top10)}")
print(f"    scanpy-diff top-10: {sorted(sd_top10)}")
print(f"    Common genes:       {sorted(sc_top10 & sd_top10)}")

print("\n  Per-cluster log2FC correlation (Pearson r):")
for c, r in zip(clusters, corr_per_cluster):
    bar = "|" * int(r * 20) if not np.isnan(r) else ""
    print(f"    Cluster {c}: {r:.4f}  {bar}")

print(f"\n  Results saved to: {RESULTS_DIR}/")
print("  Files:")
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"    {f:45s} ({size/1024:.1f} KB)")
print("\n[DONE] Analysis complete!")
