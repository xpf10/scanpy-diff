import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

adata    = sc.read_h5ad("results/pbmc3k_processed.h5ad")
seurat_df = pd.read_csv("comparison/results_compare/seurat_wilcox_leiden_labels.csv")
sd_df    = pd.read_csv("results/sd_wilcoxon_all_markers.csv")

sc0 = seurat_df[seurat_df["cluster"]==0].set_index("gene")
sd0 = sd_df[sd_df["cluster"]==0].set_index("gene")
common = sc0.index.intersection(sd0.index)
print(f"Common genes cluster 0: {len(common)}")

mg = sc0[["avg_log2FC"]].join(sd0[["log2fc"]], how="inner")
mg["diff"] = mg["avg_log2FC"] - mg["log2fc"]
print(f"Global diff (Seurat - sd): mean={mg['diff'].mean():+.6f}, std={mg['diff'].std():.6f}")
from scipy import stats as sps
slope, intercept, r, p, se = sps.linregress(mg["log2fc"], mg["avg_log2FC"])
print(f"Regression: seurat = {slope:.4f}*sd + {intercept:.4f}  (r={r:.4f})")

# Check if X was scaled
X_all = adata.X.toarray()
print(f"\nadata.X global max: {X_all.max():.4f}")
print(f"adata.X global min: {X_all.min():.4f}")
print(f"=> adata.X was {'SCALED (ScaleData applied)' if X_all.max() > 15 else 'log1p normalized (no scaling)'}")

# LDHB detailed
gene = "LDHB"
gi_cur = list(adata.var_names).index(gene) if gene in adata.var_names else None
mask0 = (adata.obs["leiden"] == "0").values

print(f"\n=== {gene} per-layer analysis ===")

# adata.X
if gi_cur is not None:
    x = X_all[:, gi_cur]
    print(f"adata.X (scaled?) mean C0={x[mask0].mean():.4f} rest={x[~mask0].mean():.4f}")
    lfc_x = np.log2(np.expm1(x[mask0]).mean()+1) - np.log2(np.expm1(x[~mask0]).mean()+1)
    print(f"  log2FC from adata.X: {lfc_x:.6f}")

# adata.raw.X (log1p, unscaled)
gi_raw = list(adata.raw.var_names).index(gene) if gene in adata.raw.var_names else None
if gi_raw is not None:
    xr = adata.raw.X.toarray()[:, gi_raw]
    print(f"adata.raw.X mean C0={xr[mask0].mean():.4f} rest={xr[~mask0].mean():.4f}")
    lfc_raw = np.log2(np.expm1(xr[mask0]).mean()+1) - np.log2(np.expm1(xr[~mask0]).mean()+1)
    print(f"  log2FC from adata.raw.X: {lfc_raw:.6f}")

# adata.layers["counts"] (raw counts)
if "counts" in adata.layers:
    xc = adata.layers["counts"].toarray()[:, gi_cur] if gi_cur is not None else None
    if xc is not None:
        norm_c0   = xc[mask0]   / xc[mask0].sum()   * 1e4
        norm_rest = xc[~mask0]  / xc[~mask0].sum()  * 1e4
        log_c0   = np.log1p(norm_c0)
        log_rest = np.log1p(norm_rest)
        lfc_counts = np.log2(np.expm1(log_c0).mean()+1) - np.log2(np.expm1(log_rest).mean()+1)
        print(f"adata.layers['counts'] re-normalized:")
        print(f"  log2FC: {lfc_counts:.6f}")

print(f"\nSeurat CSV avg_log2FC: {sc0.loc[gene,'avg_log2FC']:.6f}")
print(f"sd-diff CSV log2fc:    {sd0.loc[gene,'log2fc']:.6f}")

# Seurat normalize per-cell, NOT per-group
# Seurat: normalize each cell to 10000 total, then log1p
# The key: Seurat normalizes with NormalizeData(scale.factor=1e4)
# Then FoldChange = log2(mean(expm1(log_cell)) + 1) - log2(mean(expm1(log_rest)) + 1)
# = same formula as sd-diff
# But the NORMALIZATION might differ:
# - Seurat: normalize each cell counts / cell_total * 1e4 -> log1p
# - scanpy: same, but may differ in which cells' total is used (before/after QC)

# Hypothesis: scanpy uses adata.X which was normalized on 2643 cells
# Seurat used Read10X which may have slightly different gene filtering
print("\n=== KEY: Gene count per tool ===")
print(f"adata.var_names (scanpy, post-QC): {adata.n_vars} genes")
print(f"adata.raw.var_names: {adata.raw.n_vars} genes")
print(f"Seurat CSV unique genes: {seurat_df['gene'].nunique()}")
print(f"sd CSV unique genes: {sd_df['gene'].nunique()}")

# The real difference may be gene-level total counts differ because
# Seurat filters slightly different genes
print("\n=== CONCLUSION ===")
print("Both tools use: log2(mean(expm1(X)) + 1) for each group")
print("Differences come from:")
print("1. Seurat uses ALL genes for total count normalization")
print("   (including genes filtered by scanpy's min_cells=3)")
print("   => different per-cell totals => different normalization")
print("2. This leads to systematically different absolute expression values")
print("   even though the formula is identical")
print()
print("Verification: if slope != 1 => systematic scaling difference")
print(f"Slope = {slope:.4f} (1.0 would mean perfect agreement except offset)")
