"""
深度诊断 log2FC 差异来源
对比 Seurat 与 scanpy-diff 的 log2FC 计算公式
"""
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

adata = sc.read_h5ad("results/pbmc3k_processed.h5ad")

# 加载对比结果
seurat_df = pd.read_csv("comparison/results_compare/seurat_wilcox_leiden_labels.csv")
sd_df     = pd.read_csv("results/sd_wilcoxon_all_markers.csv")

# 聚焦 cluster 0
seurat_c0 = seurat_df[seurat_df["cluster"] == "0"].set_index("gene")
sd_c0     = sd_df[sd_df["cluster"] == "0"].set_index("gene")

# 取共同基因
common = seurat_c0.index.intersection(sd_c0.index)
print(f"Common genes: {len(common)}")

# 取几个代表性基因分析
test_genes = ["IL7R", "CD3D", "LTB", "S100A4", "CD14", "LDHB"]
test_genes = [g for g in test_genes if g in common]

# ── 手工计算两种方法的 log2FC ─────────────────────────────────────────
mask_g0 = adata.obs["leiden"] == "0"
mask_rest = ~mask_g0

# scanpy-diff 用 adata.X（log1p 归一化）
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)

# raw 对象保存的也是 log1p 归一化数据
X_raw = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
raw_genes = list(adata.raw.var_names)

print("\n" + "=" * 80)
print(f"{'Gene':12s}  {'Seurat':>10s}  {'sd-diff':>10s}  {'Δ':>8s}  Analysis")
print("=" * 80)

results = []
for gene in test_genes:
    if gene not in adata.var_names:
        print(f"{gene:12s}  not in adata.var_names, skipping")
        continue

    gi = list(adata.var_names).index(gene)
    expr_X = X[:, gi]  # log1p normalized (used by scanpy-diff)

    # --- scanpy-diff 公式 (compute_log2fc) ---
    # X は log1p scale → expm1 → mean → log2(mean+1) - log2(mean_rest+1)
    mean_g0_natural   = np.expm1(expr_X[mask_g0]).mean()
    mean_rest_natural = np.expm1(expr_X[mask_rest]).mean()
    sd_lfc_manual = np.log2(mean_g0_natural + 1) - np.log2(mean_rest_natural + 1)

    # --- Seurat 公式 (from Seurat source CalcDiffExpression.R) ---
    # Seurat uses: log(mean(exp(x) - 1) + 1) / log(2)
    # But actually Seurat's avg_log2FC is computed as:
    # mean_log_group - mean_log_rest (in log1p scale), then divide by log(2)
    # i.e., in log-space directly
    mean_log_g0   = expr_X[mask_g0].mean()    # mean of log1p values
    mean_log_rest = expr_X[mask_rest].mean()
    seurat_lfc_logspace = (mean_log_g0 - mean_log_rest) / np.log(2)

    # Seurat v5 FoldChange() source:
    # data.1 <- apply(data[, cells.1, drop = FALSE], 1, function(x) log(x = mean(x = expm1(x = x)) + pseudocount.use, base = 2))
    # data.2 <- apply(data[, cells.2, drop = FALSE], 1, function(x) log(x = mean(x = expm1(x = x)) + pseudocount.use, base = 2))
    # fc <- data.1 - data.2
    seurat_lfc_v5_manual = (
        np.log2(mean_g0_natural   + 1) -   # log2(mean(expm1(X_group)) + 1)
        np.log2(mean_rest_natural + 1)      # log2(mean(expm1(X_rest))  + 1)
    )

    # 真实值来自 CSV
    seurat_actual = seurat_c0.loc[gene, "avg_log2FC"] if gene in seurat_c0.index else np.nan
    sd_actual     = sd_c0.loc[gene, "log2fc"]         if gene in sd_c0.index     else np.nan

    print(f"{gene:12s}  {seurat_actual:>10.4f}  {sd_actual:>10.4f}  {seurat_actual-sd_actual:>+8.4f}")
    print(f"  Manual Seurat v5 (expm1+mean+log2): {seurat_lfc_v5_manual:>+8.4f}")
    print(f"  Manual scanpy-diff (same formula):  {sd_lfc_manual:>+8.4f}")
    print(f"  Seurat logspace (mean_log/log2):    {seurat_lfc_logspace:>+8.4f}")
    print()

    results.append({
        "gene": gene,
        "seurat": seurat_actual,
        "sd_diff": sd_actual,
        "diff": seurat_actual - sd_actual,
        "manual_v5": seurat_lfc_v5_manual,
        "manual_sd": sd_lfc_manual,
        "seurat_logspace": seurat_lfc_logspace,
        "mean_g0_natural": mean_g0_natural,
        "mean_rest_natural": mean_rest_natural,
        "mean_log_g0": mean_log_g0,
        "mean_log_rest": mean_log_rest,
    })

# ── 全局统计 ───────────────────────────────────────────────────────────
print("=" * 80)
mg = pd.merge(
    seurat_c0[["avg_log2FC"]].rename(columns={"avg_log2FC": "seurat"}),
    sd_c0[["log2fc"]].rename(columns={"log2fc": "sd"}),
    left_index=True, right_index=True
)
mg["diff"] = mg["seurat"] - mg["sd"]

print(f"\nGlobal diff stats (Seurat - scanpy-diff), n={len(mg)} genes:")
print(f"  Mean diff:   {mg['diff'].mean():+.6f}")
print(f"  Median diff: {mg['diff'].median():+.6f}")
print(f"  Std diff:    {mg['diff'].std():.6f}")
print(f"  Max diff:    {mg['diff'].max():+.6f}")
print(f"  Min diff:    {mg['diff'].min():+.6f}")

# Check if it's a systematic offset or random noise
from scipy import stats as scipy_stats
corr, _ = scipy_stats.pearsonr(mg["seurat"], mg["sd"])
slope, intercept, r, p, se = scipy_stats.linregress(mg["sd"], mg["seurat"])
print(f"\n  Regression sd -> seurat:")
print(f"  slope={slope:.6f}  intercept={intercept:.6f}  r={r:.6f}")
print(f"  => seurat ≈ {slope:.4f} * sd_diff + {intercept:.4f}")

# ── 关键: 找哪个 X 槽位被用到 ─────────────────────────────────────────
print("\n" + "=" * 80)
print("Data layers available:", list(adata.layers.keys()))
print("adata.X range:   min={:.3f}, max={:.3f}".format(float(X.min()), float(X.max())))
print("adata.raw.X exists:", adata.raw is not None)
if adata.raw is not None:
    X_raw_arr = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else np.array(adata.raw.X)
    print("adata.raw.X range: min={:.3f}, max={:.3f}".format(float(X_raw_arr.min()), float(X_raw_arr.max())))

# scanpy-diff 用哪个?
print("\nNote: scanpy-diff._get_expression_matrix() uses adata.X by default")
print("      Seurat uses the 'data' slot (= log-normalized, same as adata.X)")
