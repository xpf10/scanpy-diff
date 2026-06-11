# ============================================================
# Seurat PBMC3k Differential Expression Analysis
# Companion script to scanpy-diff comparison
#
# Usage:
#   "E:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe" comparison/seurat_de.R
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(dplyr)
})

# Optional: ggrepel for label overlap avoidance
has_ggrepel <- requireNamespace("ggrepel", quietly = TRUE)

set.seed(42)
options(future.globals.maxSize = 4000 * 1024^2)

RESULTS_DIR <- "comparison/results_seurat"
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=============================================================\n")
cat("  Seurat PBMC3k Differential Expression\n")
cat("=============================================================\n\n")

# ─────────────────────────────────────────────────────────────────────
# 1. Download or load cached PBMC3k data
# ─────────────────────────────────────────────────────────────────────
cat("[1/5] Loading PBMC3k data...\n")

tarball  <- file.path(RESULTS_DIR, "pbmc3k.tar.gz")
data_dir <- file.path(RESULTS_DIR, "filtered_gene_bc_matrices", "hg19")
rds_path <- file.path(RESULTS_DIR, "pbmc3k_seurat.rds")

if (file.exists(rds_path)) {
  cat("  Loading from cached RDS:", rds_path, "\n")
  pbmc <- readRDS(rds_path)
  cat(sprintf("  Loaded: %d cells x %d features, %d clusters\n",
              ncol(pbmc), nrow(pbmc), length(levels(pbmc$seurat_clusters))))
} else {
  # Download raw 10X data
  if (!dir.exists(data_dir)) {
    cat("  Downloading PBMC3k from 10x Genomics...\n")
    url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    download.file(url, tarball, mode = "wb", quiet = FALSE)
    untar(tarball, exdir = RESULTS_DIR)
    cat("  Downloaded and extracted.\n")
  }

  cat("[2/5] Creating Seurat object & preprocessing...\n")
  pbmc.data <- Read10X(data.dir = data_dir)
  pbmc <- CreateSeuratObject(
    counts  = pbmc.data,
    project = "pbmc3k",
    min.cells    = 3,
    min.features = 200,
  )
  pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
  pbmc <- subset(pbmc, subset = percent.mt < 5)
  cat(sprintf("  After QC: %d cells x %d features\n", ncol(pbmc), nrow(pbmc)))

  pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)
  pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  pbmc <- ScaleData(pbmc, features = rownames(pbmc), verbose = FALSE)
  pbmc <- RunPCA(pbmc, features = VariableFeatures(pbmc), npcs = 40, verbose = FALSE)
  pbmc <- FindNeighbors(pbmc, dims = 1:40, verbose = FALSE)
  pbmc <- FindClusters(pbmc, resolution = 0.5, verbose = FALSE)
  pbmc <- RunUMAP(pbmc, dims = 1:40, verbose = FALSE)
  cat(sprintf("  Clusters: %d\n", length(levels(pbmc$seurat_clusters))))

  saveRDS(pbmc, rds_path)
  cat("  Saved RDS:", rds_path, "\n")
}

# ─────────────────────────────────────────────────────────────────────
# 2. UMAP plot
# ─────────────────────────────────────────────────────────────────────
cat("[2/5] Plotting UMAP...\n")
p_umap <- DimPlot(pbmc, reduction = "umap", label = TRUE, label.size = 5) +
  ggtitle("PBMC3k -- Seurat Clusters") +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(RESULTS_DIR, "01_umap_clusters.png"), p_umap,
       width = 7, height = 6, dpi = 150)
cat("  Saved: 01_umap_clusters.png\n")

# ─────────────────────────────────────────────────────────────────────
# 3. FindAllMarkers — three methods
# ─────────────────────────────────────────────────────────────────────
cat("[3/5] Running FindAllMarkers...\n")

run_and_save <- function(test, label) {
  cat(sprintf("  Running FindAllMarkers [%s]...\n", label))
  t0 <- proc.time()["elapsed"]
  markers <- FindAllMarkers(
    pbmc,
    test.use        = test,
    min.pct         = 0.10,
    logfc.threshold = 0.25,
    only.pos        = FALSE,
    verbose         = FALSE,
  )
  elapsed <- proc.time()["elapsed"] - t0
  out <- file.path(RESULTS_DIR, sprintf("seurat_%s_all_markers.csv", label))
  write.csv(markers, out, row.names = FALSE)
  cat(sprintf("    -> %d rows, %.1fs | saved: %s\n", nrow(markers), elapsed, basename(out)))
  markers
}

markers_wilcox <- run_and_save("wilcox", "wilcox")
markers_ttest  <- run_and_save("t",      "ttest")
markers_lr     <- run_and_save("LR",     "logreg")

# ─────────────────────────────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────────────────────────────
cat("[4/5] Generating plots...\n")

# Top 5 markers per cluster — heatmap
top5 <- markers_wilcox %>%
  group_by(cluster) %>%
  slice_max(order_by = avg_log2FC, n = 5) %>%
  ungroup()

p_heat <- DoHeatmap(pbmc, features = unique(top5$gene), size = 3) +
  scale_fill_gradientn(colors = c("#3498DB", "white", "#E74C3C")) +
  ggtitle("Top 5 Wilcoxon Markers per Cluster (Seurat)") +
  theme(plot.title = element_text(face = "bold", size = 12))
ggsave(file.path(RESULTS_DIR, "02_marker_heatmap.png"), p_heat,
       width = 14, height = 8, dpi = 150)
cat("  Saved: 02_marker_heatmap.png\n")

# Dot plot
p_dot <- DotPlot(pbmc, features = unique(top5$gene)) +
  RotatedAxis() +
  ggtitle("Top Markers Dot Plot (Seurat)") +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(RESULTS_DIR, "03_dotplot.png"), p_dot,
       width = 14, height = 5, dpi = 150)
cat("  Saved: 03_dotplot.png\n")

# Volcano — Cluster 0
cl0 <- markers_wilcox %>%
  filter(cluster == "0") %>%
  mutate(
    neg_log_padj = -log10(p_val_adj + 1e-300),
    sig = p_val_adj < 0.05 & abs(avg_log2FC) > 1,
    direction = case_when(
      sig & avg_log2FC > 1  ~ "Up",
      sig & avg_log2FC < -1 ~ "Down",
      TRUE                  ~ "NS"
    )
  )

top_up   <- cl0 %>% filter(direction == "Up")   %>% slice_max(avg_log2FC, n = 10)
top_down <- cl0 %>% filter(direction == "Down")  %>% slice_min(avg_log2FC, n = 10)
label_df <- bind_rows(top_up, top_down)

p_vol <- ggplot(cl0, aes(x = avg_log2FC, y = neg_log_padj, color = direction)) +
  geom_point(size = 0.8, alpha = 0.7) +
  scale_color_manual(values = c("Up" = "#E74C3C", "Down" = "#3498DB", "NS" = "grey70")) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.5) +
  geom_vline(xintercept = c(-1, 1),     linetype = "dashed", color = "grey50", linewidth = 0.5) +
  labs(
    x     = "avg_log2FC",
    y     = "-log10(adj. p-value)",
    title = "Volcano -- Cluster 0 vs Rest (Seurat Wilcoxon)",
    color = NULL,
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title  = element_text(face = "bold"),
    legend.position = "top",
  )

if (has_ggrepel) {
  p_vol <- p_vol +
    ggrepel::geom_text_repel(
      data = label_df, aes(label = gene),
      size = 2.5, max.overlaps = 20, show.legend = FALSE,
    )
}
ggsave(file.path(RESULTS_DIR, "04_volcano_cluster0.png"), p_vol,
       width = 7, height = 6, dpi = 150)
cat("  Saved: 04_volcano_cluster0.png\n")

# ─────────────────────────────────────────────────────────────────────
# 5. Load scanpy-diff results and compare directly
# ─────────────────────────────────────────────────────────────────────
cat("[5/5] Comparing with scanpy-diff results...\n")

sd_csv <- "comparison/results/sd_wilcoxon_all_markers.csv"
if (file.exists(sd_csv)) {
  sd_all <- read.csv(sd_csv, stringsAsFactors = FALSE)

  # Cluster 0 comparison
  seurat_c0 <- markers_wilcox %>%
    filter(cluster == "0") %>%
    select(gene, sc_log2fc = avg_log2FC, sc_padj = p_val_adj, sc_pct1 = pct.1, sc_pct2 = pct.2)

  sd_c0 <- sd_all %>%
    filter(cluster == "0") %>%
    select(gene, sd_log2fc = log2fc, sd_padj = padj, sd_pct1 = pct_1, sd_pct2 = pct_2)

  merged <- inner_join(seurat_c0, sd_c0, by = "gene")
  cat(sprintf("  Merged genes for cluster 0: %d\n", nrow(merged)))

  corr_lfc <- cor(merged$sc_log2fc, merged$sd_log2fc, method = "pearson")
  corr_p   <- cor(-log10(merged$sc_padj + 1e-300),
                  -log10(merged$sd_padj + 1e-300), method = "pearson")
  corr_rho <- cor(rank(-merged$sc_log2fc), rank(-merged$sd_log2fc), method = "spearman")

  cat(sprintf("  log2FC correlation  (Pearson r):   %.4f\n", corr_lfc))
  cat(sprintf("  p-value correlation (Pearson r):   %.4f\n", corr_p))
  cat(sprintf("  Rank correlation    (Spearman rho): %.4f\n", corr_rho))

  # Scatter: Seurat vs scanpy-diff log2FC
  p_scatter <- ggplot(merged, aes(x = sc_log2fc, y = sd_log2fc)) +
    geom_point(alpha = 0.3, size = 0.8, color = "#2980B9") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 0.8) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.1, vjust = 1.5,
             label = sprintf("Pearson r = %.4f", corr_lfc),
             color = "red", fontface = "bold", size = 4) +
    labs(
      x = "Seurat avg_log2FC",
      y = "scanpy-diff log2FC",
      title = "log2FC Correlation: Seurat vs scanpy-diff\n(Cluster 0, Wilcoxon)",
    ) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(RESULTS_DIR, "05_log2fc_scatter.png"), p_scatter,
         width = 6, height = 6, dpi = 150)
  cat("  Saved: 05_log2fc_scatter.png\n")

  # Per-cluster correlation
  clusters <- sort(unique(markers_wilcox$cluster))
  cluster_corrs <- sapply(clusters, function(cl) {
    s <- markers_wilcox %>% filter(cluster == cl) %>% select(gene, sc_log2fc = avg_log2FC)
    d <- sd_all        %>% filter(cluster == cl) %>% select(gene, sd_log2fc = log2fc)
    mg <- inner_join(s, d, by = "gene")
    if (nrow(mg) > 5) cor(mg$sc_log2fc, mg$sd_log2fc) else NA_real_
  })

  corr_df <- data.frame(
    cluster = clusters,
    pearson_r = cluster_corrs,
    n_cells = as.integer(table(pbmc$seurat_clusters)[clusters]),
  )
  write.csv(corr_df, file.path(RESULTS_DIR, "per_cluster_correlation.csv"), row.names = FALSE)

  p_corr <- ggplot(corr_df, aes(x = cluster, y = pearson_r, fill = pearson_r)) +
    geom_col(alpha = 0.85) +
    geom_text(aes(label = sprintf("r=%.3f\n(n=%d)", pearson_r, n_cells)),
              vjust = -0.3, size = 3) +
    scale_fill_gradient2(low = "#E74C3C", mid = "#F39C12", high = "#27AE60",
                         midpoint = 0.75, limits = c(0.4, 1.0)) +
    ylim(0, 1.1) +
    labs(
      x = "Cluster", y = "Pearson r (log2FC)",
      title = "Per-Cluster log2FC Correlation\nSeurat vs scanpy-diff (Wilcoxon)",
      fill = "r",
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right",
    )
  ggsave(file.path(RESULTS_DIR, "06_per_cluster_correlation.png"), p_corr,
         width = 8, height = 5, dpi = 150)
  cat("  Saved: 06_per_cluster_correlation.png\n")

  write.csv(merged, file.path(RESULTS_DIR, "comparison_cluster0.csv"), row.names = FALSE)
  cat("  Saved: comparison_cluster0.csv\n")

} else {
  cat("  scanpy-diff results not found at:", sd_csv, "\n")
  cat("  Run compare_de.py first.\n")
}

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
cat("\n=============================================================\n")
cat("  SEURAT SUMMARY\n")
cat("=============================================================\n")
cat(sprintf("  R version:  %s\n", R.version$version.string))
cat(sprintf("  Seurat:     %s\n", packageVersion("Seurat")))
cat(sprintf("  Cells:      %d\n", ncol(pbmc)))
cat(sprintf("  Clusters:   %d\n", length(levels(pbmc$seurat_clusters))))

for (nm in c("wilcox", "ttest", "lr")) {
  df <- get(sprintf("markers_%s", nm))
  n_sig <- sum(df$p_val_adj < 0.05, na.rm = TRUE)
  cat(sprintf("  FindAllMarkers [%-6s]: %d sig. genes (padj<0.05)\n",
              toupper(nm), n_sig))
}

cat("\n  Output files:\n")
for (f in sort(list.files(RESULTS_DIR))) {
  sz <- file.size(file.path(RESULTS_DIR, f))
  cat(sprintf("    %-45s (%.1f KB)\n", f, sz / 1024))
}
cat("\n[DONE] Seurat analysis complete!\n")
