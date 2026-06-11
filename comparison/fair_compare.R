# ============================================================
# Fair comparison: import scanpy Leiden labels into Seurat
# then run FindAllMarkers on IDENTICAL cell groupings
# ============================================================
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))

RSCRIPT   <- "E:/Program Files/R/R-4.5.3/bin/x64/Rscript.exe"
OUT_DIR   <- "comparison/results_compare"
LABEL_CSV <- "comparison/results_compare/scanpy_leiden_labels.csv"

cat("=============================================================\n")
cat("  Fair Comparison: Same Leiden labels in Seurat\n")
cat("=============================================================\n\n")

# ── 1. Load Seurat object ─────────────────────────────────────────────
cat("[1/4] Loading Seurat object...\n")
pbmc <- readRDS("comparison/results_seurat/pbmc3k_seurat.rds")
cat(sprintf("  Seurat: %d cells\n", ncol(pbmc)))

# ── 2. Import scanpy Leiden labels ────────────────────────────────────
cat("[2/4] Importing scanpy Leiden labels...\n")
labels <- read.csv(LABEL_CSV, stringsAsFactors = FALSE, row.names = 1)
labels$leiden <- as.character(labels$leiden)
cat(sprintf("  Loaded %d cell labels\n", nrow(labels)))

# Match barcodes (Seurat may have "-1" suffix from 10x)
common <- intersect(colnames(pbmc), rownames(labels))
cat(sprintf("  Matching barcodes: %d / %d Seurat cells\n",
            length(common), ncol(pbmc)))

# Subset to common cells and assign Leiden clusters
pbmc_sub <- pbmc[, common]
leiden_vec <- labels[common, "leiden"]
pbmc_sub$leiden <- leiden_vec
Idents(pbmc_sub) <- "leiden"

cat("  Leiden cluster sizes in Seurat:\n")
print(sort(table(pbmc_sub$leiden)))

# ── 3. FindAllMarkers with Leiden labels ──────────────────────────────
cat("\n[3/4] Running FindAllMarkers with Leiden labels [wilcox]...\n")
t0 <- proc.time()["elapsed"]
markers_wilcox_leiden <- FindAllMarkers(
  pbmc_sub,
  test.use        = "wilcox",
  min.pct         = 0.10,
  logfc.threshold = 0.25,
  only.pos        = FALSE,
  verbose         = FALSE,
)
elapsed <- proc.time()["elapsed"] - t0
out_csv <- file.path(OUT_DIR, "seurat_wilcox_leiden_labels.csv")
write.csv(markers_wilcox_leiden, out_csv, row.names = FALSE)
cat(sprintf("  -> %d rows, %.1fs | saved: %s\n",
            nrow(markers_wilcox_leiden), elapsed, basename(out_csv)))

cat("\n[3b/4] Running FindAllMarkers with Leiden labels [t-test]...\n")
t0 <- proc.time()["elapsed"]
markers_ttest_leiden <- FindAllMarkers(
  pbmc_sub,
  test.use        = "t",
  min.pct         = 0.10,
  logfc.threshold = 0.25,
  only.pos        = FALSE,
  verbose         = FALSE,
)
elapsed <- proc.time()["elapsed"] - t0
out_csv2 <- file.path(OUT_DIR, "seurat_ttest_leiden_labels.csv")
write.csv(markers_ttest_leiden, out_csv2, row.names = FALSE)
cat(sprintf("  -> %d rows, %.1fs | saved: %s\n",
            nrow(markers_ttest_leiden), elapsed, basename(out_csv2)))

# ── 4. Compare with scanpy-diff ───────────────────────────────────────
cat("\n[4/4] Computing correlations (same cell groupings)...\n")

sd_wil <- read.csv("results/sd_wilcoxon_all_markers.csv", stringsAsFactors = FALSE)
sd_tt  <- read.csv("results/sd_t-test_all_markers.csv",   stringsAsFactors = FALSE)

pairs <- list(
  list(label = "Wilcoxon", seurat = markers_wilcox_leiden, sd = sd_wil),
  list(label = "t-test",   seurat = markers_ttest_leiden,  sd = sd_tt)
)
clusters <- sort(unique(markers_wilcox_leiden$cluster))

corr_rows <- list()
for (pr in pairs) {
  for (cl in clusters) {
    s <- pr$seurat %>% filter(cluster == cl) %>%
      select(gene, sc_lfc = avg_log2FC, sc_padj = p_val_adj)
    d <- pr$sd %>% filter(cluster == cl) %>%
      select(gene, sd_lfc = log2fc, sd_padj = padj)
    mg <- inner_join(s, d, by = "gene")
    if (nrow(mg) < 5) next
    r_lfc  <- cor(mg$sc_lfc, mg$sd_lfc, method = "pearson")
    r_padj <- cor(-log10(mg$sc_padj + 1e-300),
                  -log10(mg$sd_padj + 1e-300), method = "pearson")
    rho    <- cor(rank(-mg$sc_lfc), rank(-mg$sd_lfc), method = "spearman")
    corr_rows[[length(corr_rows) + 1]] <- data.frame(
      method = pr$label, cluster = cl,
      pearson_lfc = r_lfc, pearson_padj = r_padj, spearman_rho = rho,
      n_genes = nrow(mg)
    )
  }
}
corr_df <- bind_rows(corr_rows)

cat("\n  === Correlation Summary (SAME cell groupings) ===\n")
summary_tbl <- corr_df %>%
  group_by(method) %>%
  summarise(
    mean_r_lfc  = mean(pearson_lfc,  na.rm = TRUE),
    mean_r_padj = mean(pearson_padj, na.rm = TRUE),
    mean_rho    = mean(spearman_rho, na.rm = TRUE),
    .groups = "drop"
  )
cat(sprintf("  %-10s  %-14s  %-14s  %-14s\n",
            "Method", "r (log2FC)", "r (padj)", "rho (rank)"))
cat(strrep("-", 56), "\n")
for (i in seq_len(nrow(summary_tbl))) {
  cat(sprintf("  %-10s  %-14.4f  %-14.4f  %-14.4f\n",
              summary_tbl$method[i],
              summary_tbl$mean_r_lfc[i],
              summary_tbl$mean_r_padj[i],
              summary_tbl$mean_rho[i]))
}
write.csv(corr_df, file.path(OUT_DIR, "fair_comparison_correlations.csv"), row.names = FALSE)

# ── Figure: Comparison bar chart (before vs after) ───────────────────
old_corr <- read.csv(file.path(OUT_DIR, "per_cluster_correlations.csv"),
                     stringsAsFactors = FALSE)
old_summary <- old_corr %>%
  filter(method %in% c("Wilcoxon", "t-test")) %>%
  group_by(method) %>%
  summarise(mean_r_lfc = mean(pearson_lfc, na.rm = TRUE), .groups = "drop") %>%
  mutate(comparison = "Different clusters\n(Louvain vs Leiden)")

new_summary <- summary_tbl %>%
  filter(method %in% c("Wilcoxon", "t-test")) %>%
  select(method, mean_r_lfc) %>%
  mutate(comparison = "Same clusters\n(Leiden in both)")

combined <- bind_rows(old_summary, new_summary)
combined$comparison <- factor(combined$comparison,
  levels = c("Different clusters\n(Louvain vs Leiden)",
             "Same clusters\n(Leiden in both)"))

p <- ggplot(combined, aes(x = method, y = mean_r_lfc, fill = comparison)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.88) +
  geom_text(aes(label = sprintf("r=%.4f", mean_r_lfc)),
            position = position_dodge(0.7), vjust = -0.5,
            size = 4, fontface = "bold") +
  scale_fill_manual(
    values = c("Different clusters\n(Louvain vs Leiden)" = "#E74C3C",
               "Same clusters\n(Leiden in both)"         = "#27AE60"),
    name = "Comparison type"
  ) +
  ylim(0, 1.15) +
  labs(
    x = "Method",
    y = "Mean Pearson r (log2FC, across clusters)",
    title = "Fair vs Unfair Comparison",
    subtitle = "Using the same Leiden cluster labels eliminates the clustering algorithm bias"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey30", size = 10),
    legend.position = "top",
    legend.title = element_text(face = "bold")
  )
ggsave(file.path(OUT_DIR, "06_fair_vs_unfair_comparison.png"), p,
       width = 8, height = 6, dpi = 150)
cat("\n  Saved: 06_fair_vs_unfair_comparison.png\n")

# Per-cluster fair comparison bar
p2 <- ggplot(corr_df, aes(x = cluster, y = pearson_lfc, fill = pearson_lfc)) +
  geom_col(alpha = 0.88, width = 0.7) +
  geom_text(aes(label = sprintf("%.4f", pearson_lfc)),
            vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#E74C3C", mid = "#F39C12", high = "#27AE60",
                       midpoint = 0.88, limits = c(0.7, 1.0), name = "r") +
  facet_wrap(~method) +
  ylim(0, 1.12) +
  labs(
    x = "Leiden Cluster", y = "Pearson r (log2FC)",
    title = "Per-Cluster log2FC Correlation (FAIR: same Leiden labels)",
    subtitle = "Seurat FindAllMarkers vs scanpy-diff find_all_markers"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "#2C3E50"),
    strip.text = element_text(color = "white", face = "bold")
  )
ggsave(file.path(OUT_DIR, "07_fair_per_cluster_correlation.png"), p2,
       width = 10, height = 5, dpi = 150)
cat("  Saved: 07_fair_per_cluster_correlation.png\n")

cat("\n[DONE]\n")
