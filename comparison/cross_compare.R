# ============================================================
# Cross-comparison: Seurat vs scanpy-diff
# Reads CSVs from both tools and generates comparison figures
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})
has_ggrepel <- requireNamespace("ggrepel", quietly = TRUE)

SEURAT_DIR  <- "comparison/results_seurat"
SD_DIR      <- "results"
OUT_DIR     <- "comparison/results_compare"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=============================================================\n")
cat("  Seurat vs scanpy-diff Cross-Comparison\n")
cat("=============================================================\n\n")

# ── Load data ─────────────────────────────────────────────────────────
cat("[1/3] Loading results...\n")

seurat_wil <- read.csv(file.path(SEURAT_DIR, "seurat_wilcox_all_markers.csv"),  stringsAsFactors = FALSE)
seurat_tt  <- read.csv(file.path(SEURAT_DIR, "seurat_ttest_all_markers.csv"),   stringsAsFactors = FALSE)
seurat_lr  <- read.csv(file.path(SEURAT_DIR, "seurat_logreg_all_markers.csv"),  stringsAsFactors = FALSE)
sd_wil     <- read.csv(file.path(SD_DIR,     "sd_wilcoxon_all_markers.csv"),    stringsAsFactors = FALSE)
sd_tt      <- read.csv(file.path(SD_DIR,     "sd_t-test_all_markers.csv"),      stringsAsFactors = FALSE)
sd_lr      <- read.csv(file.path(SD_DIR,     "sd_logreg_all_markers.csv"),      stringsAsFactors = FALSE)

cat(sprintf("  Seurat  wilcox: %d rows\n", nrow(seurat_wil)))
cat(sprintf("  Seurat  t-test: %d rows\n", nrow(seurat_tt)))
cat(sprintf("  Seurat  logreg: %d rows\n", nrow(seurat_lr)))
cat(sprintf("  sd-diff wilcox: %d rows\n", nrow(sd_wil)))
cat(sprintf("  sd-diff t-test: %d rows\n", nrow(sd_tt)))
cat(sprintf("  sd-diff logreg: %d rows\n", nrow(sd_lr)))

# ── Helper: merge one cluster ─────────────────────────────────────────
merge_cluster <- function(seurat_df, sd_df, cl,
                          sc_lfc_col = "avg_log2FC",
                          sc_padj_col = "p_val_adj") {
  s <- seurat_df %>%
    filter(cluster == cl) %>%
    select(gene,
           sc_lfc  = all_of(sc_lfc_col),
           sc_padj = all_of(sc_padj_col),
           sc_pct1 = pct.1,
           sc_pct2 = pct.2)
  d <- sd_df %>%
    filter(cluster == cl) %>%
    select(gene,
           sd_lfc  = log2fc,
           sd_padj = padj,
           sd_pct1 = pct_1,
           sd_pct2 = pct_2)
  inner_join(s, d, by = "gene")
}

# ── Per-cluster correlations for all method pairs ─────────────────────
pairs <- list(
  list(label = "Wilcoxon",  seurat = seurat_wil, sd = sd_wil),
  list(label = "t-test",    seurat = seurat_tt,  sd = sd_tt),
  list(label = "Logreg",    seurat = seurat_lr,  sd = sd_lr)
)

clusters <- sort(unique(seurat_wil$cluster))

corr_rows <- list()
for (pr in pairs) {
  for (cl in clusters) {
    mg <- tryCatch(merge_cluster(pr$seurat, pr$sd, cl), error = function(e) NULL)
    if (is.null(mg) || nrow(mg) < 5) next
    r_lfc  <- cor(mg$sc_lfc,  mg$sd_lfc,  method = "pearson")
    r_padj <- cor(-log10(mg$sc_padj + 1e-300),
                  -log10(mg$sd_padj + 1e-300), method = "pearson")
    rho    <- cor(rank(-mg$sc_lfc), rank(-mg$sd_lfc), method = "spearman")
    n_sig_s <- sum(mg$sc_padj < 0.05, na.rm = TRUE)
    n_sig_d <- sum(mg$sd_padj < 0.05, na.rm = TRUE)
    corr_rows[[length(corr_rows) + 1]] <- data.frame(
      method = pr$label, cluster = cl,
      pearson_lfc = r_lfc, pearson_padj = r_padj, spearman_rho = rho,
      n_genes = nrow(mg), n_sig_seurat = n_sig_s, n_sig_sd = n_sig_d
    )
  }
}
corr_df <- bind_rows(corr_rows)
write.csv(corr_df, file.path(OUT_DIR, "per_cluster_correlations.csv"), row.names = FALSE)
cat("\n[2/3] Generating plots...\n")

# ── Figure 1: Per-cluster log2FC correlation (faceted by method) ──────
p1 <- ggplot(corr_df, aes(x = cluster, y = pearson_lfc, fill = pearson_lfc)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", pearson_lfc)),
            vjust = -0.4, size = 3.2, fontface = "bold") +
  scale_fill_gradient2(
    low = "#E74C3C", mid = "#F39C12", high = "#27AE60",
    midpoint = 0.75, limits = c(0.3, 1.0), name = "r"
  ) +
  facet_wrap(~method, ncol = 3) +
  ylim(0, 1.18) +
  labs(
    x = "Cluster", y = "Pearson r (log2FC)",
    title = "Per-Cluster log2FC Correlation: Seurat vs scanpy-diff",
    subtitle = "Higher r = more consistent results between the two tools"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "grey40"),
    strip.background = element_rect(fill = "#2C3E50"),
    strip.text       = element_text(color = "white", face = "bold"),
    legend.position  = "right"
  )
ggsave(file.path(OUT_DIR, "01_per_cluster_lfc_correlation.png"), p1,
       width = 13, height = 5, dpi = 150)
cat("  Saved: 01_per_cluster_lfc_correlation.png\n")

# ── Figure 2: p-value correlation faceted ────────────────────────────
p2 <- ggplot(corr_df, aes(x = cluster, y = pearson_padj, fill = pearson_padj)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", pearson_padj)),
            vjust = -0.4, size = 3.2, fontface = "bold") +
  scale_fill_gradient2(
    low = "#E74C3C", mid = "#F39C12", high = "#27AE60",
    midpoint = 0.85, limits = c(0.4, 1.0), name = "r"
  ) +
  facet_wrap(~method, ncol = 3) +
  ylim(0, 1.18) +
  labs(
    x = "Cluster", y = "Pearson r (-log10 padj)",
    title = "Per-Cluster p-value Correlation: Seurat vs scanpy-diff"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "#2C3E50"),
    strip.text = element_text(color = "white", face = "bold"),
    legend.position = "right"
  )
ggsave(file.path(OUT_DIR, "02_per_cluster_pval_correlation.png"), p2,
       width = 13, height = 5, dpi = 150)
cat("  Saved: 02_per_cluster_pval_correlation.png\n")

# ── Figure 3: log2FC scatter (Wilcoxon, Cluster 0) ────────────────────
mg0 <- merge_cluster(seurat_wil, sd_wil, clusters[1])
r0  <- cor(mg0$sc_lfc, mg0$sd_lfc)

# Color by significance agreement
mg0 <- mg0 %>% mutate(
  category = case_when(
    sc_padj < 0.05 & sd_padj < 0.05 ~ "Both sig.",
    sc_padj < 0.05 & sd_padj >= 0.05 ~ "Seurat only",
    sc_padj >= 0.05 & sd_padj < 0.05 ~ "sd-diff only",
    TRUE ~ "Neither"
  )
)
cols <- c("Both sig." = "#27AE60", "Seurat only" = "#2980B9",
          "sd-diff only" = "#E74C3C", "Neither" = "grey70")

p3 <- ggplot(mg0, aes(x = sc_lfc, y = sd_lfc, color = category)) +
  geom_point(alpha = 0.4, size = 0.9) +
  geom_abline(slope = 1, intercept = 0, color = "black",
              linetype = "dashed", linewidth = 0.8) +
  scale_color_manual(values = cols, name = "Significance") +
  annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.6,
           label = sprintf("Pearson r = %.4f\nn = %d genes", r0, nrow(mg0)),
           color = "#2C3E50", fontface = "bold", size = 4.2) +
  labs(
    x = "Seurat avg_log2FC (Wilcoxon)",
    y = "scanpy-diff log2FC (Wilcoxon)",
    title = sprintf("log2FC Scatter: Seurat vs scanpy-diff\nCluster %s vs Rest", clusters[1])
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

if (has_ggrepel) {
  top_genes <- mg0 %>%
    filter(category == "Both sig.", sc_lfc > 0) %>%
    slice_max(sc_lfc, n = 10)
  p3 <- p3 +
    ggrepel::geom_text_repel(
      data = top_genes, aes(label = gene),
      size = 2.5, color = "#27AE60", max.overlaps = 15, show.legend = FALSE
    )
}
ggsave(file.path(OUT_DIR, "03_lfc_scatter_cluster0.png"), p3,
       width = 7, height = 6, dpi = 150)
cat("  Saved: 03_lfc_scatter_cluster0.png\n")

# ── Figure 4: Significant gene counts comparison ──────────────────────
sig_rows <- list()
for (pr in pairs) {
  for (cl in clusters) {
    n_s <- sum(pr$seurat$cluster == cl & pr$seurat$p_val_adj < 0.05, na.rm = TRUE)
    n_d <- sum(pr$sd$cluster     == cl & pr$sd$padj           < 0.05, na.rm = TRUE)
    sig_rows[[length(sig_rows) + 1]] <- data.frame(
      method = pr$label, cluster = cl,
      Seurat = n_s, scanpy_diff = n_d
    )
  }
}
sig_df <- bind_rows(sig_rows)

library(tidyr)
sig_long <- sig_df %>%
  pivot_longer(c(Seurat, scanpy_diff), names_to = "tool", values_to = "n_sig") %>%
  mutate(tool = recode(tool, "scanpy_diff" = "scanpy-diff"))

p4 <- ggplot(sig_long, aes(x = cluster, y = n_sig, fill = tool)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, alpha = 0.85) +
  scale_fill_manual(values = c("Seurat" = "#2980B9", "scanpy-diff" = "#E74C3C"),
                    name = "Tool") +
  facet_wrap(~method, ncol = 3) +
  labs(
    x = "Cluster", y = "# Significant Genes (padj < 0.05)",
    title = "Significant DE Genes: Seurat vs scanpy-diff"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "#2C3E50"),
    strip.text = element_text(color = "white", face = "bold"),
    legend.position = "top"
  )
ggsave(file.path(OUT_DIR, "04_sig_gene_counts.png"), p4,
       width = 13, height = 5, dpi = 150)
cat("  Saved: 04_sig_gene_counts.png\n")

# ── Figure 5: Top-N gene overlap across methods ───────────────────────
top_ns <- c(10, 20, 50, 100, 200)
overlap_rows <- list()
for (pr in pairs) {
  for (cl in clusters) {
    s_genes <- pr$seurat %>% filter(cluster == cl) %>%
      arrange(desc(avg_log2FC)) %>% pull(gene)
    d_genes <- pr$sd     %>% filter(cluster == cl) %>%
      arrange(desc(log2fc)) %>% pull(gene)
    for (n in top_ns) {
      ov <- length(intersect(head(s_genes, n), head(d_genes, n)))
      overlap_rows[[length(overlap_rows) + 1]] <- data.frame(
        method = pr$label, cluster = cl,
        top_n = n, overlap_pct = ov / n * 100
      )
    }
  }
}
ov_df <- bind_rows(overlap_rows)

p5 <- ov_df %>%
  group_by(method, top_n) %>%
  summarise(mean_overlap = mean(overlap_pct), .groups = "drop") %>%
  ggplot(aes(x = factor(top_n), y = mean_overlap, color = method, group = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.0f%%", mean_overlap)),
            vjust = -0.8, size = 3, show.legend = FALSE) +
  scale_color_manual(
    values = c("Wilcoxon" = "#27AE60", "t-test" = "#2980B9", "Logreg" = "#8E44AD"),
    name = "Method"
  ) +
  ylim(0, 105) +
  labs(
    x = "Top N genes (ranked by log2FC)",
    y = "Mean overlap across clusters (%)",
    title = "Top-N Gene Ranking Overlap: Seurat vs scanpy-diff",
    subtitle = "Average across all clusters"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "grey40"),
    legend.position = "right"
  )
ggsave(file.path(OUT_DIR, "05_topN_overlap.png"), p5,
       width = 8, height = 5, dpi = 150)
cat("  Saved: 05_topN_overlap.png\n")

# ── Summary table ─────────────────────────────────────────────────────
cat("\n[3/3] Summary\n")
cat("=============================================================\n")
summary_tbl <- corr_df %>%
  group_by(method) %>%
  summarise(
    mean_r_lfc  = mean(pearson_lfc,  na.rm = TRUE),
    mean_r_padj = mean(pearson_padj, na.rm = TRUE),
    mean_rho    = mean(spearman_rho, na.rm = TRUE),
    .groups = "drop"
  )

cat(sprintf("  %-10s  %-16s  %-16s  %-16s\n",
            "Method", "Mean r (log2FC)", "Mean r (padj)", "Mean rho (rank)"))
cat(strrep("-", 64), "\n")
for (i in seq_len(nrow(summary_tbl))) {
  cat(sprintf("  %-10s  %-16.4f  %-16.4f  %-16.4f\n",
              summary_tbl$method[i],
              summary_tbl$mean_r_lfc[i],
              summary_tbl$mean_r_padj[i],
              summary_tbl$mean_rho[i]))
}

write.csv(summary_tbl, file.path(OUT_DIR, "summary_table.csv"), row.names = FALSE)
cat("\n  Output files:\n")
for (f in sort(list.files(OUT_DIR))) {
  sz <- file.size(file.path(OUT_DIR, f))
  cat(sprintf("    %-48s (%.1f KB)\n", f, sz / 1024))
}
cat("\n[DONE]\n")
