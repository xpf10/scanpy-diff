# scanpy-diff

**scanpy-diff** is a scanpy plugin providing Seurat-style differential gene expression analysis (`FindMarkers` / `FindAllMarkers`). Supports Wilcoxon, t-test, logistic regression, ROC, and DESeq2 methods with multiple testing correction and visualization.

> 中文用户请参考下方完整文档。English API docs are in the Python docstrings.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 功能特性

| 功能 | Seurat 对应 | scanpy-diff |
|------|-------------|-------------|
| 单组 vs 参考 | `FindMarkers()` | `sd.find_markers()` |
| 所有组 vs 参考 | `FindAllMarkers()` | `sd.find_all_markers()` |
| Wilcoxon 秩和检验 | `test = "wilcox"` | `method = "wilcoxon"` |
| Welch t 检验 | `test = "t"` | `method = "t-test"` |
| 逻辑回归检验 | `test = "LR"` | `method = "logreg"` |
| ROC AUC 分析 | `test = "roc"` | `method = "roc"` |
| DESeq2 检验 | `test = "DESeq2"` | `method = "deseq2"` |
| 多重比较校正 | BH, Bonferroni | BH, Bonferroni, BY, Holm |
| 最小表达比例过滤 | `min.pct` | `min_pct` |
| Fold Change 过滤 | `logfc.threshold` | `logfc_threshold` |
| 火山图 | — | `sd.pl.volcano()` |
| 热图 | — | `sd.pl.marker_heatmap()` |
| 点图 | — | `sd.pl.dotplot()` |

---

## 📦 安装

```bash
pip install scanpy-diff
```

或从源码安装：

```bash
git clone https://github.com/xpf10/scanpy-diff.git
cd scanpy-diff
pip install -e ".[dev]"
```

可选依赖（DESeq2 方法）：

```bash
pip install pydeseq2
```

---

## 🚀 快速开始

```python
import scanpy as sc
import scanpy_diff as sd

# 加载数据（需要已完成聚类）
adata = sc.datasets.pbmc3k_processed()

# -----------------------------------------------
# 1. 单组差异分析（类似 Seurat FindMarkers）
# -----------------------------------------------

# Cluster 0 vs 所有其他细胞（默认 Wilcoxon 检验）
markers = sd.find_markers(adata, groupby='louvain', group='0')
print(markers.head(10))

# Cluster 0 vs Cluster 1 的对比
markers_01 = sd.find_markers(
    adata,
    groupby='louvain',
    group='0',
    reference='1',
    method='wilcoxon',
    min_pct=0.25,
    logfc_threshold=0.5,
    padj_cutoff=0.05,
)
print(markers_01.head(10))

# -----------------------------------------------
# 2. 所有组差异分析（类似 Seurat FindAllMarkers）
# -----------------------------------------------

all_markers = sd.find_all_markers(
    adata,
    groupby='louvain',
    method='wilcoxon',
    n_genes_per_group=50,
    min_pct=0.1,
    logfc_threshold=0.25,
    padj_cutoff=0.05,
    only_positive=True,
)
print(all_markers.head(20))

# -----------------------------------------------
# 3. 结果处理
# -----------------------------------------------

# 进一步过滤
filtered = sd.filter_markers(
    markers,
    min_log2fc=1.0,
    max_padj=0.01,
    min_pct=0.25,
)

# 获取每个 cluster 的 top 10 标记基因
top10 = sd.top_markers(all_markers, n=10)

# 转换为字典格式（可直接传入 sc.pl.dotplot）
marker_dict = sd.markers_to_dict(all_markers, n=20)
sc.pl.dotplot(adata, marker_dict, groupby='louvain')

# -----------------------------------------------
# 4. 可视化
# -----------------------------------------------

# 火山图
sd.pl.volcano(markers, log2fc_cutoff=1.0, padj_cutoff=0.05, n_label=10)

# 标记基因热图
sd.pl.marker_heatmap(adata, all_markers, groupby='louvain', n_genes=5)

# 点图
sd.pl.dotplot(adata, all_markers, groupby='louvain', n_genes=5)

# 每个 cluster 的标记基因数量柱状图
sd.pl.summary_barplot(all_markers)

# log2FC 热图（基因 × cluster）
sd.pl.log2fc_heatmap(all_markers, n_genes=5)
```

---

## 📊 返回值结构

`find_markers()` 返回一个 `pd.DataFrame`，列含义如下（与 Seurat 对照）：

| 列名 | Seurat 对应 | 说明 |
|------|-------------|------|
| `gene` | 行名 | 基因名 |
| `scores` | — | 检验统计量（AUC/t值/LR统计量）|
| `log2fc` | `avg_log2FC` | 平均 log2 fold change |
| `pct_1` | `pct.1` | group 中表达该基因的细胞比例 |
| `pct_2` | `pct.2` | reference 中表达该基因的细胞比例 |
| `pval` | `p_val` | 原始 p 值 |
| `padj` | `p_val_adj` | 校正后 p 值 |

`find_all_markers()` 额外返回：

| 列名 | Seurat 对应 | 说明 |
|------|-------------|------|
| `cluster` | `cluster` | 来源 cluster |

---

## 🔬 方法说明

### Wilcoxon（默认，推荐）

```python
markers = sd.find_markers(adata, groupby='leiden', group='0', method='wilcoxon')
```

- 非参数检验，不假设正态分布
- 对噪声鲁棒，是 Seurat 默认方法
- 适合大多数 scRNA-seq 数据

### t-test

```python
markers = sd.find_markers(adata, groupby='leiden', group='0', method='t-test')
```

- Welch t 检验（不假设等方差）
- 速度快，适合大数据集
- 假设数据近似正态分布

### 逻辑回归（logreg）

```python
markers = sd.find_markers(adata, groupby='leiden', group='0', method='logreg')
```

- 基于逻辑回归的似然比检验
- 对应 Seurat 的 `LR` 方法
- 可以添加协变量（后续版本）

### ROC 分析

```python
markers = sd.find_markers(adata, groupby='leiden', group='0', method='roc')
```

- 计算每个基因的 AUC（分类能力）
- 返回 AUC 值作为 scores
- AUC > 0.5 表示上调

### DESeq2（伪批量）

```python
markers = sd.find_markers(
    adata, groupby='leiden', group='0',
    method='deseq2', layer='counts'  # 需要原始计数
)
```

- 需要安装 `pydeseq2`：`pip install pydeseq2`
- 使用原始计数数据（整数）
- 适合处理批次效应显著的数据

---

## ⚙️ 参数详解

### `find_markers()` 关键参数

```python
sd.find_markers(
    adata,
    groupby='leiden',        # obs 列名
    group='0',               # 要分析的 cluster
    reference='rest',        # 参考组（默认所有其他细胞）
    method='wilcoxon',       # 检验方法
    layer=None,              # 使用的层（默认 adata.X）
    n_genes=None,            # 返回 top N 基因
    min_pct=0.1,             # group 中最低表达比例（等同 Seurat min.pct）
    min_pct_reference=0.0,   # reference 中最低表达比例
    logfc_threshold=0.25,    # 最低 |log2FC|（等同 Seurat logfc.threshold）
    pval_cutoff=1.0,         # 原始 p 值过滤
    padj_cutoff=1.0,         # 校正 p 值过滤
    only_positive=False,     # 只返回上调基因
    correction_method='fdr_bh',  # 多重比较校正方法
    use_raw=False,           # 使用 adata.raw
    verbose=True,            # 打印进度
)
```

---

## 🧪 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 运行特定测试类
pytest tests/test_diff.py::TestFindMarkers -v

# 查看覆盖率
pytest tests/ --cov=scanpy_diff --cov-report=html
```

---

## 📁 项目结构

```
scanpy_diff/
├── scanpy_diff/
│   ├── __init__.py       # 公共 API
│   ├── _diff.py          # find_markers(), find_all_markers()
│   ├── _stats.py         # 统计检验实现
│   ├── _utils.py         # 工具函数
│   └── pl.py             # 可视化函数
├── tests/
│   ├── __init__.py
│   └── test_diff.py      # 测试套件
├── pyproject.toml
└── README.md
```

---

## 📝 引用

如果您在研究中使用了 scanpy-diff，请引用：

- Seurat: Hao et al. (2021) *Cell* 
- scanpy: Wolf et al. (2018) *Genome Biology*

---

## 🔍 与 Seurat 的对比验证

项目包含完整的交叉验证脚本（`comparison/` 目录）：

| 脚本 | 说明 |
|------|------|
| `seurat_de.R` | Seurat PBMC3k 完整分析流程 |
| `compare_de.py` | scanpy vs scanpy-diff 全方法对比 |
| `cross_compare.R` | Seurat vs scanpy-diff 跨平台比较 |
| `fair_compare.R` | 使用相同 Leiden 标签的公平对比 |

运行顺序：`seurat_de.R` → `compare_de.py` → `cross_compare.R` → `fair_compare.R`

---

## 📜 许可证

MIT License
