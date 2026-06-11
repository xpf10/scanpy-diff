"""
scanpy_diff: Differential Gene Expression Analysis Plugin for Scanpy

A comprehensive differential expression analysis package that mirrors
Seurat's FindMarkers/FindAllMarkers functionality, with multiple
statistical testing methods and visualization tools.

Usage
-----
>>> import scanpy as sc
>>> import scanpy_diff as sd

>>> # Find markers for one cluster vs all others
>>> markers = sd.find_markers(adata, groupby='leiden', group='0')

>>> # Find markers for all clusters
>>> all_markers = sd.find_all_markers(adata, groupby='leiden')

>>> # Visualize
>>> sd.pl.volcano(markers)
>>> sd.pl.marker_heatmap(adata, all_markers)
"""

from . import pl
from ._diff import find_all_markers, find_markers
from ._utils import (
    filter_markers,
    markers_to_dict,
    rank_markers,
    store_in_adata,
    top_markers,
)

__version__ = "0.1.0"

__all__ = [
    "find_markers",
    "find_all_markers",
    "filter_markers",
    "rank_markers",
    "top_markers",
    "markers_to_dict",
    "store_in_adata",
    "pl",
]
