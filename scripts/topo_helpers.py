"""
Topology helpers — Các hàm dùng chung cho nhiều skill topology.

Cách dùng:
    from topo_helpers import build_strtree, find_single_intersection, ...
"""

import numpy as np
import geopandas as gpd
from shapely import STRtree
from shapely.geometry import Point, LineString, GeometryCollection
from shapely.ops import substring

COORD_PRECISION = 6


# ──────────────────────────────────────────────
# 1. Spatial index
# ──────────────────────────────────────────────

def build_strtree(gdf: gpd.GeoDataFrame):
    """
    Xây dựng STRtree từ GeoDataFrame (chỉ lấy LineString hợp lệ).

    Returns:
        (STRtree, np.ndarray of valid indices)
    """
    mask = gdf.geometry.notna() & (gdf.geometry.type == "LineString")
    valid_idx = gdf.index[mask].values
    geoms = gdf.geometry[mask].values
    return STRtree(geoms), valid_idx


# ──────────────────────────────────────────────
# 2. Recalc length
# ──────────────────────────────────────────────

def recalc_length(gdf: gpd.GeoDataFrame, idx) -> float:
    """
    Tính lại chiều dài geometry tại index và cập nhật cột 'Len'.

    Returns:
        Chiều dài mới (float).
    """
    geom = gdf.at[idx, "geometry"]
    length = round(geom.length, COORD_PRECISION) if geom and not geom.is_empty else 0.0
    gdf.at[idx, "Len"] = length
    return length
