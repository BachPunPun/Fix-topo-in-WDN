# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely import equals
from pathlib import Path
from collections import defaultdict

from read_file import read_shp
from save_file import save_shp
from topo_helpers import recalc_length
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"


# ── Hàm hỗ trợ: tìm cột ID ───────────────────────────────────
def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID."""
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


# ── Hàm hỗ trợ: đếm thuộc tính non-null/non-empty ───────────
def _count_non_empty_attrs(gdf: gpd.GeoDataFrame, idx: int) -> int:
    """Đếm số cột có giá trị non-null và non-empty (trừ geometry)."""
    count = 0
    for col in gdf.columns:
        if col == "geometry":
            continue
        val = gdf.at[idx, col]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        count += 1
    return count


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_duplicate_lines(gdf: gpd.GeoDataFrame) -> list[list[int]]:
    """
    Phát hiện các nhóm đường thẳng trùng geometry.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Lọc chi tiết — nhóm theo endpoint key, xác nhận bằng shapely.equals.

    Returns:
        List các nhóm (mỗi nhóm là list index) có geometry trùng nhau.
    """
    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    mask = gdf.geometry.notna() & (gdf.geometry.type == "LineString")
    valid_indices = gdf.index[mask].values

    # Bước 2: Lọc chi tiết — nhóm theo sorted endpoint key
    endpoint_groups = defaultdict(list)
    for idx in valid_indices:
        geom = gdf.at[idx, "geometry"]
        coords = geom.coords
        if len(coords) < 2:
            continue
        vs = (round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION))
        ve = (round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION))
        # Sort để gộp A→B và B→A vào cùng nhóm
        key = tuple(sorted([vs, ve]))
        endpoint_groups[key].append(idx)

    # Lọc chỉ giữ nhóm >= 2 đường
    candidate_groups = [grp for grp in endpoint_groups.values() if len(grp) >= 2]

    # Xác nhận bằng shapely.equals — tách nhóm con nếu cần
    duplicate_groups = []
    for grp in candidate_groups:
        # So sánh pairwise trong nhóm, gom thành sub-groups
        visited = set()
        for i in range(len(grp)):
            if grp[i] in visited:
                continue
            sub_group = [grp[i]]
            geom_i = gdf.at[grp[i], "geometry"]
            # Chuẩn hóa geometry: so sánh cả chiều thuận và ngược
            geom_i_rev = LineString(list(geom_i.coords)[::-1])
            for j in range(i + 1, len(grp)):
                if grp[j] in visited:
                    continue
                geom_j = gdf.at[grp[j], "geometry"]
                # So sánh chiều thuận hoặc ngược
                if equals(geom_i, geom_j) or equals(geom_i_rev, geom_j):
                    sub_group.append(grp[j])
                    visited.add(grp[j])
            if len(sub_group) >= 2:
                duplicate_groups.append(sub_group)
                visited.add(grp[i])

    return duplicate_groups


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_duplicate_lines(gdf: gpd.GeoDataFrame, duplicate_groups: list[list[int]]) -> list[dict]:
    """
    Sửa lỗi Duplicate Lines:
    - Mỗi nhóm: giữ đường có nhiều thuộc tính non-null/non-empty nhất
    - Nếu bằng nhau: chọn ID nhỏ nhất
    - Xóa các đường còn lại

    Returns:
        Danh sách kết quả, mỗi phần tử là 1 dict cho 1 nhóm duplicate.
    """
    id_col = _find_id_col(gdf)
    results = []
    drop_indices = []

    for grp in duplicate_groups:
        # Tính số thuộc tính non-null cho mỗi đường trong nhóm
        scored = []
        for idx in grp:
            attr_count = _count_non_empty_attrs(gdf, idx)
            row_id = str(gdf.at[idx, id_col]) if id_col else str(idx)
            scored.append((idx, row_id, attr_count))

        # Sắp xếp: nhiều thuộc tính nhất trước, nếu bằng nhau thì ID nhỏ nhất
        scored.sort(key=lambda x: (-x[2], x[1]))

        keep_idx, keep_id, _ = scored[0]
        removed_ids = [s[1] for s in scored[1:]]
        removed_indices = [s[0] for s in scored[1:]]

        # Lấy chiều dài
        geom = gdf.at[keep_idx, "geometry"]
        length = round(geom.length, COORD_PRECISION)

        # Đánh dấu xóa
        drop_indices.extend(removed_indices)

        results.append({
            "ID-Keep": keep_id,
            "ID-Removed": ",".join(removed_ids),
            "Len": length,
            "Status": "FIXED",
        })

    # Xóa các đường trùng
    gdf.drop(index=drop_indices, inplace=True)
    gdf.reset_index(drop=True, inplace=True)

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    # Phát hiện lỗi
    duplicate_groups = detect_duplicate_lines(gdf)

    # Sửa lỗi
    results = correct_duplicate_lines(gdf, duplicate_groups)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(results)} trường hợp Duplicate Lines ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(results)} trường hợp Duplicate Lines ✅")
    print(f"{'STT':<5} | {'ID-Keep':<38} | {'ID-Removed':<38} | {'Len':<12} | Status")
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['ID-Keep']:<38} | {r['ID-Removed']:<38} | {r['Len']:<12} | {r['Status']}")


if __name__ == "__main__":
    main()
