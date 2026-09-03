# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from pathlib import Path

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


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_multi_linestring(gdf: gpd.GeoDataFrame) -> list[int]:
    """Lọc các đường thẳng có kiểu geometry là MultiLineString."""
    mask = gdf.geometry.notna() & (gdf.geometry.type == "MultiLineString")
    return gdf.index[mask].tolist()


# ── Bước 2-a: Gộp bằng linemerge ──────────────────────────────
def _try_linemerge(geom: MultiLineString) -> LineString | MultiLineString:
    """Gộp tự động bằng shapely.ops.linemerge."""
    merged = linemerge(geom)
    return merged


# ── Bước 2-b: Farthest Endpoint ───────────────────────────────
def _farthest_endpoint_merge(geom: MultiLineString) -> LineString | None:
    """
    Nếu linemerge vẫn trả về MultiLineString:
    - Tìm 2 điểm đầu vs hoặc cuối ve xa nhất trong tất cả các parts
    - Sắp xếp các parts theo thứ tự khoảng cách từ start_pt
    - Nối các parts thành một LineString duy nhất
    """
    parts = list(geom.geoms)
    if len(parts) < 2:
        return None

    # Thu thập tất cả endpoint (vs, ve) của mỗi part
    endpoints = []
    for p in parts:
        coords = list(p.coords)
        vs = coords[0]
        ve = coords[-1]
        endpoints.append(vs)
        endpoints.append(ve)

    # Tìm 2 endpoint xa nhất
    max_dist = -1.0
    start_pt = None
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dx = endpoints[i][0] - endpoints[j][0]
            dy = endpoints[i][1] - endpoints[j][1]
            dist = dx * dx + dy * dy
            if dist > max_dist:
                max_dist = dist
                start_pt = endpoints[i]

    # Sắp xếp parts theo khoảng cách từ start_pt
    def part_dist(part):
        coords = list(part.coords)
        vs = coords[0]
        ve = coords[-1]
        d_start = (vs[0] - start_pt[0]) ** 2 + (vs[1] - start_pt[1]) ** 2
        d_end   = (ve[0] - start_pt[0]) ** 2 + (ve[1] - start_pt[1]) ** 2
        return min(d_start, d_end)

    sorted_parts = sorted(parts, key=part_dist)

    # Đảm bảo part đầu tiên có hướng đúng (start_pt ở đầu)
    first_coords = list(sorted_parts[0].coords)
    d_vs = (first_coords[0][0] - start_pt[0]) ** 2 + (first_coords[0][1] - start_pt[1]) ** 2
    d_ve = (first_coords[-1][0] - start_pt[0]) ** 2 + (first_coords[-1][1] - start_pt[1]) ** 2
    if d_ve < d_vs:
        first_coords = first_coords[::-1]  # đảo hướng để start_pt ở đầu

    # Nối các parts: đảm bảo hướng nối liên tục
    all_coords = first_coords
    for k in range(1, len(sorted_parts)):
        next_coords = list(sorted_parts[k].coords)
        # Kiểm tra nối đuôi-đầu hay đuôi-cuối
        tail = all_coords[-1]
        d_to_start = (tail[0] - next_coords[0][0]) ** 2 + (tail[1] - next_coords[0][1]) ** 2
        d_to_end   = (tail[0] - next_coords[-1][0]) ** 2 + (tail[1] - next_coords[-1][1]) ** 2

        if d_to_end < d_to_start:
            next_coords = next_coords[::-1]  # đảo hướng

        # Tránh trùng vertex tại điểm nối
        if all_coords[-1] == next_coords[0]:
            next_coords = next_coords[1:]

        all_coords.extend(next_coords)

    if len(all_coords) < 2:
        return None

    return LineString(all_coords)


# ── Sửa lỗi chính ─────────────────────────────────────────────
def correct_multi_linestring(gdf: gpd.GeoDataFrame, error_indices: list[int]) -> list[dict]:
    """
    Sửa lỗi MultiLineString theo thuật toán:
    1. Gộp bằng linemerge
    2. Nếu vẫn MultiLineString → dùng Farthest Endpoint
    3. Kiểm tra lại geometry
    4. Tính lại chiều dài
    """
    id_col = _find_id_col(gdf)
    results = []

    for idx in error_indices:
        geom = gdf.at[idx, "geometry"]
        row_id = str(gdf.at[idx, id_col]) if id_col else str(idx)
        num_parts = len(list(geom.geoms))
        old_len = round(geom.length, COORD_PRECISION)

        # Bước 1: Gộp bằng linemerge
        merged = _try_linemerge(geom)

        # Bước 2: Nếu vẫn MultiLineString → Farthest Endpoint
        if isinstance(merged, MultiLineString):
            merged = _farthest_endpoint_merge(merged)
            if merged is None:
                results.append({
                    "ID": row_id, "Parts": num_parts,
                    "OldLen": old_len, "NewLen": old_len,
                    "Status": "SKIPPED",
                })
                continue

        # Bước 3: Kiểm tra lại geometry
        if not isinstance(merged, LineString) or merged.is_empty or len(list(merged.coords)) < 2:
            results.append({
                "ID": row_id, "Parts": num_parts,
                "OldLen": old_len, "NewLen": old_len,
                "Status": "SKIPPED",
            })
            continue

        # Áp dụng sửa lỗi
        gdf.at[idx, "geometry"] = merged

        # Bước 4: Tính lại chiều dài
        new_len = recalc_length(gdf, idx)

        results.append({
            "ID": row_id, "Parts": num_parts,
            "OldLen": old_len, "NewLen": round(new_len, COORD_PRECISION),
            "Status": "FIXED",
        })

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    # Phát hiện lỗi
    error_indices = detect_multi_linestring(gdf)

    # Sửa lỗi
    results = correct_multi_linestring(gdf, error_indices)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(results)} trường hợp Multi LineString ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(results)} trường hợp Multi LineString ✅")
    print(f"{'STT':<5} | {'ID':<38} | {'Parts':<5} | {'OldLen':<12} | {'NewLen':<12} | Status")
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['ID']:<38} | {r['Parts']:<5} | {r['OldLen']:<12} | {r['NewLen']:<12} | {r['Status']}")


if __name__ == "__main__":
    main()
