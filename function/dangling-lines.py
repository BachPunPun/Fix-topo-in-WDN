# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import STRtree
from shapely.ops import nearest_points
from collections import defaultdict
from pathlib import Path

from read_file import read_shp
from save_file import save_shp
from graph_builder import build_graph
from topo_helpers import build_strtree, recalc_length
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"
ITERATIONS_MAX = 1
DIST_THRESHOLD = 0.15


# ── Hàm hỗ trợ: tìm cột ID ───────────────────────────────────
def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID."""
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


# ── Hàm hỗ trợ: làm tròn tọa độ ──────────────────────────────
def _round_pt(coord):
    """Làm tròn tọa độ."""
    return (round(coord[0], COORD_PRECISION), round(coord[1], COORD_PRECISION))


# ── Hàm hỗ trợ: lấy endpoints ─────────────────────────────────
def _get_endpoints(line):
    """Trả về (vs, ve) đã làm tròn."""
    coords = line.coords
    return _round_pt(coords[0]), _round_pt(coords[-1])


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_dangling(gdf: gpd.GeoDataFrame, dist_threshold: float = DIST_THRESHOLD) -> list[dict]:
    """
    Phát hiện lỗi Dangling Lines.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng G.degree lọc line source có endpoint bậc 1 (des — dead end source).
    Bước 3: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 4: Kiểm tra từng cặp (source, target):
        - Dùng disjoint kiểm tra: chỉ lấy cặp KHÔNG giao nhau (disjoint=True)
        - Kiểm tra line target có endpoint bậc 1 (det — dead end target)
        - 0 < dist(des, line target) <= dist_threshold VÀ 0 < dist(det, line source) <= dist_threshold
        → LỖI DANGLING LINES

    Returns:
        List[dict] với mỗi phần tử chứa:
          - src_idx: index line source
          - tgt_idx: index line target
          - des: dead end source (Point)
          - des_coord: tọa độ des đã làm tròn
          - des_type: "vs" hoặc "ve"
          - det: dead end target (Point)
          - det_coord: tọa độ det đã làm tròn
          - det_type: "vs" hoặc "ve"
          - dist_des_to_tgt: khoảng cách từ des đến line target
          - dist_det_to_src: khoảng cách từ det đến line source
    """
    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    tree, valid_idx = build_strtree(gdf)
    geoms = gdf.geometry[valid_idx].values

    # Bước 2: Xây dựng đồ thị để xác định bậc
    G = build_graph(gdf)

    # Tìm tất cả line có endpoint bậc 1
    deg1_lines = {}  # idx -> list of (type, Point, coord)
    for vi, idx in enumerate(valid_idx):
        geom = gdf.at[idx, "geometry"]
        vs_coord = _round_pt(geom.coords[0])
        ve_coord = _round_pt(geom.coords[-1])

        dead_ends = []
        if vs_coord in G and G.degree(vs_coord) == 1:
            dead_ends.append(("vs", Point(vs_coord), vs_coord))
        if ve_coord in G and G.degree(ve_coord) == 1:
            dead_ends.append(("ve", Point(ve_coord), ve_coord))

        if dead_ends:
            deg1_lines[idx] = dead_ends

    if not deg1_lines:
        return []

    results = []
    seen_pairs = set()

    # Bước 3: Dùng STRtree — tìm các cặp đường thẳng gần nhau
    for src_idx, des_list in deg1_lines.items():
        geom_src = gdf.at[src_idx, "geometry"]

        for des_type, des_pt, des_coord in des_list:
            # Tìm các target gần des bằng STRtree query
            buffered_des = des_pt.buffer(dist_threshold)
            candidate_indices = tree.query(buffered_des)

            for ci in candidate_indices:
                tgt_idx = valid_idx[ci]

                if tgt_idx == src_idx:
                    continue

                # Dùng frozenset để tránh trùng cặp (A,B) và (B,A)
                pair_key = frozenset({src_idx, tgt_idx})
                if pair_key in seen_pairs:
                    continue

                geom_tgt = gdf.at[tgt_idx, "geometry"]

                # Bước 4: Kiểm tra disjoint — chỉ lấy cặp KHÔNG giao nhau
                if not geom_src.disjoint(geom_tgt):
                    continue

                # Kiểm tra line target có endpoint bậc 1
                if tgt_idx not in deg1_lines:
                    continue

                # Tính khoảng cách từ des đến line target
                dist_des_to_tgt = des_pt.distance(geom_tgt)

                # 0 < dist(des, line target) <= dist_threshold
                if dist_des_to_tgt <= 0 or dist_des_to_tgt > dist_threshold:
                    continue

                # Kiểm tra từng dead end target (det)
                det_list = deg1_lines[tgt_idx]
                found = False
                for det_type, det_pt, det_coord in det_list:
                    # Tính khoảng cách từ det đến line source
                    dist_det_to_src = det_pt.distance(geom_src)

                    # 0 < dist(det, line source) <= dist_threshold
                    if dist_det_to_src <= 0 or dist_det_to_src > dist_threshold:
                        continue

                    # → LỖI DANGLING LINES
                    seen_pairs.add(pair_key)
                    results.append({
                        "src_idx": src_idx,
                        "tgt_idx": tgt_idx,
                        "des": des_pt,
                        "des_coord": des_coord,
                        "des_type": des_type,
                        "det": det_pt,
                        "det_coord": det_coord,
                        "det_type": det_type,
                        "dist_des_to_tgt": dist_des_to_tgt,
                        "dist_det_to_src": dist_det_to_src,
                    })
                    found = True
                    break  # Chỉ cần 1 det thỏa mãn là đủ

                if found:
                    break  # Đã tìm thấy lỗi cho cặp này

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_dangling(gdf: gpd.GeoDataFrame, detections: list[dict]) -> tuple[list[dict], gpd.GeoDataFrame]:
    """
    Sửa lỗi Dangling Lines:

    Nguyên tắc:
      - Cố định line target
      - Dời dead end source (des) đến dead end target (det)
      - Chỉ tính lại chiều dài line source

    Bước 1: Dời des của line source đến det của line target
    Bước 2: Tính lại chiều dài line source

    Returns:
        (danh sách kết quả báo cáo, gdf đã cập nhật)
    """
    id_col = _find_id_col(gdf)
    report_results = []
    modified = set()

    for d in detections:
        src_idx = d["src_idx"]
        tgt_idx = d["tgt_idx"]
        des_type = d["des_type"]
        det_coord = d["det_coord"]

        src_id = str(gdf.at[src_idx, id_col]) if id_col else str(src_idx)
        tgt_id = str(gdf.at[tgt_idx, id_col]) if id_col else str(tgt_idx)

        geom_src = gdf.at[src_idx, "geometry"]
        old_len_src = round(geom_src.length, 4)

        status = "FIXED"

        # Bước 1: Dời des của line source đến det của line target
        src_coords = list(geom_src.coords)
        # Preserve Z dimension if source geometry has Z
        if geom_src.has_z:
            if len(det_coord) >= 3:
                snap_coord = (round(det_coord[0], COORD_PRECISION), round(det_coord[1], COORD_PRECISION), det_coord[2])
            else:
                z_val = src_coords[0][2] if des_type == "vs" else src_coords[-1][2]
                snap_coord = (round(det_coord[0], COORD_PRECISION), round(det_coord[1], COORD_PRECISION), z_val)
        else:
            snap_coord = (round(det_coord[0], COORD_PRECISION), round(det_coord[1], COORD_PRECISION))
        if des_type == "vs":
            new_src_coords = [snap_coord] + src_coords[1:]
        else:
            new_src_coords = src_coords[:-1] + [snap_coord]

        if len(new_src_coords) >= 2:
            new_src_geom = LineString(new_src_coords)
            if not new_src_geom.is_empty and new_src_geom.length > 0:
                gdf.at[src_idx, "geometry"] = new_src_geom
                recalc_length(gdf, src_idx)
                modified.add(src_idx)
            else:
                status = "SKIPPED"
        else:
            status = "SKIPPED"

        # Bước 2: Tính lại chiều dài line source
        new_len_src = round(gdf.at[src_idx, "geometry"].length, 4)

        report_results.append({
            "ID-Source": src_id,
            "OldLen-Source": old_len_src,
            "NewLen-Source": new_len_src,
            "ID-Target": tgt_id,
            "Status": status,
        })

    # Đảm bảo kiểu dữ liệu
    if 'ID' in gdf.columns:
        gdf['ID'] = gdf['ID'].astype(str)
    if 'Len' in gdf.columns:
        gdf['Len'] = gdf['Len'].astype(float)

    return report_results, gdf


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    all_results = []

    # Lặp detect → correct tối đa ITERATIONS_MAX lần
    for iteration in range(1, ITERATIONS_MAX + 1):
        # Phát hiện lỗi
        detections = detect_dangling(gdf)

        if not detections:
            break

        # Sửa lỗi
        results, gdf = correct_dangling(gdf, detections)
        all_results.extend(results)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in all_results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(all_results)} trường hợp Dangling Lines ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(all_results)} trường hợp Dangling Lines ✅")
    print(f"{'STT':<5} | {'ID-Source':<10} | {'OldLen-Source':<14} | {'NewLen-Source':<14} | {'ID-Target':<10} | Status")
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} | {r['ID-Source']:<10} | {r['OldLen-Source']:<14} | {r['NewLen-Source']:<14} | {r['ID-Target']:<10} | {r['Status']}")


if __name__ == "__main__":
    main()
