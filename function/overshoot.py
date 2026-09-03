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
from shapely.ops import substring
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
ITERATIONS_MAX = 5
ANGEL_MIN = 30
ANGEL_MAX = 150
DIST_THRESHOLD = 0.1


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


# ── Hàm hỗ trợ: trích xuất 1 giao điểm duy nhất ──────────────
def _extract_single_point(geom):
    """
    Trích xuất giao điểm duy nhất từ kết quả intersection.
    Trả về Point nếu chỉ có đúng 1 giao điểm, None nếu không.
    """
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == 'Point':
        return geom

    if geom.geom_type in ('GeometryCollection', 'MultiPoint'):
        pts = [g for g in geom.geoms if g.geom_type == 'Point']
        if len(pts) == 1:
            return pts[0]

    return None


# ── Hàm hỗ trợ: tìm vector hướng segment chứa pt ─────────────
def _get_seg_vector(geom, pt):
    """Tìm vector hướng segment của geom chứa điểm pt."""
    coords = list(geom.coords)
    best_seg = None
    best_dist = float('inf')
    for i in range(len(coords) - 1):
        p1, p2 = Point(coords[i]), Point(coords[i + 1])
        seg = LineString([p1, p2])
        d = seg.distance(pt)
        if d < best_dist:
            best_dist = d
            best_seg = (coords[i], coords[i + 1])
    if best_seg is None:
        return None
    p1, p2 = best_seg
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return dx, dy


# ── Hàm hỗ trợ: tính góc giữa 2 vector ───────────────────────
def _calc_angle(v1, v2):
    """Tính góc giữa 2 vector (0°-90°)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    ang = math.degrees(math.acos(cos_val))
    ang = min(ang, 180.0 - ang)
    return ang


# ── Hàm hỗ trợ: kiểm tra pt có phải endpoint ─────────────────
def _is_endpoint(pt, line, tol=1e-5):
    """Kiểm tra pt có phải đầu/cuối của line."""
    vs, ve = _get_endpoints(line)
    rpt = _round_pt((pt.x, pt.y))
    return rpt == vs or rpt == ve


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_overshoot(gdf: gpd.GeoDataFrame, dist_threshold: float = DIST_THRESHOLD) -> list[dict]:
    """
    Phát hiện lỗi Overshoot.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng G.degree lọc line source có endpoint bậc 1 (vso).
    Bước 3: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 4: Kiểm tra từng cặp (source, target):
        - Tính intersection → chỉ lấy 1 giao điểm duy nhất
        - pi KHÔNG phải endpoint của source → tiếp tục
        - 0 < dist(pi, vso) <= dist_threshold → tiếp tục
        - Góc giao từ 30° đến 150° → tiếp tục
        → LỖI OVERSHOOT

    Returns:
        List[dict] với mỗi phần tử chứa:
          - src_idx: index line source
          - tgt_idx: index line target
          - pi: giao điểm (Point)
          - vso: endpoint bậc 1 của source (Point)
          - vso_coord: tọa độ vso đã làm tròn
          - angle: góc tại giao điểm
    """
    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    tree, valid_idx = build_strtree(gdf)
    geoms = gdf.geometry[valid_idx].values

    # Bước 2: Xây dựng đồ thị để xác định bậc
    G = build_graph(gdf)

    # Tìm tất cả line source có endpoint bậc 1
    deg1_sources = {}  # src_idx -> list of vso Points
    for vi, idx in enumerate(valid_idx):
        geom = gdf.at[idx, "geometry"]
        vs_coord = _round_pt(geom.coords[0])
        ve_coord = _round_pt(geom.coords[-1])

        vso_list = []
        if vs_coord in G and G.degree(vs_coord) == 1:
            vso_list.append(("vs", Point(vs_coord), vs_coord))
        if ve_coord in G and G.degree(ve_coord) == 1:
            vso_list.append(("ve", Point(ve_coord), ve_coord))

        if vso_list:
            deg1_sources[idx] = vso_list

    if not deg1_sources:
        return []

    results = []
    seen_pairs = set()

    # Bước 3: Dùng STRtree tìm các cặp gần nhau
    left, right = tree.query(geoms, predicate='intersects')

    for l, r in zip(left, right):
        idx1 = valid_idx[l]
        idx2 = valid_idx[r]

        if idx1 == idx2:
            continue

        # Kiểm tra cả hai hướng: idx1 là source, idx2 là target VÀ ngược lại
        for src_idx, tgt_idx in [(idx1, idx2), (idx2, idx1)]:
            if src_idx not in deg1_sources:
                continue

            pair_key = (src_idx, tgt_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            geom_src = gdf.at[src_idx, "geometry"]
            geom_tgt = gdf.at[tgt_idx, "geometry"]

            # Bước 4: Tính intersection
            inter = geom_src.intersection(geom_tgt)

            # Chỉ lấy 1 giao điểm duy nhất
            pi = _extract_single_point(inter)
            if pi is None:
                continue

            # pi KHÔNG phải endpoint của source → tiếp tục
            if _is_endpoint(pi, geom_src):
                continue

            # Kiểm tra khoảng cách từ pi đến vso
            for vso_type, vso_pt, vso_coord in deg1_sources[src_idx]:
                dist_pi_vso = pi.distance(vso_pt)

                # 0 < dist(pi, vso) <= dist_threshold
                if dist_pi_vso <= 0 or dist_pi_vso > dist_threshold:
                    continue

                # Tính góc giao
                v1 = _get_seg_vector(geom_src, pi)
                v2 = _get_seg_vector(geom_tgt, pi)
                if not v1 or not v2:
                    continue

                ang = _calc_angle(v1, v2)
                if ang is None:
                    continue

                # Kiểm tra góc trong khoảng [ANGEL_MIN, ANGEL_MAX]
                if not (ANGEL_MIN <= ang <= ANGEL_MAX):
                    continue

                # → LỖI OVERSHOOT
                results.append({
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx,
                    "pi": pi,
                    "vso": vso_pt,
                    "vso_coord": vso_coord,
                    "vso_type": vso_type,
                    "angle": ang,
                })

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_overshoot(gdf: gpd.GeoDataFrame, detections: list[dict], dist_threshold: float = DIST_THRESHOLD) -> tuple[list[dict], gpd.GeoDataFrame]:
    """
    Sửa lỗi Overshoot.

    Nguyên tắc:
      - Mỗi line source chỉ xử lý tối đa hai line target (một ở đầu, một ở cuối).
      - MỘT line target có thể bị nhiều line source overshoot vào → phải gom tất
        cả điểm pi của target đó và CẮT ĐÚNG MỘT LẦN tại mọi điểm (giống
        correct_t_junction). Tránh cắt lặp lại đường target gốc cho từng source
        — vốn sinh ra các đoạn chồng nhau, trùng ID và bùng nổ qua các vòng lặp.

    Pha 1 (theo source): cắt bỏ đuôi overshoot của mỗi source tới pi, đồng thời
                         gom điểm pi theo từng target.
    Pha 2 (theo target): cắt mỗi target một lần tại tất cả pi nội bộ; pi sát đầu/
                         cuối target chỉ trim phần thừa (không tách).

    Returns:
        Danh sách kết quả báo cáo (gom theo source).
    """
    id_col = _find_id_col(gdf)
    modified_sources = set()

    # Gom detections theo source
    source_detections = defaultdict(list)
    for det in detections:
        source_detections[det["src_idx"]].append(det)

    cuts_by_target = defaultdict(list)   # tgt_idx -> list[pi Point] (gộp mọi source)
    source_info = {}                     # src_idx -> thông tin để dựng báo cáo

    # ── Pha 1: cắt đuôi source + gom điểm cắt theo target ──
    for src_idx, dets in source_detections.items():
        # Mỗi source chỉ xử lý tối đa 2 target (1 ở đầu, 1 ở cuối) — chọn theo vso_type
        best_by_type = {}
        for det in dets:
            t = det["vso_type"]
            if t not in best_by_type or \
               det["pi"].distance(det["vso"]) < best_by_type[t]["pi"].distance(best_by_type[t]["vso"]):
                best_by_type[t] = det
        selected_dets = list(best_by_type.values())

        src_id = str(gdf.at[src_idx, id_col]) if id_col else str(src_idx)
        old_len_src = round(gdf.at[src_idx, "geometry"].length, 4)
        tgt_idxs, tgt_ids, angles = [], [], []
        all_ok = True

        for det in selected_dets:
            tgt_idx = det["tgt_idx"]
            pi = det["pi"]
            vso_type = det["vso_type"]

            tgt_idxs.append(tgt_idx)
            tgt_ids.append(str(gdf.at[tgt_idx, id_col]) if id_col else str(tgt_idx))
            angles.append(round(det["angle"], 1))

            # Cắt đuôi overshoot của source (mỗi source chỉ một lần)
            if src_idx not in modified_sources:
                geom_src = gdf.at[src_idx, "geometry"]
                d_pi = geom_src.project(pi)
                if vso_type == "vs":
                    new_src_geom = substring(geom_src, d_pi, geom_src.length)
                else:
                    new_src_geom = substring(geom_src, 0, d_pi)

                if new_src_geom is not None and not new_src_geom.is_empty \
                   and new_src_geom.geom_type == 'LineString' and new_src_geom.length > 0:
                    gdf.at[src_idx, "geometry"] = new_src_geom
                    recalc_length(gdf, src_idx)
                    modified_sources.add(src_idx)
                else:
                    all_ok = False

            # Gom điểm cắt theo target (chưa cắt vội)
            cuts_by_target[tgt_idx].append(pi)

        source_info[src_idx] = {
            "ID-Source": src_id, "OldLen-Source": old_len_src,
            "tgt_idxs": tgt_idxs, "tgt_ids": tgt_ids, "angles": angles,
            "all_ok": all_ok,
        }

    # ── Pha 2: cắt mỗi target đúng MỘT lần tại tất cả pi ──
    new_rows = []
    targets_to_remove = set()
    target_result = {}   # tgt_idx -> (old_len_str, new_len_str)

    for tgt_idx, pis in cuts_by_target.items():
        geom_tgt = gdf.at[tgt_idx, "geometry"]
        if geom_tgt is None or geom_tgt.is_empty:
            continue
        tgt_id = str(gdf.at[tgt_idx, id_col]) if id_col else str(tgt_idx)
        L = geom_tgt.length
        old_len_str = str(round(L, 4))

        # Phân loại điểm cắt: sát đầu/cuối → trim; còn lại → cắt nội bộ
        start_trim, end_trim, interior = 0.0, L, []
        for d in sorted(geom_tgt.project(p) for p in pis):
            if d <= dist_threshold:
                start_trim = max(start_trim, d)
            elif d >= L - dist_threshold:
                end_trim = min(end_trim, d)
            else:
                interior.append(d)

        if start_trim >= end_trim:
            target_result[tgt_idx] = (old_len_str, old_len_str)
            continue

        breakpoints = [start_trim]
        for d in interior:
            if start_trim < d < end_trim and abs(d - breakpoints[-1]) > 1e-8:
                breakpoints.append(d)
        breakpoints.append(end_trim)

        parts = []
        for i in range(len(breakpoints) - 1):
            seg = substring(geom_tgt, breakpoints[i], breakpoints[i + 1])
            if seg is not None and not seg.is_empty and seg.geom_type == 'LineString' and seg.length > 0:
                parts.append(seg)

        if len(parts) == 0:
            target_result[tgt_idx] = (old_len_str, old_len_str)
            continue

        if len(parts) == 1:
            # Chỉ trim phần thừa, không tách → cập nhật geometry tại chỗ
            if abs(parts[0].length - L) > 1e-8:
                gdf.at[tgt_idx, "geometry"] = parts[0]
                recalc_length(gdf, tgt_idx)
            target_result[tgt_idx] = (old_len_str, str(round(parts[0].length, 4)))
            continue

        # ≥2 phần → tách target thành nhiều đoạn (một bộ duy nhất, không chồng lặp)
        targets_to_remove.add(tgt_idx)
        tgt_row = gdf.loc[tgt_idx].copy()
        part_lens = []
        for part_i, part_geom in enumerate(parts):
            new_row = tgt_row.copy()
            new_id = f"{tgt_id}-{part_i + 1}"
            if id_col:
                new_row[id_col] = new_id
            new_row['geometry'] = part_geom
            new_row['ID'] = new_id
            new_row['Len'] = round(part_geom.length, 2)
            part_lens.append(f"{part_geom.length:.4f}")
            new_rows.append(new_row)
        target_result[tgt_idx] = (old_len_str, "+".join(part_lens))

    # ── Tổng hợp báo cáo theo source (trước khi đổi index của gdf) ──
    results_by_source = {}
    for src_idx, info in source_info.items():
        old_lens, new_lens = [], []
        for ti in info["tgt_idxs"]:
            ol, nl = target_result.get(ti, ("", ""))
            old_lens.append(ol)
            new_lens.append(nl)
        results_by_source[src_idx] = {
            "ID-Source": info["ID-Source"],
            "OldLen-Source": info["OldLen-Source"],
            "NewLen-Source": round(gdf.at[src_idx, "geometry"].length, 4)
                             if src_idx in gdf.index else info["OldLen-Source"],
            "ID-Target": ",".join(info["tgt_ids"]),
            "OldLen-Target": ",".join(str(x) for x in old_lens),
            "NewLen-Target": ",".join(str(x) for x in new_lens),
            "Angle": ",".join(str(a) for a in info["angles"]),
            "Status": "FIXED" if info["all_ok"] else "SKIPPED",
        }

    # ── Áp dụng thay đổi: xóa target đã tách, thêm các đoạn mới ──
    if targets_to_remove:
        gdf = gdf.drop(index=list(targets_to_remove))

    if new_rows:
        new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, new_gdf], ignore_index=True), crs=gdf.crs)

    # Đảm bảo kiểu dữ liệu
    if 'ID' in gdf.columns:
        gdf['ID'] = gdf['ID'].astype(str)
    if 'Len' in gdf.columns:
        gdf['Len'] = gdf['Len'].astype(float)

    return list(results_by_source.values()), gdf


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    all_results = []

    # Lặp detect → correct tối đa ITERATIONS_MAX lần
    for iteration in range(1, ITERATIONS_MAX + 1):
        # Phát hiện lỗi
        detections = detect_overshoot(gdf)

        if not detections:
            break

        # Sửa lỗi
        results, gdf = correct_overshoot(gdf, detections)
        all_results.extend(results)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in all_results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(all_results)} trường hợp Overshoot ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(all_results)} trường hợp Overshoot ✅")
    print(f"{'STT':<5} | {'ID-Source':<38} | {'OldLen-Source':<14} | {'NewLen-Source':<14} | {'ID-Target':<38} | {'OldLen-Target':<14} | {'NewLen-Target':<25} | {'Angle':<10} | Status")
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} | {r['ID-Source']:<38} | {r['OldLen-Source']:<14} | {r['NewLen-Source']:<14} | {r['ID-Target']:<38} | {r['OldLen-Target']:<14} | {r['NewLen-Target']:<25} | {r['Angle']:<10} | {r['Status']}")


if __name__ == "__main__":
    main()
