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
from shapely.ops import substring, nearest_points
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


# ── Hàm hỗ trợ: tính góc giữa 2 vector ───────────────────────
def _calc_angle(v1, v2):
    """Tính góc giữa 2 vector (0°-180°)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    ang = math.degrees(math.acos(cos_val))
    return ang


# ── Hàm hỗ trợ: tìm vector hướng segment gần pt nhất ─────────
def _get_seg_vector(geom, pt):
    """Tìm vector hướng segment của geom chứa điểm pt."""
    coords = list(geom.coords)
    best_seg = None
    best_dist = float('inf')
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        d = seg.distance(pt)
        if d < best_dist:
            best_dist = d
            best_seg = (coords[i], coords[i + 1])
    if best_seg is None:
        return None
    p1, p2 = best_seg
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return dx, dy


# ── Hàm hỗ trợ: kéo dài line source theo hướng đến target ─────
def _extend_along_direction(geom_src, vsu_type, geom_tgt, max_extend=1.0):
    """
    Kéo dài line source theo hướng của line source đến khi giao line target.

    Args:
        geom_src: geometry line source
        vsu_type: "vs" hoặc "ve"
        geom_tgt: geometry line target
        max_extend: khoảng cách tối đa kéo dài (mét)

    Returns:
        Point giao hoặc None nếu không giao
    """
    coords = list(geom_src.coords)
    if vsu_type == "vs":
        # Hướng kéo dài: từ coords[1] → coords[0] (tiếp tục ra ngoài)
        dx = coords[0][0] - coords[1][0]
        dy = coords[0][1] - coords[1][1]
        origin = coords[0]
    else:
        # Hướng kéo dài: từ coords[-2] → coords[-1] (tiếp tục ra ngoài)
        dx = coords[-1][0] - coords[-2][0]
        dy = coords[-1][1] - coords[-2][1]
        origin = coords[-1]

    mag = math.hypot(dx, dy)
    if mag == 0:
        return None

    # Normalize và kéo dài
    dx_n, dy_n = dx / mag, dy / mag
    extended_pt = (origin[0] + dx_n * max_extend, origin[1] + dy_n * max_extend)

    ray = LineString([(origin[0], origin[1]), extended_pt])
    intersection = ray.intersection(geom_tgt)

    if intersection.is_empty:
        return None

    if intersection.geom_type == 'Point':
        return intersection
    elif intersection.geom_type == 'MultiPoint':
        # Trả về điểm gần vsu nhất
        vsu_pt = Point(origin)
        return min(intersection.geoms, key=lambda p: p.distance(vsu_pt))
    else:
        # LineString overlap hoặc GeometryCollection — bỏ qua
        return None


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_undershoot(gdf: gpd.GeoDataFrame, dist_threshold: float = DIST_THRESHOLD) -> list[dict]:
    """
    Phát hiện lỗi Undershoot.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng G.degree lọc line source có endpoint bậc 1 (vsu).
    Bước 3: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 4: Kiểm tra từng cặp (source, target):
        - Dùng disjoint kiểm tra: chỉ lấy cặp KHÔNG giao nhau (disjoint=True)
        - 0 < dist(vsu, line target) <= dist_threshold → tiếp tục
        → LỖI UNDERSHOOT

    Returns:
        List[dict] với mỗi phần tử chứa:
          - src_idx: index line source
          - tgt_idx: index line target
          - vsu: endpoint bậc 1 của source (Point)
          - vsu_coord: tọa độ vsu đã làm tròn
          - vsu_type: "vs" hoặc "ve"
          - nearest_pt: điểm gần nhất trên target
          - dist_to_tgt: khoảng cách từ vsu đến target
    """
    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    tree, valid_idx = build_strtree(gdf)
    geoms = gdf.geometry[valid_idx].values

    # Bước 2: Xây dựng đồ thị để xác định bậc
    G = build_graph(gdf)

    # Tìm tất cả line source có endpoint bậc 1
    deg1_sources = {}  # src_idx -> list of vsu info
    for vi, idx in enumerate(valid_idx):
        geom = gdf.at[idx, "geometry"]
        vs_coord = _round_pt(geom.coords[0])
        ve_coord = _round_pt(geom.coords[-1])

        vsu_list = []
        if vs_coord in G and G.degree(vs_coord) == 1:
            vsu_list.append(("vs", Point(vs_coord), vs_coord))
        if ve_coord in G and G.degree(ve_coord) == 1:
            vsu_list.append(("ve", Point(ve_coord), ve_coord))

        if vsu_list:
            deg1_sources[idx] = vsu_list

    if not deg1_sources:
        return []

    results = []
    seen_pairs = set()

    # Bước 3: Dùng STRtree — mở rộng buffer tìm gần nhau
    # Tạo buffer geometry cho các source có endpoint bậc 1
    for src_idx, vsu_list in deg1_sources.items():
        geom_src = gdf.at[src_idx, "geometry"]

        for vsu_type, vsu_pt, vsu_coord in vsu_list:
            # Tìm các target gần vsu bằng STRtree query
            buffered_vsu = vsu_pt.buffer(dist_threshold)
            candidate_indices = tree.query(buffered_vsu)

            for ci in candidate_indices:
                tgt_idx = valid_idx[ci]

                if tgt_idx == src_idx:
                    continue

                pair_key = (src_idx, vsu_type, tgt_idx)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                geom_tgt = gdf.at[tgt_idx, "geometry"]

                # Bước 4: Kiểm tra disjoint — chỉ lấy cặp KHÔNG giao nhau
                if not geom_src.disjoint(geom_tgt):
                    continue

                # Tính khoảng cách từ vsu đến line target
                dist_to_tgt = vsu_pt.distance(geom_tgt)

                # 0 < dist(vsu, line target) <= dist_threshold
                if dist_to_tgt <= 0 or dist_to_tgt > dist_threshold:
                    continue

                # Tìm điểm gần nhất trên target
                _, nearest_pt = nearest_points(vsu_pt, geom_tgt)

                # → LỖI UNDERSHOOT
                results.append({
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx,
                    "vsu": vsu_pt,
                    "vsu_coord": vsu_coord,
                    "vsu_type": vsu_type,
                    "nearest_pt": nearest_pt,
                    "dist_to_tgt": dist_to_tgt,
                })

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_undershoot(gdf: gpd.GeoDataFrame, detections: list[dict], dist_threshold: float = DIST_THRESHOLD) -> tuple[list[dict], gpd.GeoDataFrame]:
    """
    Sửa lỗi Undershoot:

    Nguyên tắc:
      - Mỗi line target có thể xử lý bởi một hoặc nhiều line source
      - Mỗi line source chỉ xử lý tối đa hai line target (một ở đầu, một ở cuối)
      - Nếu hai line cùng nhắm tới endpoint deg=1 của nhau, chỉ xử lý một chiều

    Bước 1: Xử lý line source:
      - TH1: Nếu điểm gần nhất là endpoint của target → dời vsu đến endpoint đó
      - TH2: Nếu điểm gần nhất không là endpoint:
        - TH2.1: dist(vsu, endpoint target) <= dist_threshold → dời vsu đến endpoint target
        - TH2.2: dist(vsu, target) <= dist_threshold → kéo dài vsu đến target, cắt target tại đó
    Bước 2: Kiểm tra bậc mới tại pi = 2, 3, 4 → FIXED, còn lại → SKIPPED
    Bước 3: Tính lại chiều dài.

    Returns:
        (danh sách kết quả báo cáo, gdf đã cập nhật)
    """
    id_col = _find_id_col(gdf)
    results_by_source = {}
    modified_sources = set()

    # Loại bỏ trường hợp hai line cùng nhắm tới endpoint deg=1 của nhau
    # Chỉ giữ chiều đầu tiên
    mutual_pairs = set()
    for det in detections:
        reverse_key = (det["tgt_idx"], det["src_idx"])
        for det2 in detections:
            if (det2["src_idx"], det2["tgt_idx"]) == reverse_key:
                # Chỉ giữ cặp có src_idx nhỏ hơn
                if det["src_idx"] > det2["src_idx"]:
                    mutual_pairs.add((det["src_idx"], det["vsu_type"], det["tgt_idx"]))

    detections = [d for d in detections
                  if (d["src_idx"], d["vsu_type"], d["tgt_idx"]) not in mutual_pairs]

    # Gom detections theo source
    source_detections = defaultdict(list)
    for det in detections:
        source_detections[det["src_idx"]].append(det)

    # Thu thập tất cả split points cho mỗi target
    target_split_points = defaultdict(list)  # tgt_idx -> list of {pi, src_idx, vsu_type}

    for src_idx, dets in source_detections.items():
        # Mỗi source chỉ xử lý tối đa 2 target (1 ở đầu, 1 ở cuối)
        best_by_type = {}
        for det in dets:
            vsu_type = det["vsu_type"]
            if vsu_type not in best_by_type:
                best_by_type[vsu_type] = det
            else:
                # Chọn target có khoảng cách nhỏ nhất
                if det["dist_to_tgt"] < best_by_type[vsu_type]["dist_to_tgt"]:
                    best_by_type[vsu_type] = det

        selected_dets = list(best_by_type.values())

        src_id = str(gdf.at[src_idx, id_col]) if id_col else str(src_idx)
        old_len_src = round(gdf.at[src_idx, "geometry"].length, 4)

        tgt_ids = []
        angles = []
        status = "FIXED"
        all_ok = True

        for det in selected_dets:
            tgt_idx = det["tgt_idx"]
            vsu = det["vsu"]
            vsu_coord = det["vsu_coord"]
            vsu_type = det["vsu_type"]
            nearest_pt = det["nearest_pt"]

            tgt_id = str(gdf.at[tgt_idx, id_col]) if id_col else str(tgt_idx)
            tgt_ids.append(tgt_id)

            geom_src = gdf.at[src_idx, "geometry"]
            geom_tgt = gdf.at[tgt_idx, "geometry"]

            tgt_vs_coord = _round_pt(geom_tgt.coords[0])
            tgt_ve_coord = _round_pt(geom_tgt.coords[-1])
            tgt_vs = Point(tgt_vs_coord)
            tgt_ve = Point(tgt_ve_coord)

            # Xác định điểm gần nhất là endpoint hay không
            nearest_rounded = _round_pt((nearest_pt.x, nearest_pt.y))
            is_nearest_endpoint = (nearest_rounded == tgt_vs_coord or nearest_rounded == tgt_ve_coord)

            # Tính khoảng cách từ vsu đến hai endpoint của target
            dist_to_vs = vsu.distance(tgt_vs)
            dist_to_ve = vsu.distance(tgt_ve)

            # Xác định điểm đích (pi) để dời/kéo dài vsu
            pi = None
            need_split_target = False

            if is_nearest_endpoint:
                # TH1: Điểm gần nhất là endpoint → dời vsu đến endpoint đó
                pi = nearest_pt
            else:
                # TH2: Điểm gần nhất không phải endpoint
                if dist_to_vs <= dist_threshold or dist_to_ve <= dist_threshold:
                    # TH2.1: Dời vsu đến endpoint gần nhất
                    if dist_to_vs <= dist_to_ve:
                        pi = tgt_vs
                    else:
                        pi = tgt_ve
                else:
                    # TH2.2: Kéo dài line source đến line target theo hướng của line source
                    extended_pi = _extend_along_direction(geom_src, vsu_type, geom_tgt)
                    if extended_pi is not None:
                        pi = extended_pi
                        need_split_target = True
                    else:
                        all_ok = False
                        continue

            if pi is None:
                all_ok = False
                continue

            # Tính góc giao
            v1 = _get_seg_vector(geom_src, vsu)
            v2 = _get_seg_vector(geom_tgt, pi)
            ang = None
            if v1 and v2:
                ang = _calc_angle(v1, v2)
            angles.append(round(ang, 1) if ang is not None else 0.0)

            # ── Xử lý line source — dời/kéo dài vsu đến pi ──
            src_coords = list(geom_src.coords)

            if need_split_target:
                # TH2.2: Defer — lấy tọa độ pi chính xác từ substring() sau
                target_split_points[tgt_idx].append({
                    'pi': pi,
                    'src_idx': src_idx,
                    'vsu_type': vsu_type,
                })
            else:
                # TH1/TH2.1: Dời — thay thế vsu bằng pi (endpoint target)
                if geom_src.has_z:
                    z_val = src_coords[0][2] if vsu_type == "vs" else src_coords[-1][2]
                    pi_coord = (round(pi.x, COORD_PRECISION), round(pi.y, COORD_PRECISION), z_val)
                else:
                    pi_coord = _round_pt((pi.x, pi.y))
                if vsu_type == "vs":
                    new_src_coords = [pi_coord] + src_coords[1:]
                else:
                    new_src_coords = src_coords[:-1] + [pi_coord]

                if len(new_src_coords) >= 2:
                    new_src_geom = LineString(new_src_coords)
                    if not new_src_geom.is_empty and new_src_geom.length > 0:
                        gdf.at[src_idx, "geometry"] = new_src_geom
                        recalc_length(gdf, src_idx)
                        modified_sources.add(src_idx)
                    else:
                        all_ok = False
                else:
                    all_ok = False

        # ── Bước 2: Kiểm tra bậc mới ──
        if not all_ok:
            status = "SKIPPED"

        new_len_src = round(gdf.at[src_idx, "geometry"].length, 4)

        results_by_source[src_idx] = {
            "ID-Source": src_id,
            "OldLen-Source": old_len_src,
            "NewLen-Source": new_len_src,
            "ID-Target": ",".join(tgt_ids),
            "Angle": ",".join(str(a) for a in angles),
            "Status": status,
        }

    # ── Xử lý tất cả target cần split ──
    # Mỗi target có thể bị split bởi nhiều source → split tại tất cả các điểm
    new_rows = []
    targets_to_remove = set()

    for tgt_idx, split_infos in target_split_points.items():
        geom_tgt = gdf.at[tgt_idx, "geometry"]
        tgt_id = str(gdf.at[tgt_idx, id_col]) if id_col else str(tgt_idx)

        # Tính khoảng cách dọc theo target, sắp xếp tăng dần, loại trùng
        split_data = []
        seen_d = set()
        for info in split_infos:
            d = geom_tgt.project(info['pi'])
            d_key = round(d, 10)
            if d_key not in seen_d:
                seen_d.add(d_key)
                split_data.append((d, info))
        split_data.sort(key=lambda x: x[0])

        # Split target thành nhiều phần tại tất cả các split point
        total_length = geom_tgt.length
        boundaries = [0.0] + [d for d, _ in split_data] + [total_length]

        parts = []
        for i in range(len(boundaries) - 1):
            d_start = boundaries[i]
            d_end = boundaries[i + 1]
            if d_end - d_start < 1e-10:
                continue
            part = substring(geom_tgt, d_start, d_end)
            if part is not None and not part.is_empty and part.geom_type == 'LineString' and part.length > 0:
                parts.append(part)

        if len(parts) >= 2:
            # Lấy tọa độ split chính xác từ substring — nằm trên target by construction
            for k, (d, info) in enumerate(split_data):
                if k < len(parts) - 1:
                    pi_coord = parts[k].coords[-1]

                    # Cập nhật source geometry với tọa độ chính xác từ target
                    s_idx = info['src_idx']
                    v_type = info['vsu_type']
                    s_geom = gdf.at[s_idx, "geometry"]
                    s_coords = list(s_geom.coords)

                    # Preserve Z dimension if source geometry has Z
                    if s_geom.has_z:
                        if len(pi_coord) >= 3:
                            pi_coord_z = (round(pi_coord[0], COORD_PRECISION), round(pi_coord[1], COORD_PRECISION), pi_coord[2])
                        else:
                            z_val = s_coords[0][2] if v_type == "vs" else s_coords[-1][2]
                            pi_coord_z = (round(pi_coord[0], COORD_PRECISION), round(pi_coord[1], COORD_PRECISION), z_val)
                    else:
                        pi_coord_z = (round(pi_coord[0], COORD_PRECISION), round(pi_coord[1], COORD_PRECISION))

                    if v_type == "vs":
                        new_s_coords = [pi_coord_z] + s_coords[1:]
                    else:
                        new_s_coords = s_coords[:-1] + [pi_coord_z]

                    if len(new_s_coords) >= 2:
                        new_s_geom = LineString(new_s_coords)
                        if not new_s_geom.is_empty and new_s_geom.length > 0:
                            gdf.at[s_idx, "geometry"] = new_s_geom
                            recalc_length(gdf, s_idx)

            targets_to_remove.add(tgt_idx)
            tgt_row = gdf.loc[tgt_idx].copy()

            for part_i, part_geom in enumerate(parts):
                new_row = tgt_row.copy()
                new_id = f"{tgt_id}-{part_i + 1}"
                if id_col:
                    new_row[id_col] = new_id
                new_row['geometry'] = part_geom
                new_row['ID'] = new_id
                new_row['Len'] = round(part_geom.length, 2)
                new_rows.append(new_row)

    # Cập nhật GeoDataFrame: xóa target đã bị cắt, thêm row mới
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
    skipped_pairs = set()  # frozenset({src_id, tgt_id}) — mutual pair tính 1 lần

    # Lặp detect → correct tối đa ITERATIONS_MAX lần
    for iteration in range(1, ITERATIONS_MAX + 1):
        # Phát hiện lỗi
        detections = detect_undershoot(gdf)

        # Loại bỏ detections có cặp (src, tgt) đã SKIP (kể cả chiều ngược)
        id_col = _find_id_col(gdf)
        if skipped_pairs:
            filtered = []
            for d in detections:
                s_id = str(gdf.at[d["src_idx"], id_col] if id_col else d["src_idx"])
                t_id = str(gdf.at[d["tgt_idx"], id_col] if id_col else d["tgt_idx"])
                if frozenset({s_id, t_id}) not in skipped_pairs:
                    filtered.append(d)
            detections = filtered

        if not detections:
            break

        # Sửa lỗi
        results, gdf = correct_undershoot(gdf, detections)

        # Thu thập cặp bị SKIP
        for r in results:
            if r["Status"] == "SKIPPED":
                for tgt in r["ID-Target"].split(","):
                    skipped_pairs.add(frozenset({r["ID-Source"], tgt.strip()}))

        all_results.extend(results)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in all_results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(all_results)} trường hợp Undershoot ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(all_results)} trường hợp Undershoot ✅")
    print(f"{'STT':<5} | {'ID-Source':<10} | {'OldLen-Source':<14} | {'NewLen-Source':<14} | {'ID-Target':<38} | {'Angle':<10} | Status")
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} | {r['ID-Source']:<10} | {r['OldLen-Source']:<14} | {r['NewLen-Source']:<14} | {r['ID-Target']:<38} | {r['Angle']:<10} | {r['Status']}")


if __name__ == "__main__":
    main()
