# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import math
import geopandas as gpd
from shapely.geometry import Point, LineString, GeometryCollection, MultiPoint
from shapely import STRtree
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


# ── Hàm hỗ trợ: kiểm tra pt có trùng vertex của line ─────────
def _is_vertex(pt, geom, tol=1e-5):
    """Kiểm tra pt có trùng với bất kỳ vertex nào của geometry."""
    return any(pt.distance(Point(c)) < tol for c in geom.coords)


# ── Hàm hỗ trợ: tìm vector hướng segment chứa pt ─────────────
def _get_seg_vector(geom, pt):
    """Tìm vector hướng segment của geom chứa điểm pt."""
    coords = list(geom.coords)
    for i in range(len(coords) - 1):
        p1, p2 = Point(coords[i]), Point(coords[i + 1])
        seg = LineString([p1, p2])
        if seg.distance(pt) < 1e-4:
            dx, dy = p2.x - p1.x, p2.y - p1.y
            return dx, dy
    return None


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


# ── Hàm hỗ trợ: xác định v_origin gần pi nhất ────────────────
def _get_v_origin(geom, pt, dist_threshold: float = DIST_THRESHOLD):
    """
    Trả về endpoint (vs hoặc ve) gần pt nhất.
    Returns (v_origin_point, v_origin_coord_rounded) hoặc None nếu khoảng cách > dist_threshold.
    """
    vs = Point(geom.coords[0])
    ve = Point(geom.coords[-1])
    d_start = pt.distance(vs)
    d_end = pt.distance(ve)

    if d_start < dist_threshold or d_end < dist_threshold:
        if d_start <= d_end:
            v_origin = vs
        else:
            v_origin = ve
        coord = (round(v_origin.x, COORD_PRECISION), round(v_origin.y, COORD_PRECISION))
        return v_origin, coord

    return None


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_crossing(gdf: gpd.GeoDataFrame, dist_threshold: float = DIST_THRESHOLD) -> list[dict]:
    """
    Phát hiện lỗi Crossing.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 3: Kiểm tra từng cặp đường thẳng:
        - Tính intersection → chỉ lấy 1 giao điểm duy nhất
        - Kiểm tra pi không phải vertex của cả hai đường
        - Kiểm tra góc giao từ 30° đến 150°
        - Kiểm tra khoảng cách từ pi đến endpoint < dist_threshold

    Returns:
        List[dict] với mỗi phần tử chứa:
          - idx1, idx2: index cặp đường thẳng
          - pi: giao điểm (Point)
          - v_origin1, v_origin2: endpoint gần pi (Point)
          - coord1, coord2: tọa độ v_origin đã làm tròn
          - angle: góc tại giao điểm
    """
    results = []
    seen_pairs = set()

    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    tree, valid_idx = build_strtree(gdf)
    geoms = gdf.geometry[valid_idx].values

    # Bước 2: Dùng STRtree tìm các cặp gần nhau
    left, right = tree.query(geoms, predicate='intersects')

    for l, r in zip(left, right):
        idx1 = valid_idx[l]
        idx2 = valid_idx[r]

        # Bỏ qua tự giao và cặp đã xử lý
        if idx1 >= idx2:
            continue
        pair_key = (idx1, idx2)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        geom1 = gdf.at[idx1, "geometry"]
        geom2 = gdf.at[idx2, "geometry"]

        # Bước 3: Tính intersection
        inter = geom1.intersection(geom2)

        # Chỉ lấy 1 giao điểm duy nhất
        pt = _extract_single_point(inter)
        if pt is None:
            continue

        # Kiểm tra pi không phải vertex của cả hai đường
        if _is_vertex(pt, geom1) and _is_vertex(pt, geom2):
            continue

        # Tính góc giao
        v1 = _get_seg_vector(geom1, pt)
        v2 = _get_seg_vector(geom2, pt)
        if not v1 or not v2:
            continue

        ang = _calc_angle(v1, v2)
        if ang is None:
            continue

        # Kiểm tra góc trong khoảng [ANGEL_MIN, ANGEL_MAX]
        if not (ANGEL_MIN <= ang <= ANGEL_MAX):
            continue

        # Kiểm tra khoảng cách từ pi đến endpoint < dist_threshold
        res1 = _get_v_origin(geom1, pt, dist_threshold)
        res2 = _get_v_origin(geom2, pt, dist_threshold)
        if res1 is None or res2 is None:
            continue

        v_origin1, coord1 = res1
        v_origin2, coord2 = res2

        # → LỖI CROSSING
        results.append({
            "idx1": idx1,
            "idx2": idx2,
            "pi": pt,
            "v_origin1": v_origin1,
            "v_origin2": v_origin2,
            "coord1": coord1,
            "coord2": coord2,
            "angle": ang,
        })

    return results


# ── Hàm hỗ trợ: snap endpoint về pi ──────────────────────────
def _snap_endpoint_to_pi(geom, old_coord, pt, tol=1e-5):
    """Dời endpoint của geom từ old_coord về pt."""
    coords = list(geom.coords)
    has_z = geom.has_z
    p_old = Point(old_coord)

    if Point(coords[0]).distance(p_old) < tol:
        coords[0] = (pt.x, pt.y, coords[0][2]) if has_z else (pt.x, pt.y)
    if Point(coords[-1]).distance(p_old) < tol:
        coords[-1] = (pt.x, pt.y, coords[-1][2]) if has_z else (pt.x, pt.y)
    return LineString(coords)


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_crossing(gdf: gpd.GeoDataFrame, detections: list[dict]) -> list[dict]:
    """
    Sửa lỗi Crossing:

    Bước 1: Xác định source (deg=1) và target.
    Bước 2: Dời v_origin về pi. Nếu deg=2 → snap tất cả edge tại v_origin sang pi.
    Bước 3: Kiểm tra bậc mới tại pi = 2, 3, 4 → FIXED, còn lại → SKIPPED.
    Bước 4: Tính lại chiều dài đường thẳng sau khi sửa lỗi.

    Returns:
        Danh sách kết quả báo cáo.
    """
    G = build_graph(gdf)
    id_col = _find_id_col(gdf)
    results = []
    modified_in_iter = set()

    for det in detections:
        idx1 = det["idx1"]
        idx2 = det["idx2"]
        pt = det["pi"]
        coord1 = det["coord1"]
        coord2 = det["coord2"]
        ang = det["angle"]

        # Bỏ qua nếu đã sửa trong iteration này
        if idx1 in modified_in_iter or idx2 in modified_in_iter:
            continue

        # Bước 1: Xác định bậc tại v_origin
        deg1 = G.degree(coord1) if coord1 in G else 0
        deg2 = G.degree(coord2) if coord2 in G else 0

        # Source là line có deg=1, Target là line còn lại
        if deg1 == 1 and deg2 != 1:
            s_idx, t_idx = idx1, idx2
            s_coord, t_coord = coord1, coord2
            s_deg, t_deg = deg1, deg2
        elif deg2 == 1 and deg1 != 1:
            s_idx, t_idx = idx2, idx1
            s_coord, t_coord = coord2, coord1
            s_deg, t_deg = deg2, deg1
        else:
            s_idx, t_idx = idx1, idx2
            s_coord, t_coord = coord1, coord2
            s_deg, t_deg = deg1, deg2

        id_s = str(gdf.at[s_idx, id_col]) if id_col else str(s_idx)
        id_t = str(gdf.at[t_idx, id_col]) if id_col else str(t_idx)

        old_len_s = round(gdf.at[s_idx, "geometry"].length, 3)
        old_len_t = round(gdf.at[t_idx, "geometry"].length, 3)

        # Bước 2: Thu thập tất cả lines cần snap
        lines_to_snap = []
        if s_deg == 2:
            for u, v, k, d in G.edges(s_coord, keys=True, data=True):
                lines_to_snap.append((d['idx'], s_coord))
        else:
            lines_to_snap.append((s_idx, s_coord))

        if t_deg == 2:
            for u, v, k, d in G.edges(t_coord, keys=True, data=True):
                lines_to_snap.append((d['idx'], t_coord))
        else:
            lines_to_snap.append((t_idx, t_coord))

        unique_snaps = {(l_idx, c) for l_idx, c in lines_to_snap}
        new_deg = len(unique_snaps)

        # Bước 3: Kiểm tra bậc mới tại pi
        status = "SKIPPED"
        if new_deg in [2, 3, 4]:
            status = "FIXED"

        new_len_s = old_len_s
        new_len_t = old_len_t

        if status == "FIXED":
            for l_idx, old_c in unique_snaps:
                old_g = gdf.at[l_idx, "geometry"]
                new_g = _snap_endpoint_to_pi(old_g, old_c, pt, tol=1e-5)
                gdf.at[l_idx, "geometry"] = new_g
                modified_in_iter.add(l_idx)
                recalc_length(gdf, l_idx)

            # Bước 4: Tính lại chiều dài
            new_len_s = round(gdf.at[s_idx, "geometry"].length, 3)
            new_len_t = round(gdf.at[t_idx, "geometry"].length, 3)

        results.append({
            "ID-Source": id_s,
            "OldLen-Source": old_len_s,
            "NewLen-Source": new_len_s,
            "ID-Target": id_t,
            "OldLen-Target": old_len_t,
            "NewLen-Target": new_len_t,
            "No-Intersect": 1,
            "Angle": round(ang, 1),
            "Status": status,
        })

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    all_results = []

    # Lặp detect → correct tối đa ITERATIONS_MAX lần
    for iteration in range(1, ITERATIONS_MAX + 1):
        # Phát hiện lỗi
        detections = detect_crossing(gdf)

        if not detections:
            break

        # Sửa lỗi
        results = correct_crossing(gdf, detections)
        all_results.extend(results)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in all_results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(all_results)} trường hợp Crossing ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(all_results)} trường hợp Crossing ✅")
    print(f"{'STT':<5} | {'ID-Source':<38} | {'OldLen-Source':<14} | {'NewLen-Source':<14} | {'ID-Target':<38} | {'OldLen-Target':<14} | {'NewLen-Target':<14} | {'No-Intersect':<13} | {'Angle':<8} | Status")
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} | {r['ID-Source']:<38} | {r['OldLen-Source']:<14} | {r['NewLen-Source']:<14} | {r['ID-Target']:<38} | {r['OldLen-Target']:<14} | {r['NewLen-Target']:<14} | {r['No-Intersect']:<13} | {r['Angle']:<8} | {r['Status']}")


if __name__ == "__main__":
    main()
