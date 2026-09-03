# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
from shapely import STRtree
from shapely.ops import substring
from collections import defaultdict
from pathlib import Path

from read_file import read_shp
from save_file import save_shp
from topo_helpers import build_strtree
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"
ANGLE_MIN = 30
ANGLE_MAX = 150
DIST_MIN = 0.1  # Khoảng cách tối thiểu từ pi đến đầu/cuối line target


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


# ── Hàm hỗ trợ: kiểm tra endpoint ────────────────────────────
def _is_endpoint(pt, line):
    """Kiểm tra pt có phải đầu/cuối của line."""
    vs, ve = _get_endpoints(line)
    rpt = _round_pt(pt)
    return rpt == vs or rpt == ve


# ── Hàm hỗ trợ: tính góc giữa 2 segment ──────────────────────
def _angle_between_segments(seg1_start, seg1_end, seg2_start, seg2_end):
    """Tính góc giữa 2 segment (0-180°)."""
    d1 = np.array(seg1_end) - np.array(seg1_start)
    d2 = np.array(seg2_end) - np.array(seg2_start)
    len1 = np.linalg.norm(d1)
    len2 = np.linalg.norm(d2)
    if len1 == 0 or len2 == 0:
        return 0
    cos_angle = np.dot(d1, d2) / (len1 * len2)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(abs(cos_angle)))
    return angle


# ── Hàm hỗ trợ: tìm segment chứa điểm ────────────────────────
def _find_segment_at_point(line, pt):
    """Tìm segment của line chứa điểm pt, trả về (seg_start, seg_end)."""
    coords = list(line.coords)
    pt_arr = np.array(pt[:2])
    best_seg = None
    best_dist = float('inf')
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        d = seg.distance(Point(pt_arr))
        if d < best_dist:
            best_dist = d
            best_seg = (coords[i], coords[i + 1])
    return best_seg


# ── Hàm hỗ trợ: trích xuất 1 giao điểm duy nhất ──────────────
def _extract_single_point(geom):
    """
    Trích xuất giao điểm duy nhất từ geometry giao nhau.
    Chỉ trả về nếu có đúng 1 point.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Point':
        return (geom.x, geom.y)
    if geom.geom_type == 'MultiPoint':
        if len(geom.geoms) == 1:
            p = geom.geoms[0]
            return (p.x, p.y)
        return None
    if geom.geom_type == 'GeometryCollection':
        points = [g for g in geom.geoms if g.geom_type == 'Point']
        if len(points) == 1:
            return (points[0].x, points[0].y)
        return None
    return None


# ── Hàm hỗ trợ: trích xuất TẤT CẢ giao điểm ──────────────────
def _extract_points(geom):
    """
    Trả về danh sách TẤT CẢ giao điểm (list tọa độ) từ kết quả intersection.
    Cho phép xử lý trường hợp source giao target tại nhiều điểm — ví dụ cả hai
    đầu của source đều nằm trên thân target.
    """
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Point':
        return [(geom.x, geom.y)]
    if geom.geom_type == 'MultiPoint':
        return [(p.x, p.y) for p in geom.geoms]
    if geom.geom_type == 'GeometryCollection':
        return [(g.x, g.y) for g in geom.geoms if g.geom_type == 'Point']
    return []


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_t_junction(gdf: gpd.GeoDataFrame, dist_min: float = DIST_MIN) -> dict:
    """
    Phát hiện lỗi T-Junction.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 3: Kiểm tra từng cặp đường thẳng (source, target):
        - Tính intersection → xét MỌI giao điểm (kể cả khi cả 2 đầu
          source cùng nằm trên thân target → 2 điểm cắt)
        - pi phải là endpoint của source
        - pi KHÔNG phải endpoint của target
        - Khoảng cách từ pi đến đầu/cuối target >= dist_min
        - Góc giao từ ANGLE_MIN đến ANGLE_MAX

    Returns:
        dict[tgt_idx] -> list of {source_idx, source_id, pi, angle}
    """
    id_col = _find_id_col(gdf)

    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    mask = gdf.geometry.notna() & (gdf.geometry.type == 'LineString')
    gdf_lines = gdf[mask].copy()
    gdf_lines = gdf_lines.reset_index(drop=True)

    # Bước 2: Tạo Spatial Index
    tree = STRtree(gdf_lines.geometry.values)
    geom_arr = gdf_lines.geometry.values

    # Lưu kết quả phát hiện: dict[target_idx] -> list of {source_idx, pi, angle}
    tjunction_errors = defaultdict(list)

    for src_idx in range(len(gdf_lines)):
        src_line = geom_arr[src_idx]
        src_vs, src_ve = _get_endpoints(src_line)

        # Tìm các line gần src_line
        candidates = tree.query(src_line)

        for tgt_idx in candidates:
            if src_idx == tgt_idx:
                continue

            tgt_line = geom_arr[tgt_idx]

            # Bước 3: Tính geometry giao nhau — xét MỌI giao điểm
            inter = src_line.intersection(tgt_line)

            for pi in _extract_points(inter):
                rpi = _round_pt(pi)

                # Kiểm tra pi là endpoint của source
                if rpi != src_vs and rpi != src_ve:
                    continue

                # Kiểm tra pi KHÔNG phải endpoint của target
                if _is_endpoint(pi, tgt_line):
                    continue

                # Kiểm tra khoảng cách từ pi đến đầu/cuối của target >= dist_min
                tgt_vs, tgt_ve = _get_endpoints(tgt_line)
                dist_vs = Point(pi).distance(Point(tgt_vs))
                dist_ve = Point(pi).distance(Point(tgt_ve))
                if dist_vs < dist_min or dist_ve < dist_min:
                    continue

                # Tính góc giao
                seg_src = _find_segment_at_point(src_line, pi)
                seg_tgt = _find_segment_at_point(tgt_line, pi)
                if seg_src is None or seg_tgt is None:
                    continue

                angle = _angle_between_segments(seg_src[0], seg_src[1], seg_tgt[0], seg_tgt[1])

                if angle < ANGLE_MIN or angle > ANGLE_MAX:
                    continue

                # → LỖI T-JUNCTION (mỗi đầu source nằm trên thân target = 1 điểm cắt)
                src_id = str(gdf_lines.iloc[src_idx][id_col]) if id_col else str(src_idx)
                tjunction_errors[tgt_idx].append({
                    'source_idx': src_idx,
                    'source_id': src_id,
                    'pi': pi,
                    'angle': round(angle, 1),
                })

    return tjunction_errors, gdf_lines


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_t_junction(gdf_lines: gpd.GeoDataFrame, tjunction_errors: dict) -> tuple:
    """
    Sửa lỗi T-Junction:

    Bước 1: Sắp xếp các pi theo thứ tự trên line target.
            Dùng substring cắt line target tại tất cả pi.
            Cập nhật ID: ID-1, ID-2, ID-3...
    Bước 2: Kiểm tra số parts >= 2 → FIXED, còn lại → SKIPPED.
    Bước 3: Tính lại chiều dài các đoạn sau khi cắt.

    Returns:
        (result_gdf, report_results)
    """
    id_col = _find_id_col(gdf_lines)
    geom_arr = gdf_lines.geometry.values

    new_rows = []
    results = []
    processed_targets = set()

    # Sắp xếp theo ID-Target
    sorted_targets = sorted(tjunction_errors.keys(),
                            key=lambda idx: str(gdf_lines.iloc[idx][id_col]) if id_col else str(idx))

    for tgt_idx in sorted_targets:
        errors = tjunction_errors[tgt_idx]
        tgt_line = geom_arr[tgt_idx]
        tgt_id = str(gdf_lines.iloc[tgt_idx][id_col]) if id_col else str(tgt_idx)
        old_len = tgt_line.length

        # Lấy tất cả pi trên target này
        pi_list = [e['pi'] for e in errors]
        source_ids = sorted(set(e['source_id'] for e in errors))
        angles = [e['angle'] for e in errors]

        # Sắp xếp pi theo thứ tự từ đầu đến cuối trên line target
        # Dùng project để tìm vị trí trên line
        pi_distances = []
        for pi in pi_list:
            d = tgt_line.project(Point(pi))
            pi_distances.append((d, pi))
        pi_distances.sort(key=lambda x: x[0])

        # Loại bỏ các pi trùng nhau (cùng vị trí trên line target)
        unique_cuts = []
        for d, pi in pi_distances:
            if not unique_cuts or abs(d - unique_cuts[-1][0]) > 1e-8:
                unique_cuts.append((d, pi))

        # Cắt line target sử dụng substring
        cut_distances = [item[0] for item in unique_cuts]
        total_length = tgt_line.length

        status = "FIXED"
        parts = []

        # Tạo các đoạn cắt
        breakpoints = [0.0] + cut_distances + [total_length]
        for i in range(len(breakpoints) - 1):
            start_d = breakpoints[i]
            end_d = breakpoints[i + 1]
            if end_d - start_d < 1e-10:
                continue
            segment = substring(tgt_line, start_d, end_d)
            if segment is not None and not segment.is_empty and segment.geom_type == 'LineString' and segment.length > 0:
                parts.append(segment)

        if len(parts) < 2:
            status = "SKIPPED"
            continue

        processed_targets.add(tgt_idx)

        # Tạo các row mới với ID cập nhật
        tgt_row = gdf_lines.iloc[tgt_idx].copy()
        new_lens = []

        for part_i, part_geom in enumerate(parts):
            new_row = tgt_row.copy()
            new_id = f"{tgt_id}-{part_i + 1}"
            if id_col:
                new_row[id_col] = new_id
            new_row['geometry'] = part_geom
            new_row['ID'] = new_id
            new_row['Len'] = round(part_geom.length, 2)
            new_lens.append(f"{part_geom.length:.2f}")
            new_rows.append(new_row)

        # Báo cáo
        old_len_str = f"{old_len:.2f}"
        new_len_str = "+".join(new_lens)
        source_ids_str = ",".join(source_ids)
        angles_str = ",".join(str(a) for a in sorted(set(angles)))

        results.append({
            'ID-Target': tgt_id,
            'OldLen-Target': old_len_str,
            'NewLen-Target': new_len_str,
            'ID-Source': source_ids_str,
            'Parts': len(parts),
            'Angle': angles_str,
            'Status': status,
        })

    # Tạo GeoDataFrame kết quả
    # Xóa các target đã xử lý khỏi gdf gốc
    remaining = gdf_lines.drop(index=list(processed_targets))

    # Đảm bảo cột ID và Len
    if 'ID' not in remaining.columns:
        remaining['ID'] = remaining[id_col].astype(str) if id_col else remaining.index.astype(str)
    if 'Len' not in remaining.columns:
        remaining['Len'] = remaining.geometry.length.round(2)

    # Thêm các row mới
    if new_rows:
        new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf_lines.crs)
        result_gdf = gpd.GeoDataFrame(pd.concat([remaining, new_gdf], ignore_index=True), crs=gdf_lines.crs)
    else:
        result_gdf = remaining.copy()

    # Đảm bảo kiểu dữ liệu
    result_gdf['ID'] = result_gdf['ID'].astype(str)
    result_gdf['Len'] = result_gdf['Len'].astype(float)

    return result_gdf, results, len(tjunction_errors), len(processed_targets)


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    # Phát hiện lỗi
    tjunction_errors, gdf_lines = detect_t_junction(gdf)

    if not tjunction_errors:
        save_shp(gdf, OUTPUT_PATH)
        print("Phát hiện: 0 trường hợp T-Junction ✅")
        print("Sửa lỗi: 0/0 trường hợp T-Junction ✅")
        return

    # Sửa lỗi
    result_gdf, results, detected, fixed = correct_t_junction(gdf_lines, tjunction_errors)

    # Lưu file
    save_shp(result_gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    print(f"Phát hiện: {detected} trường hợp T-Junction ✅")
    print(f"Sửa lỗi: {fixed}/{detected} trường hợp T-Junction ✅")
    print(f"{'STT':<5} | {'ID-Target':<12} | {'OldLen-Target':<14} | {'NewLen-Target':<25} | {'ID-Source':<20} | {'Parts':<5} | {'Angle':<10} | Status")
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['ID-Target']:<12} | {r['OldLen-Target']:<14} | {r['NewLen-Target']:<25} | {r['ID-Source']:<20} | {r['Parts']:<5} | {r['Angle']:<10} | {r['Status']}")


if __name__ == "__main__":
    main()
