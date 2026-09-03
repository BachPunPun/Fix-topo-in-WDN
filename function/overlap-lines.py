# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import math

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely import STRtree
from pathlib import Path

from read_file import read_shp
from save_file import save_shp
from topo_helpers import build_strtree, recalc_length
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"
ITERATIONS_MAX = 5
RATIO_DELETE_THRESHOLD = 0.95
MATERIAL_GROUPS = [
    {'PVC', 'uPVC', 'HDPE'},
    {'DI', 'CI', 'BTCT-TA-KNT'},
]


# ── Hàm hỗ trợ: tìm cột ID ───────────────────────────────────
def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID."""
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


# ── Hàm hỗ trợ: tìm cột vật liệu ─────────────────────────────
def _find_mat_col(gdf: gpd.GeoDataFrame):
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ('material', 'vatlieu', 'chatlieu'):
        if name in norm:
            return norm[name]
    return None


def _get_mat_group(val) -> int:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return -1
    s = str(val).strip()
    for i, grp in enumerate(MATERIAL_GROUPS):
        if s in grp:
            return i
    return -1


# ── Hàm hỗ trợ: trích xuất thành phần LineString từ intersection ──
def _extract_line_parts(geom) -> list:
    """
    Trích xuất các thành phần LineString có length > 0 từ kết quả intersection.
    Trả về danh sách LineString hoặc rỗng nếu không có đoạn overlap.
    """
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []

    if isinstance(geom, MultiLineString):
        return [part for part in geom.geoms if isinstance(part, LineString) and part.length > 0]

    if isinstance(geom, GeometryCollection):
        parts = []
        for part in geom.geoms:
            if isinstance(part, LineString) and part.length > 0:
                parts.append(part)
            elif isinstance(part, MultiLineString):
                parts.extend(p for p in part.geoms if isinstance(p, LineString) and p.length > 0)
        return parts

    return []


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_overlap_lines(gdf: gpd.GeoDataFrame, ratio_delete_threshold: float = RATIO_DELETE_THRESHOLD) -> list[dict]:
    """
    Phát hiện lỗi Overlap Lines.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Dùng STRtree để tìm nhanh các cặp đường thẳng gần nhau.
    Bước 3: Kiểm tra intersection:
        - Loại trừ đường trùng hoàn toàn (shapely.equals)
        - Nếu intersection là LineString/MultiLineString → LỖI
        - Nếu là Point/MultiPoint → bỏ qua
        - Nếu là GeometryCollection → trích xuất LineString, nếu có → LỖI

    Returns:
        List[dict] với mỗi phần tử chứa:
          - idx_long: index đường dài hơn (giữ lại)
          - idx_short: index đường ngắn hơn (cắt/xóa)
          - intersection_length: chiều dài đoạn giao
    """
    results = []
    seen_pairs = set()

    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    tree, valid_idx = build_strtree(gdf)
    geoms = gdf.geometry[valid_idx].values

    # Bước 2: Dùng STRtree tìm các cặp gần nhau + lọc cùng nhóm vật liệu
    mat_col = _find_mat_col(gdf)
    mat_groups = {}
    for idx in valid_idx:
        val = gdf.at[idx, mat_col] if mat_col else None
        mat_groups[idx] = _get_mat_group(val)

    query_result = tree.query(geoms, predicate="intersects")

    for i_pos, j_pos in zip(query_result[0], query_result[1]):
        idx_i = valid_idx[i_pos]
        idx_j = valid_idx[j_pos]

        # Bỏ qua tự giao và cặp đã xử lý
        if idx_i >= idx_j:
            continue
        pair_key = (idx_i, idx_j)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        geom_i = gdf.at[idx_i, "geometry"]
        geom_j = gdf.at[idx_j, "geometry"]

        # Bước 3: Loại trừ đường trùng hoàn toàn
        if geom_i.equals(geom_j):
            continue

        # Tính intersection
        inter = geom_i.intersection(geom_j)

        # Trích xuất các thành phần LineString
        line_parts = _extract_line_parts(inter)
        if not line_parts:
            continue

        # Tính tổng chiều dài đoạn giao
        intersection_length = sum(part.length for part in line_parts)
        if intersection_length <= 0:
            continue

        # Xác định đường dài và đường ngắn
        len_i = geom_i.length
        len_j = geom_j.length

        if len_i >= len_j:
            idx_long, idx_short = idx_i, idx_j
            len_short = len_j
        else:
            idx_long, idx_short = idx_j, idx_i
            len_short = len_i

        # Lọc nhóm vật liệu: cùng nhóm → giữ; khác nhóm → chỉ giữ nếu đoạn ngắn nằm hoàn toàn trong đoạn dài
        mg_i = mat_groups.get(idx_i, -1)
        mg_j = mat_groups.get(idx_j, -1)
        same_group = (mg_i != -1 and mg_j != -1 and mg_i == mg_j)
        ratio = intersection_length / len_short if len_short > 0 else 1.0
        if not same_group and ratio < ratio_delete_threshold:
            continue

        results.append({
            "idx_long": idx_long,
            "idx_short": idx_short,
            "intersection_length": round(intersection_length, COORD_PRECISION),
        })

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_overlap_lines(gdf: gpd.GeoDataFrame, detections: list[dict], ratio_delete_threshold: float = RATIO_DELETE_THRESHOLD) -> list[dict]:
    """
    Sửa lỗi Overlap Lines theo nguyên tắc "Giữ dài — Sửa ngắn":

    1. Xác định đoạn ngắn và đoạn dài, tính chiều dài đoạn giao.
    2. Tính ratio = chiều dài đoạn giao / chiều dài đoạn ngắn:
       - ratio >= ratio_delete_threshold → XÓA toàn bộ đoạn ngắn (DELETE)
       - ratio <  ratio_delete_threshold → CẮT đoạn ngắn bằng difference(), giữ phần không giao (CUT)
    3. Tính lại chiều dài đường thẳng sau khi sửa lỗi.

    Đánh dấu kết quả:
    - DELETE: Xóa toàn bộ đoạn ngắn
    - CUT: Cắt đoạn ngắn bằng difference
    - SKIPPED: Không thể xóa hoặc cắt

    Returns:
        Danh sách kết quả báo cáo.
    """
    id_col = _find_id_col(gdf)
    results = []
    indices_to_drop = set()

    for det in detections:
        idx_long = det["idx_long"]
        idx_short = det["idx_short"]
        intersection_length = det["intersection_length"]

        # Bỏ qua nếu đoạn ngắn đã bị xóa ở lần xử lý trước
        if idx_short in indices_to_drop:
            continue

        geom_long = gdf.at[idx_long, "geometry"]
        geom_short = gdf.at[idx_short, "geometry"]

        # Kiểm tra geometry còn hợp lệ không
        if geom_short is None or geom_short.is_empty:
            continue

        len_long = round(geom_long.length, COORD_PRECISION)
        len_short = round(geom_short.length, COORD_PRECISION)

        id_long = str(gdf.at[idx_long, id_col]) if id_col else str(idx_long)
        id_short = str(gdf.at[idx_short, id_col]) if id_col else str(idx_short)

        # Tính ratio
        ratio = intersection_length / len_short if len_short > 0 else 1.0

        # Quyết định: xóa hay cắt
        if ratio >= ratio_delete_threshold:
            # XÓA toàn bộ đoạn ngắn
            indices_to_drop.add(idx_short)
            results.append({
                "ID-Keep": id_long,
                "Len-Keep": len_long,
                "ID-Removed": id_short,
                "Len-Removed": len_short,
                "Ratio": round(ratio * 100, 2),
                "Status": "DELETE",
            })
        else:
            # CẮT đoạn ngắn bằng difference()
            try:
                diff_geom = geom_short.difference(geom_long)

                # Xử lý kết quả difference
                if diff_geom is None or diff_geom.is_empty:
                    # difference rỗng → xóa luôn
                    indices_to_drop.add(idx_short)
                    results.append({
                        "ID-Keep": id_long,
                        "Len-Keep": len_long,
                        "ID-Removed": id_short,
                        "Len-Removed": len_short,
                        "Ratio": round(ratio * 100, 2),
                        "Status": "DELETE",
                    })
                elif isinstance(diff_geom, LineString):
                    if diff_geom.length > 0 and len(diff_geom.coords) >= 2:
                        gdf.at[idx_short, "geometry"] = diff_geom
                        new_len = recalc_length(gdf, idx_short)
                        results.append({
                            "ID-Keep": id_long,
                            "Len-Keep": len_long,
                            "ID-Removed": id_short,
                            "Len-Removed": len_short,
                            "Ratio": round(ratio * 100, 2),
                            "Status": "CUT",
                        })
                    else:
                        indices_to_drop.add(idx_short)
                        results.append({
                            "ID-Keep": id_long,
                            "Len-Keep": len_long,
                            "ID-Removed": id_short,
                            "Len-Removed": len_short,
                            "Ratio": round(ratio * 100, 2),
                            "Status": "DELETE",
                        })
                elif isinstance(diff_geom, MultiLineString):
                    # Giữ phần dài nhất
                    parts = [p for p in diff_geom.geoms if isinstance(p, LineString) and p.length > 0]
                    if parts:
                        longest_part = max(parts, key=lambda p: p.length)
                        gdf.at[idx_short, "geometry"] = longest_part
                        new_len = recalc_length(gdf, idx_short)
                        results.append({
                            "ID-Keep": id_long,
                            "Len-Keep": len_long,
                            "ID-Removed": id_short,
                            "Len-Removed": len_short,
                            "Ratio": round(ratio * 100, 2),
                            "Status": "CUT",
                        })
                    else:
                        indices_to_drop.add(idx_short)
                        results.append({
                            "ID-Keep": id_long,
                            "Len-Keep": len_long,
                            "ID-Removed": id_short,
                            "Len-Removed": len_short,
                            "Ratio": round(ratio * 100, 2),
                            "Status": "DELETE",
                        })
                else:
                    # Kết quả difference không phải LineString → SKIPPED
                    results.append({
                        "ID-Keep": id_long,
                        "Len-Keep": len_long,
                        "ID-Removed": id_short,
                        "Len-Removed": len_short,
                        "Ratio": round(ratio * 100, 2),
                        "Status": "SKIPPED",
                    })
            except Exception:
                results.append({
                    "ID-Keep": id_long,
                    "Len-Keep": len_long,
                    "ID-Removed": id_short,
                    "Len-Removed": len_short,
                    "Ratio": round(ratio * 100, 2),
                    "Status": "SKIPPED",
                })

    # Xóa các đoạn đã đánh dấu
    if indices_to_drop:
        gdf.drop(index=list(indices_to_drop), inplace=True)
        gdf.reset_index(drop=True, inplace=True)

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    all_results = []

    # Lặp detect → correct tối đa ITERATIONS_MAX lần
    for iteration in range(1, ITERATIONS_MAX + 1):
        # Phát hiện lỗi
        detections = detect_overlap_lines(gdf)

        if not detections:
            break

        # Sửa lỗi
        results = correct_overlap_lines(gdf, detections)
        all_results.extend(results)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in all_results if r["Status"] in ("DELETE", "CUT"))
    delete_count = sum(1 for r in all_results if r["Status"] == "DELETE")
    cut_count = sum(1 for r in all_results if r["Status"] == "CUT")
    print(f"Phát hiện: {len(all_results)} trường hợp Overlap Lines ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(all_results)} trường hợp Overlap Lines (DELETE: {delete_count}, CUT: {cut_count}) ✅")
    print(f"{'STT':<5} | {'ID-Keep':<38} | {'Len-Keep':<12} | {'ID-Removed':<38} | {'Len-Removed':<12} | {'Ratio':<8} | Status")
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} | {r['ID-Keep']:<38} | {r['Len-Keep']:<12} | {r['ID-Removed']:<38} | {r['Len-Removed']:<12} | {r['Ratio']:<8} | {r['Status']}")


if __name__ == "__main__":
    main()

