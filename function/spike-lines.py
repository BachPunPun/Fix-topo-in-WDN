# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import setup_console

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from pathlib import Path

from read_file import read_shp
from save_file import save_shp
from graph_builder import build_graph
from topo_helpers import recalc_length
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"
ANGLE_MIN = 0
ANGLE_MAX = 11.25


# ── Hàm hỗ trợ: tìm cột ID ───────────────────────────────────
def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID."""
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


# ── Hàm hỗ trợ: tính góc tại vertex giữa ─────────────────────
def _calc_angle_at_vertex(v_prev, v_curr, v_next) -> float:
    """
    Tính interior angle (độ) tại v_curr giữa v_prev và v_next.

    w1 = v_prev - v_curr
    w2 = v_next - v_curr
    góc = arccos(w1·w2 / (|w1|×|w2|))
    """
    w1 = np.array(v_prev) - np.array(v_curr)
    w2 = np.array(v_next) - np.array(v_curr)

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    if norm1 == 0 or norm2 == 0:
        return 180.0  # Vertex trùng nhau → không phải spike

    cos_angle = np.dot(w1, w2) / (norm1 * norm2)
    # Clamp để tránh lỗi floating point
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg


# ── Hàm hỗ trợ: kiểm tra vertex có trùng ─────────────────────
def _has_duplicate_vertices(coords) -> bool:
    """Kiểm tra đường thẳng có vertex trùng lặp (đầu/cuối trùng vertex trung gian)."""
    vs = (round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION))
    ve = (round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION))
    for i in range(1, len(coords) - 1):
        pt = (round(coords[i][0], COORD_PRECISION), round(coords[i][1], COORD_PRECISION))
        if pt == vs or pt == ve:
            return True
    return False


# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_spike_lines(gdf: gpd.GeoDataFrame, angle_min: float = ANGLE_MIN, angle_max: float = ANGLE_MAX) -> list[dict]:
    """
    Phát hiện lỗi Spike Lines.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Lọc chi tiết — bỏ qua đường có < 3 vertices.
    Bước 3: Kiểm tra từng bộ 3 vertex liên tiếp, tính góc.

    Returns:
        List[dict] với mỗi phần tử chứa:
          - idx: index trong gdf
          - spike_vertices: list các vị trí vertex spike (index trong coords)
          - angles: list góc tương ứng tại mỗi spike vertex
    """
    results = []

    # Bước 1: Lọc sơ bộ — chỉ lấy LineString
    mask = gdf.geometry.notna() & (gdf.geometry.type == "LineString")
    valid_indices = gdf.index[mask].values

    for idx in valid_indices:
        geom = gdf.at[idx, "geometry"]
        coords = list(geom.coords)

        # Bước 2: Lọc chi tiết — bỏ qua đường < 3 vertices
        if len(coords) < 3:
            continue

        # Bước 3: Kiểm tra vertex trùng lặp → bỏ qua (lỗi duplicate vertices)
        if _has_duplicate_vertices(coords):
            continue

        # Duyệt qua từng bộ 3 vertex liên tiếp
        spike_vertices = []
        angles = []
        for i in range(1, len(coords) - 1):
            v_prev = coords[i - 1]
            v_curr = coords[i]
            v_next = coords[i + 1]
            angle = _calc_angle_at_vertex(v_prev, v_curr, v_next)

            # Nếu angle_min < góc <= angle_max → LỖI SPIKE LINES
            if angle_min < angle <= angle_max:
                spike_vertices.append(i)
                angles.append(round(angle, COORD_PRECISION))

        if spike_vertices:
            results.append({
                "idx": idx,
                "spike_vertices": spike_vertices,
                "angles": angles,
            })

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_spike_lines(gdf: gpd.GeoDataFrame, detections: list[dict], angle_min: float = ANGLE_MIN, angle_max: float = ANGLE_MAX) -> list[dict]:
    """
    Sửa lỗi Spike Lines:
    - Xây dựng đồ thị để xác định junction nodes (degree >= 2)
    - Với mỗi spike vertex:
      + Nếu KHÔNG phải junction → xóa spike vertex
      + Nếu là junction → xóa vertex liền kề gần hơn
    - Lặp lại cho đến khi không còn spike (max_iterations = len(vertices))

    Returns:
        Danh sách kết quả báo cáo.
    """
    # Xây dựng đồ thị để xác định junction nodes
    G = build_graph(gdf)
    junction_nodes = set()
    for node, deg in G.degree():
        if deg >= 2:
            junction_nodes.add(node)

    id_col = _find_id_col(gdf)
    results = []

    for det in detections:
        idx = det["idx"]
        geom = gdf.at[idx, "geometry"]
        coords = list(geom.coords)
        old_len = round(geom.length, COORD_PRECISION)
        num_vertices_orig = len(coords)
        total_spikes_found = len(det["spike_vertices"])
        min_angle = min(det["angles"])

        # Lặp lại cho đến khi hết spike hoặc đạt max_iterations
        max_iterations = len(coords)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Tìm spike trong coords hiện tại
            spike_idx = None
            for i in range(1, len(coords) - 1):
                v_prev = coords[i - 1]
                v_curr = coords[i]
                v_next = coords[i + 1]
                angle = _calc_angle_at_vertex(v_prev, v_curr, v_next)
                if angle_min < angle <= angle_max:
                    spike_idx = i
                    break

            if spike_idx is None:
                break  # Không còn spike

            v_spike = coords[spike_idx]
            spike_node = (round(v_spike[0], COORD_PRECISION), round(v_spike[1], COORD_PRECISION))

            # Kiểm tra spike vertex có phải junction node không
            if spike_node not in junction_nodes:
                # Không phải junction → xóa spike vertex
                coords.pop(spike_idx)
            else:
                # Là junction → xóa vertex liền kề gần hơn
                v_prev = coords[spike_idx - 1]
                v_next = coords[spike_idx + 1]

                dist_prev = np.linalg.norm(np.array(v_spike) - np.array(v_prev))
                dist_next = np.linalg.norm(np.array(v_spike) - np.array(v_next))

                # Xác định vertex liền kề nào là junction (cần bảo vệ)
                prev_node = (round(v_prev[0], COORD_PRECISION), round(v_prev[1], COORD_PRECISION))
                next_node = (round(v_next[0], COORD_PRECISION), round(v_next[1], COORD_PRECISION))
                prev_is_junction = prev_node in junction_nodes
                next_is_junction = next_node in junction_nodes

                if dist_prev <= dist_next:
                    # Ưu tiên xóa v_prev (gần hơn)
                    if not prev_is_junction:
                        coords.pop(spike_idx - 1)
                    elif not next_is_junction:
                        coords.pop(spike_idx + 1)
                    else:
                        # Cả hai liền kề đều junction → xóa spike vertex
                        coords.pop(spike_idx)
                else:
                    # Ưu tiên xóa v_next (gần hơn)
                    if not next_is_junction:
                        coords.pop(spike_idx + 1)
                    elif not prev_is_junction:
                        coords.pop(spike_idx - 1)
                    else:
                        # Cả hai liền kề đều junction → xóa spike vertex
                        coords.pop(spike_idx)

            # Nếu còn < 2 vertex thì dừng
            if len(coords) < 2:
                break

        # Cập nhật geometry
        if len(coords) >= 2:
            new_geom = LineString(coords)
            gdf.at[idx, "geometry"] = new_geom
            new_len = recalc_length(gdf, idx)
        else:
            new_len = 0.0

        row_id = str(gdf.at[idx, id_col]) if id_col else str(idx)
        spike_at_vertex = ", ".join(str(v) for v in det["spike_vertices"])

        results.append({
            "ID": row_id,
            "No-Vertices": num_vertices_orig,
            "Angle": min_angle,
            "No-Spike": total_spikes_found,
            "Spike-at-Vertex": spike_at_vertex,
            "OldLen": old_len,
            "NewLen": new_len,
            "Status": "FIXED",
        })

    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Đọc file
    gdf, _ = load_layers()

    # Phát hiện lỗi
    detections = detect_spike_lines(gdf)

    # Sửa lỗi
    results = correct_spike_lines(gdf, detections)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(results)} trường hợp Spike Lines ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(results)} trường hợp Spike Lines ✅")
    print(f"{'STT':<5} | {'ID':<38} | {'No-Vertices':<12} | {'Angle':<8} | {'No-Spike':<9} | {'Spike-at-Vertex':<16} | {'OldLen':<12} | {'NewLen':<12} | Status")
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['ID']:<38} | {r['No-Vertices']:<12} | {r['Angle']:<8.4f} | {r['No-Spike']:<9} | {r['Spike-at-Vertex']:<16} | {r['OldLen']:<12} | {r['NewLen']:<12} | {r['Status']}")


if __name__ == "__main__":
    main()
