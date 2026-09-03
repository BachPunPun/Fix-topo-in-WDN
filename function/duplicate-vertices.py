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
from topo_helpers import recalc_length
from load_layers import load_layers
import config

# ── Cấu hình tham số ──────────────────────────────────────────
COORD_PRECISION = 6
SPIKE_ANGLE_MAX = 1  # Góc tối đa (độ) được coi là spike
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"


# ── Hàm hỗ trợ: tìm cột ID ───────────────────────────────────
def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID."""
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


# ── Hàm hỗ trợ: tính góc tại vertex ──────────────────────────
def _calc_angle_at_vertex(v_prev, v_curr, v_next) -> float:
    """Tính interior angle (độ) tại v_curr giữa v_prev và v_next."""
    w1 = np.array(v_prev) - np.array(v_curr)
    w2 = np.array(v_next) - np.array(v_curr)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    cos_angle = np.clip(np.dot(w1, w2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ── Hàm hỗ trợ: kiểm tra xóa vertex có tạo spike không ───────
def _creates_spike(coords, remove_idx) -> bool:
    """
    Kiểm tra: nếu xóa vertex tại remove_idx, geometry còn lại có tạo spike không?
    Returns True nếu tạo spike (góc ≤ SPIKE_ANGLE_MAX).
    """
    test = [c for j, c in enumerate(coords) if j != remove_idx]
    if len(test) < 3:
        return False
    for i in range(1, len(test) - 1):
        angle = _calc_angle_at_vertex(test[i - 1], test[i], test[i + 1])
        if 0 <= angle <= SPIKE_ANGLE_MAX:
            return True
    return False

# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_duplicate_vertices(gdf: gpd.GeoDataFrame) -> list[dict]:
    """
    Phát hiện lỗi Duplicate Vertices.

    Bước 1: Lọc sơ bộ — chỉ lấy LineString hợp lệ.
    Bước 2: Lọc chi tiết — bỏ qua đường có < 3 vertices.
    Bước 3: Kiểm tra từng đường thẳng — so sánh điểm đầu/cuối
            với các vertices trung gian. Nếu trùng → LỖI.

    Returns:
        List[dict] với mỗi phần tử chứa:
          - idx: index trong gdf
          - dup_indices: list các vị trí vertex trung gian bị trùng (index trong coords)
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

        # Bước 3: So sánh điểm đầu và điểm cuối với các vertices trung gian
        vs = (round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION))
        ve = (round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION))

        dup_indices = []
        for i in range(1, len(coords) - 1):
            pt = (round(coords[i][0], COORD_PRECISION), round(coords[i][1], COORD_PRECISION))
            if pt == vs or pt == ve:
                dup_indices.append(i)

        if dup_indices:
            results.append({
                "idx": idx,
                "dup_indices": dup_indices,
            })

    return results


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_duplicate_vertices(gdf: gpd.GeoDataFrame, detections: list[dict]) -> list[dict]:
    """
    Sửa lỗi Duplicate Vertices:
    - Xóa các đỉnh trung gian bị trùng với đầu/cuối, chỉ giữ lại 1 đỉnh duy nhất.
    - Tính lại chiều dài đường thẳng sau khi xóa.

    Đánh dấu kết quả:
    - FIXED: Xóa vertex trùng thành công
    - SKIPPED: Nếu sau khi xóa còn < 2 đỉnh

    Returns:
        Danh sách kết quả báo cáo.
    """
    id_col = _find_id_col(gdf)
    results = []

    for det in detections:
        idx = det["idx"]
        dup_indices = det["dup_indices"]
        geom = gdf.at[idx, "geometry"]
        coords = list(geom.coords)
        old_len = round(geom.length, COORD_PRECISION)
        num_vertices = len(coords)

        # Tạo chuỗi mô tả các đỉnh trùng (ví dụ: "0=3=5" nghĩa là đỉnh 0, 3, 5 trùng nhau)
        # Nhóm các đỉnh trùng theo tọa độ
        dup_groups = {}
        vs = (round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION))
        ve = (round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION))

        for i in dup_indices:
            pt = (round(coords[i][0], COORD_PRECISION), round(coords[i][1], COORD_PRECISION))
            if pt == vs:
                key = ("start", 0)
            else:
                key = ("end", len(coords) - 1)

            if key not in dup_groups:
                dup_groups[key] = [key[1]]
            dup_groups[key].append(i)

        # Tạo chuỗi mô tả dup_vertices
        dup_desc_parts = []
        for key, indices in dup_groups.items():
            dup_desc_parts.append("=".join(str(i) for i in indices))
        dup_desc = ", ".join(dup_desc_parts)

        # Lặp xóa endpoint cho đến khi không còn duplicate vertex
        max_iter = num_vertices // 2
        for _pass in range(max_iter):
            # Detect duplicate vertices trong coords hiện tại
            vs = (round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION))
            ve = (round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION))

            current_dups = []
            if len(coords) >= 3:
                for i in range(1, len(coords) - 1):
                    pt = (round(coords[i][0], COORD_PRECISION),
                          round(coords[i][1], COORD_PRECISION))
                    if pt == vs or pt == ve:
                        current_dups.append(i)

            if not current_dups:
                break  # Không còn duplicate → dừng

            # Luôn xóa ENDPOINT, giữ vertex trùng trung gian
            final_remove = set()
            for dup_i in current_dups:
                pt = (round(coords[dup_i][0], COORD_PRECISION),
                      round(coords[dup_i][1], COORD_PRECISION))
                if pt == vs:
                    final_remove.add(0)                  # xóa V[0] (start)
                else:
                    final_remove.add(len(coords) - 1)    # xóa V[-1] (end)

            for i in sorted(final_remove, reverse=True):
                coords.pop(i)

            # Dọn spike do retrace (góc 0°-1°)
            changed = True
            while changed and len(coords) >= 3:
                changed = False
                for i in range(1, len(coords) - 1):
                    angle = _calc_angle_at_vertex(coords[i - 1], coords[i], coords[i + 1])
                    if 0 <= angle <= SPIKE_ANGLE_MAX:
                        coords.pop(i)
                        changed = True
                        break

        # Kiểm tra sau khi xóa còn >= 2 đỉnh không
        if len(coords) < 2:
            row_id = str(gdf.at[idx, id_col]) if id_col else str(idx)
            results.append({
                "ID": row_id,
                "No-Vertices": num_vertices,
                "Dup-Vertices": dup_desc,
                "OldLen": old_len,
                "NewLen": 0.0,
                "Status": "SKIPPED",
            })
            continue

        # Cập nhật geometry
        new_geom = LineString(coords)
        gdf.at[idx, "geometry"] = new_geom
        new_len = recalc_length(gdf, idx)

        row_id = str(gdf.at[idx, id_col]) if id_col else str(idx)
        results.append({
            "ID": row_id,
            "No-Vertices": num_vertices,
            "Dup-Vertices": dup_desc,
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
    detections = detect_duplicate_vertices(gdf)

    # Sửa lỗi
    results = correct_duplicate_vertices(gdf, detections)

    # Lưu file
    save_shp(gdf, OUTPUT_PATH)

    # ── Báo cáo ──
    fixed_count = sum(1 for r in results if r["Status"] == "FIXED")
    print(f"Phát hiện: {len(results)} trường hợp Duplicate Vertices ✅")
    print(f"Sửa lỗi: {fixed_count}/{len(results)} trường hợp Duplicate Vertices ✅")
    print(f"{'STT':<5} | {'ID':<38} | {'No-Vertices':<12} | {'Dup-Vertices':<15} | {'OldLen':<12} | {'NewLen':<12} | Status")
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['ID']:<38} | {r['No-Vertices']:<12} | {r['Dup-Vertices']:<15} | {r['OldLen']:<12} | {r['NewLen']:<12} | {r['Status']}")


if __name__ == "__main__":
    main()
