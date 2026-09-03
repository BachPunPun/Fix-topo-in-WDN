import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import math
import pandas as pd
import geopandas as gpd

import setup_console
import networkx as nx
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection, MultiLineString
from shapely.ops import substring

from read_file import read_shp
from save_file import save_shp
from graph_builder import build_graph
from load_layers import load_layers
import config

# 🧮 Cấu hình tham số
COORD_PRECISION = 6
OUTPUT_PATH = config.OUTPUT_DIR / f"{config.RUN_NAME}-{Path(__file__).stem}.shp"
ITERATIONS_MAX = 100
ANGEL_MIN = 1.0
ANGEL_MAX = 179.0
DIST_THRESHOLD = 0.1  # ngưỡng khoảng cách (tolerance) giữa 2 đoạn để coi là khe hở
MATERIAL_GROUPS = [
    {'PVC', 'uPVC', 'HDPE'},
    {'DI', 'CI', 'BTCT-TA-KNT'},
]


# 🛠️ Helper: tìm cột vật liệu
def _find_mat_col(gdf):
    norm = {c.lower().strip(): c for c in gdf.columns}
    for name in ('material', 'vatlieu', 'chatlieu'):
        if name in norm:
            return norm[name]
    return None


def _get_mat_group(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return -1
    s = str(val).strip()
    for i, grp in enumerate(MATERIAL_GROUPS):
        if s in grp:
            return i
    return -1


# 🛠️ Helper
def get_angle(pt, coords):
    for i, c in enumerate(coords):
        if math.isclose(pt.x, c[0], abs_tol=1e-5) and math.isclose(pt.y, c[1], abs_tol=1e-5):
            if i > 0 and i < len(coords) - 1:
                p1 = coords[i-1]
                p2 = c
                p3 = coords[i+1]
                v1 = (p1[0]-p2[0], p1[1]-p2[1])
                v2 = (p3[0]-p2[0], p3[1]-p2[1])
                mag1 = math.hypot(*v1)
                mag2 = math.hypot(*v2)
                if mag1 == 0 or mag2 == 0:
                    return 180.0
                val = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1*mag2)
                val = max(-1.0, min(1.0, val))
                return math.degrees(math.acos(val))
            else:
                return 180.0
    return 180.0

# ── Bước 1: Phát hiện lỗi ─────────────────────────────────────
def detect_sliver_gap_lines(gdf, dist_threshold: float = DIST_THRESHOLD):
    gdf_valid = gdf[gdf.geometry.notna() & (gdf.geometry.type == 'LineString')].copy()

    id_col = None
    for col in ['ID', 'id', 'OBJECTID', 'objectid', 'FID']:
        if col in gdf_valid.columns:
            id_col = col
            break

    mat_col = _find_mat_col(gdf_valid)
    if mat_col:
        gdf_valid['_mat_group'] = gdf_valid[mat_col].apply(_get_mat_group)
    else:
        gdf_valid['_mat_group'] = -1

    errors = []

    for idx1, row1 in gdf_valid.iterrows():
        geom1 = row1.geometry
        mg1 = row1['_mat_group']
        possible_matches_index = list(gdf_valid.sindex.intersection(geom1.bounds))
        possible_matches = gdf_valid.iloc[possible_matches_index]

        for idx2, row2 in possible_matches.iterrows():
            if idx1 >= idx2:
                continue
            mg2 = row2['_mat_group']
            if mg1 == -1 or mg2 == -1 or mg1 != mg2:
                continue
            geom2 = row2.geometry

            inter = geom1.intersection(geom2)

            pts = []
            if isinstance(inter, Point):
                continue
            elif isinstance(inter, MultiPoint):
                pts = list(inter.geoms)
            elif isinstance(inter, (LineString, MultiLineString)):
                continue
            elif isinstance(inter, GeometryCollection):
                for g in inter.geoms:
                    if isinstance(g, Point):
                        pts.append(g)

            if len(pts) < 2:
                continue

            coords1 = list(geom1.coords)
            coords2 = list(geom2.coords)

            def is_vertex(pt, coords):
                for c in coords:
                    if math.isclose(pt.x, c[0], abs_tol=1e-5) and math.isclose(pt.y, c[1], abs_tol=1e-5):
                        return True
                return False

            valid_pts = []
            for p in pts:
                if is_vertex(p, coords1) and is_vertex(p, coords2):
                    valid_pts.append(p)

            if len(valid_pts) < 2:
                continue

            valid_pts_final = []
            for p in valid_pts:
                angle1 = get_angle(p, coords1)
                angle2 = get_angle(p, coords2)
                if angle1 <= ANGEL_MIN or angle2 <= ANGEL_MIN:
                    continue
                valid_pts_final.append(p)

            if len(valid_pts_final) < 2:
                continue

            valid_pts_final.sort(key=lambda p: geom1.project(p))

            for i in range(len(valid_pts_final) - 1):
                p1 = valid_pts_final[i]
                p2 = valid_pts_final[i+1]

                d1_1 = geom1.project(p1)
                d1_2 = geom1.project(p2)
                d2_1 = geom2.project(p1)
                d2_2 = geom2.project(p2)

                seg1 = substring(geom1, min(d1_1, d1_2), max(d1_1, d1_2))
                seg2 = substring(geom2, min(d2_1, d2_2), max(d2_1, d2_2))

                if seg1.equals(seg2):
                    continue

                mid1 = seg1.interpolate(seg1.length / 2)
                mid2 = seg2.interpolate(seg2.length / 2)

                dist = mid1.distance(mid2)
                if 0 < dist <= dist_threshold:
                    is_geom1_short = geom1.length < geom2.length
                    short_idx = idx1 if is_geom1_short else idx2
                    long_idx = idx2 if is_geom1_short else idx1
                    short_geom = geom1 if is_geom1_short else geom2

                    errors.append({
                        'short_idx': short_idx,
                        'long_idx': long_idx,
                        'p1': p1,
                        'p2': p2,
                        'dist': dist,
                        'no_intersect': len(valid_pts_final),
                        'id_short': str(gdf_valid.loc[short_idx, id_col]) if id_col else str(short_idx),
                        'id_long': str(gdf_valid.loc[long_idx, id_col]) if id_col else str(long_idx),
                        'old_len': short_geom.length
                    })

    return errors


# ── Bước 2: Sửa lỗi ──────────────────────────────────────────
def correct_sliver_gap_lines(gdf, errors):
    gdf_valid = gdf[gdf.geometry.notna() & (gdf.geometry.type == 'LineString')].copy()

    new_rows = []
    for err in errors:
        short_idx = err['short_idx']

        if short_idx not in gdf_valid.index:
            err['status'] = 'SKIPPED'
            err['new_len'] = 0.0
            continue

        current_geom = gdf_valid.loc[short_idx, 'geometry']
        if current_geom is None or current_geom.is_empty:
            err['status'] = 'SKIPPED'
            err['new_len'] = 0.0
            continue

        coords = list(current_geom.coords)
        p1 = err['p1']
        p2 = err['p2']

        idxA = -1
        idxB = -1
        for i, c in enumerate(coords):
            if math.isclose(p1.x, c[0], abs_tol=1e-5) and math.isclose(p1.y, c[1], abs_tol=1e-5):
                idxA = i
            if math.isclose(p2.x, c[0], abs_tol=1e-5) and math.isclose(p2.y, c[1], abs_tol=1e-5):
                idxB = i

        if idxA == -1 or idxB == -1 or idxA == idxB:
            err['status'] = 'SKIPPED'
            err['new_len'] = current_geom.length
            continue

        idx_start = min(idxA, idxB)
        idx_end = max(idxA, idxB)

        if len(coords) <= 3:
            gdf_valid = gdf_valid.drop(index=short_idx)
            err['new_len'] = 0.0
            err['status'] = 'FIXED-DELETE'
        else:
            remnants = []
            if idx_start > 0:
                remnants.append(LineString(coords[0:idx_start+1]))
            if idx_end < len(coords) - 1:
                remnants.append(LineString(coords[idx_end:]))

            if not remnants:
                gdf_valid = gdf_valid.drop(index=short_idx)
                err['new_len'] = 0.0
                err['status'] = 'FIXED-DELETE'
            else:
                gdf_valid.loc[short_idx, 'geometry'] = remnants[0]
                for rem in remnants[1:]:
                    new_row = gdf_valid.loc[short_idx].copy()
                    new_row['geometry'] = rem
                    new_rows.append(new_row)

                err['new_len'] = sum(r.length for r in remnants)
                err['status'] = 'FIXED-SPLIT'

    if new_rows:
        new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf_valid.crs)
        gdf_valid = pd.concat([gdf_valid, new_gdf], ignore_index=True)

    return errors, gdf_valid


# ── Main ──────────────────────────────────────────────────────
def main():
    gdf, _ = load_layers()

    # Phát hiện lỗi
    errors = detect_sliver_gap_lines(gdf)

    # Sửa lỗi
    errors, gdf_result = correct_sliver_gap_lines(gdf, errors)

    # Lưu file
    save_shp(gdf_result, OUTPUT_PATH)

    # Báo cáo
    fixed_cnt = sum(1 for e in errors if e.get('status', '').startswith('FIXED'))

    print(f"Phát hiện: {len(errors)} trường hợp Sliver Gap Lines ✅")
    print(f"Sửa lỗi: {fixed_cnt}/{len(errors)} trường hợp Sliver Gap Lines ✅")
    if len(errors) > 0:
        print("STT | ID-Short | OldLen-Short | NewLen-Short | ID-Long | No-Intersect | Mid-Point-Dist | Status")
        for i, err in enumerate(errors, 1):
            print(f"{i} | {err['id_short']} | {err['old_len']:.2f} | {err['new_len']:.2f} | {err['id_long']} | {err['no_intersect']} | {err['dist']:.2f} | {err['status']}")

if __name__ == '__main__':
    main()
