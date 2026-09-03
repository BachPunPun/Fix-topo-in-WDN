# -*- coding: utf-8 -*-
import sys, os
# function/ nằm cạnh scripts/ và config.py (ở thư mục gốc dự án).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))  # helpers dùng chung
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))             # config.py ở thư mục gốc
sys.path.insert(0, os.path.dirname(__file__))                                 # các module cạnh nhau
import setup_console

import math
import importlib
import networkx as nx
import geopandas as gpd
import pandas as pd
from pathlib import Path
from collections import defaultdict
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection, MultiLineString
from shapely.ops import substring

from read_file import read_shp
from save_file import save_shp
from graph_builder import build_graph
from load_layers import load_layers, split_layers
import config

# ── Cấu hình tham số (đường dẫn data khai báo tập trung tại config.py) ──
COORD_PRECISION = config.COORD_PRECISION
OUTPUT_PATH = config.OUTPUT_DIR

# ── Import các module sửa lỗi ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
mod_multi      = importlib.import_module("multi-linestring")
mod_dupline    = importlib.import_module("duplicate-lines")
mod_dupvert    = importlib.import_module("duplicate-vertices")
mod_spike      = importlib.import_module("spike-lines")
mod_overlap    = importlib.import_module("overlap-lines")
mod_sliver     = importlib.import_module("sliver-gap-lines")
mod_crossing   = importlib.import_module("crossing")
mod_tjunction  = importlib.import_module("t-junction")
mod_overshoot  = importlib.import_module("overshoot")
mod_undershoot = importlib.import_module("undershoot")
mod_dangling   = importlib.import_module("dangling-lines")


# ══════════════════════════════════════════════════════════════
# Hàm hỗ trợ
# ══════════════════════════════════════════════════════════════
def _calc_network_stats(gdf: gpd.GeoDataFrame) -> dict:
    G = build_graph(gdf)
    edges = G.number_of_edges()

    cc_list = sorted(nx.connected_components(G), key=len, reverse=True)
    cc = len(cc_list)

    if cc_list:
        sub_largest = G.subgraph(cc_list[0])
        largest_edges = sub_largest.number_of_edges()
        largest_len = sum(d.get('length', 0.0) for _, _, d in sub_largest.edges(data=True))
    else:
        largest_edges = 0
        largest_len = 0.0

    total_len = float(gdf.geometry.length.sum()) if not gdf.empty else 0.0
    len_ratio = (largest_len / total_len * 100) if total_len > 0 else 0.0

    return {
        "Edges": edges,
        "LargestEdges": largest_edges,
        "CC": cc,
        "TotalLen": total_len,
        "LargestLen": largest_len,
        "LenRatio": len_ratio,
    }


def _get_cc_detail(gdf: gpd.GeoDataFrame) -> list[tuple]:
    G = build_graph(gdf)
    id_col = _find_id_col(gdf)
    components = []
    for comp_nodes in nx.connected_components(G):
        sub = G.subgraph(comp_nodes)
        edge_ids = []
        for u, v, d in sub.edges(data=True):
            idx = d.get("idx")
            if idx is not None and id_col:
                edge_ids.append(str(gdf.at[idx, id_col]) if idx in gdf.index else str(idx))
            elif idx is not None:
                edge_ids.append(str(idx))
        components.append((sub.number_of_edges(), sorted(edge_ids)))
    components.sort(key=lambda x: -x[0])
    return components


def _find_id_col(gdf: gpd.GeoDataFrame) -> str:
    """Ưu tiên GlobalID, sau đó OBJECTID, FID, ID."""
    norm = {str(c).lower().strip(): c for c in gdf.columns}
    for name in ("globalid", "objectid", "fid", "id"):
        if name in norm:
            return norm[name]
    return ""


def _get_id_info(gdf: gpd.GeoDataFrame, idx) -> dict:
    """Trả về dict {'GlobalID': ..., 'OBJECTID': ..., 'FID': ...} cho feature tại index `idx`."""
    ids = {"GlobalID": "", "OBJECTID": "", "FID": ""}
    if idx not in gdf.index:
        return ids
    row = gdf.loc[idx]
    norm = {str(c).lower().strip(): c for c in gdf.columns}

    for key, aliases in [
        ("GlobalID", ("globalid", "global_id", "guid")),
        ("OBJECTID", ("objectid", "object_id", "oid")),
        ("FID", ("fid", "featureid", "feature_id")),
    ]:
        for alias in aliases:
            if alias in norm:
                val = row[norm[alias]]
                if pd.notna(val) and str(val).strip() != "":
                    ids[key] = str(val).strip()
                    break

    if not any(ids.values()) and "id" in norm:
        val = row[norm["id"]]
        if pd.notna(val) and str(val).strip() != "":
            ids["GlobalID"] = str(val).strip()

    return ids


def _get_id_info_iloc(gdf: gpd.GeoDataFrame, pos: int) -> dict:
    """Như _get_id_info nhưng theo vị trí (iloc)."""
    if 0 <= pos < len(gdf):
        return _get_id_info(gdf, gdf.index[pos])
    return {"GlobalID": "", "OBJECTID": "", "FID": ""}


def _gid(gdf, idx, id_col=None):
    """Dictionary chứa các cột ID (GlobalID, OBJECTID, FID) của feature tại index idx."""
    return _get_id_info(gdf, idx)


def _gid_iloc(gdf, pos, id_col=None):
    """Như _gid nhưng theo VỊ TRÍ (iloc)."""
    return _get_id_info_iloc(gdf, pos)


ERROR_COLUMNS_ORDER = [
    ("MultiLS",    "ERR_MULT",  "Lỗi MultiLineString"),
    ("DupLines",   "ERR_DUPL",  "Lỗi ống trùng (DupLines)"),
    ("DupVtx",     "ERR_DUPV",  "Lỗi đỉnh trùng (DupVtx)"),
    ("Spike",      "ERR_SPIK",  "Lỗi ống góc nhọn (Spike)"),
    ("Overlap",    "ERR_OVER",  "Lỗi ống chồng lấp (Overlap)"),
    ("Sliver",     "ERR_SLIV",  "Lỗi ống có khe hở (Sliver)"),
    ("Crossing",   "ERR_CROS",  "Lỗi giao cắt chữ thập (Crossing)"),
    ("T-Junction", "ERR_TJUN",  "Lỗi giao cắt chữ tê (T-Junction)"),
    ("Overshoot",  "ERR_OVSH",  "Lỗi ống vượt quá (Overshoot)"),
    ("Undershoot", "ERR_UNDR",  "Lỗi ống chưa tới (Undershoot)"),
    ("Dangling",   "ERR_DNGL",  "Lỗi ống tự do (Dangling)"),
]


def _error_records_to_gdf(error_records, src_crs):
    """Gôm error_records thành GeoDataFrame theo cấu trúc:
    | GlobalID | OBJECTID | FID | 11 loại lỗi (0/1) | geometry |
    Mỗi ID/feature là duy nhất, không bị trùng lặp ID.
    Tên cột thuộc tính <= 10 ký tự ASCII để tương thích chuẩn ESRI Shapefile / DBF.
    """
    all_cols = ["GlobalID", "OBJECTID", "FID"] + [code for _, code, _ in ERROR_COLUMNS_ORDER]
    if not error_records:
        return gpd.GeoDataFrame(columns=all_cols + ["geometry"], crs=src_crs)

    grouped = {}

    for r in error_records:
        if not r or r[0] is None or r[0].is_empty:
            continue
        geom = r[0]
        err_key = r[1]
        id_data = r[2] if len(r) > 2 else ""

        if isinstance(id_data, dict):
            gid = str(id_data.get("GlobalID", "") or "").strip()
            oid = str(id_data.get("OBJECTID", "") or "").strip()
            fid = str(id_data.get("FID", "") or "").strip()
        elif isinstance(id_data, str):
            gid = id_data.strip()
            oid = ""
            fid = ""
        else:
            gid = str(id_data or "").strip()
            oid = ""
            fid = ""

        if gid or oid or fid:
            group_key = (gid, oid, fid)
        else:
            group_key = ("geom", geom.wkt)

        if group_key not in grouped:
            grouped[group_key] = {
                "GlobalID": gid,
                "OBJECTID": oid,
                "FID": fid,
                "geometry": geom,
                "errors": set(),
            }

        grouped[group_key]["errors"].add(err_key)

    rows = []
    for feat in grouped.values():
        row_dict = {
            "GlobalID": feat["GlobalID"],
            "OBJECTID": feat["OBJECTID"],
            "FID": feat["FID"],
        }
        for ek, col_code, _ in ERROR_COLUMNS_ORDER:
            row_dict[col_code] = 1 if ek in feat["errors"] else 0

        row_dict["geometry"] = feat["geometry"]
        rows.append(row_dict)

    if not rows:
        return gpd.GeoDataFrame(columns=all_cols + ["geometry"], crs=src_crs)

    return gpd.GeoDataFrame(rows, crs=src_crs)


# ══════════════════════════════════════════════════════════════
# Các hàm xử lý từng bước
# Mỗi hàm nhận gdf, trả về (gdf, detect_cnt, fixed_cnt, skip_cnt, error_list)
# ══════════════════════════════════════════════════════════════

def _step_multi_linestring(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    error_indices = mod_multi.detect_multi_linestring(gdf)
    for idx in error_indices:
        if idx in gdf.index:
            error_list.append((gdf.loc[idx, 'geometry'], "MultiLS", _gid(gdf, idx, id_col)))
    results = mod_multi.correct_multi_linestring(gdf, error_indices)
    detect_cnt = len(results)
    fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_duplicate_lines(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    dup_groups = mod_dupline.detect_duplicate_lines(gdf)
    for group in dup_groups:
        for idx in group[1:]:
            if idx in gdf.index:
                error_list.append((gdf.loc[idx, 'geometry'], "DupLines", _gid(gdf, idx, id_col)))
    results = mod_dupline.correct_duplicate_lines(gdf, dup_groups)
    detect_cnt = len(results)
    fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_duplicate_vertices(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    detections = mod_dupvert.detect_duplicate_vertices(gdf)
    for det in detections:
        idx = det["idx"]
        if idx in gdf.index:
            error_list.append((gdf.loc[idx, 'geometry'], "DupVtx", _gid(gdf, idx, id_col)))
    results = mod_dupvert.correct_duplicate_vertices(gdf, detections)
    detect_cnt = len(results)
    fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_spike_lines(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    detections = mod_spike.detect_spike_lines(gdf)
    for det in detections:
        idx = det["idx"]
        if idx in gdf.index:
            error_list.append((gdf.loc[idx, 'geometry'], "Spike", _gid(gdf, idx, id_col)))
    results = mod_spike.correct_spike_lines(gdf, detections)
    detect_cnt = len(results)
    fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_overlap_lines(gdf, tolerance=None):
    error_list = []
    all_results = []
    for _ in range(mod_overlap.ITERATIONS_MAX):
        id_col = _find_id_col(gdf)
        detections = mod_overlap.detect_overlap_lines(gdf)
        if not detections:
            break
        for det in detections:
            idx_short = det["idx_short"]
            if idx_short in gdf.index:
                error_list.append((gdf.loc[idx_short, 'geometry'], "Overlap", _gid(gdf, idx_short, id_col)))
        results = mod_overlap.correct_overlap_lines(gdf, detections)
        all_results.extend(results)
    detect_cnt = len(all_results)
    fixed_cnt = sum(1 for r in all_results if r["Status"] in ("FIXED", "DELETE", "CUT"))
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_sliver_gap_lines(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    errors = mod_sliver.detect_sliver_gap_lines(gdf, dist_threshold=tolerance) if tolerance is not None else mod_sliver.detect_sliver_gap_lines(gdf)
    for err in errors:
        short_idx = err['short_idx']
        if short_idx in gdf.index:
            error_list.append((gdf.loc[short_idx, 'geometry'], "Sliver", _gid(gdf, short_idx, id_col)))
    errors, gdf = mod_sliver.correct_sliver_gap_lines(gdf, errors)
    detect_cnt = len(errors)
    fixed_cnt = sum(1 for e in errors if e.get('status', '').startswith('FIXED'))
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_crossing(gdf, tolerance=None):
    error_list = []
    all_results = []
    for _ in range(mod_crossing.ITERATIONS_MAX):
        id_col = _find_id_col(gdf)
        detections = mod_crossing.detect_crossing(gdf, dist_threshold=tolerance) if tolerance is not None else mod_crossing.detect_crossing(gdf)
        if not detections:
            break
        for det in detections:
            idx = det["idx1"]
            if idx in gdf.index:
                error_list.append((gdf.loc[idx, 'geometry'], "Crossing", _gid(gdf, idx, id_col)))
        results = mod_crossing.correct_crossing(gdf, detections)
        all_results.extend(results)
    detect_cnt = len(all_results)
    fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_t_junction(gdf, tolerance=None):
    error_list = []
    tjunction_errors, gdf_lines = mod_tjunction.detect_t_junction(gdf, dist_min=tolerance) if tolerance is not None else mod_tjunction.detect_t_junction(gdf)
    if tjunction_errors:
        id_col_l = _find_id_col(gdf_lines)
        for tgt_idx in tjunction_errors:
            if tgt_idx < len(gdf_lines):
                error_list.append((gdf_lines.iloc[tgt_idx].geometry, "T-Junction",
                                   _gid_iloc(gdf_lines, tgt_idx, id_col_l)))
        result_gdf, results, detected, fixed = mod_tjunction.correct_t_junction(gdf_lines, tjunction_errors)
        gdf = result_gdf.copy()
        detect_cnt = detected
        fixed_cnt = fixed
        skip_cnt = detect_cnt - fixed_cnt
    else:
        detect_cnt = 0
        fixed_cnt = 0
        skip_cnt = 0
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_overshoot(gdf, tolerance=None):
    error_list = []
    all_results = []
    skipped_ids = set()  # Track bằng ID string
    id_col = _find_id_col(gdf)
    for _ in range(mod_overshoot.ITERATIONS_MAX):
        detections = mod_overshoot.detect_overshoot(gdf, dist_threshold=tolerance) if tolerance is not None else mod_overshoot.detect_overshoot(gdf)
        if skipped_ids:
            detections = [d for d in detections
                          if str(gdf.at[d["src_idx"], id_col] if id_col else d["src_idx"]) not in skipped_ids]
        if not detections:
            break
        src_geoms = {}
        src_gids = {}
        for det in detections:
            si = det["src_idx"]
            if si not in src_geoms and si in gdf.index:
                src_geoms[si] = gdf.loc[si, 'geometry']
                src_gids[si] = _gid(gdf, si, id_col)
        results, gdf = mod_overshoot.correct_overshoot(gdf, detections, dist_threshold=tolerance) if tolerance is not None else mod_overshoot.correct_overshoot(gdf, detections)
        src_geom_list = list(src_geoms.values())
        src_gid_list = list(src_gids.values())
        for i, r in enumerate(results):
            geom = src_geom_list[i] if i < len(src_geom_list) else src_geom_list[-1]
            gid = src_gid_list[i] if i < len(src_gid_list) else (src_gid_list[-1] if src_gid_list else "")
            error_list.append((geom, "Overshoot", gid))
            if r["Status"] == "SKIPPED":
                skipped_ids.add(r["ID-Source"])
        all_results.extend(results)
    detect_cnt = len(all_results)
    fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_undershoot(gdf, tolerance=None):
    error_list = []
    all_results = []
    skipped_pairs = set()  # frozenset({src_id, tgt_id}) — mutual pair tính 1 lần
    id_col = _find_id_col(gdf)
    for _ in range(mod_undershoot.ITERATIONS_MAX):
        detections = mod_undershoot.detect_undershoot(gdf, dist_threshold=tolerance) if tolerance is not None else mod_undershoot.detect_undershoot(gdf)
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
        src_geoms = {}
        src_gids = {}
        for det in detections:
            si = det["src_idx"]
            if si not in src_geoms and si in gdf.index:
                src_geoms[si] = gdf.loc[si, 'geometry']
                src_gids[si] = _gid(gdf, si, id_col)
        results, gdf = mod_undershoot.correct_undershoot(gdf, detections, dist_threshold=tolerance) if tolerance is not None else mod_undershoot.correct_undershoot(gdf, detections)
        src_geom_list = list(src_geoms.values())
        src_gid_list = list(src_gids.values())
        for i, r in enumerate(results):
            geom = src_geom_list[i] if i < len(src_geom_list) else src_geom_list[-1]
            gid = src_gid_list[i] if i < len(src_gid_list) else (src_gid_list[-1] if src_gid_list else "")
            error_list.append((geom, "Undershoot", gid))
            if r["Status"] == "SKIPPED":
                for tgt in r["ID-Target"].split(","):
                    skipped_pairs.add(frozenset({r["ID-Source"], tgt.strip()}))
        all_results.extend(results)
    detect_cnt = len(all_results)
    fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


def _step_dangling(gdf, tolerance=None):
    error_list = []
    id_col = _find_id_col(gdf)
    detections = mod_dangling.detect_dangling(gdf, dist_threshold=tolerance) if tolerance is not None else mod_dangling.detect_dangling(gdf)
    if detections:
        for det in detections:
            si = det["src_idx"]
            if si in gdf.index:
                error_list.append((gdf.loc[si, 'geometry'], "Dangling", _gid(gdf, si, id_col)))
        results, gdf = mod_dangling.correct_dangling(gdf, detections)
    else:
        results = []
    detect_cnt = len(results)
    fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")
    skip_cnt = detect_cnt - fixed_cnt
    return gdf, detect_cnt, fixed_cnt, skip_cnt, error_list


# ══════════════════════════════════════════════════════════════
# CHỈ PHÁT HIỆN (detect-only) — KHÔNG sửa, KHÔNG đụng vào gdf.
# Mỗi hàm nhận gdf gốc, trả về error_list = [(geometry, err_key, global_id), ...].
# Với các loại chạy lặp (Overlap/Crossing/Overshoot/Undershoot) chỉ dò MỘT lượt
# trên dữ liệu gốc (vòng lặp nhiều lượt là để sửa dây chuyền, không cần khi chỉ dò).
# ══════════════════════════════════════════════════════════════
def _detect_multi_linestring(gdf):
    out = []
    id_col = _find_id_col(gdf)
    for idx in mod_multi.detect_multi_linestring(gdf):
        if idx in gdf.index:
            out.append((gdf.loc[idx, 'geometry'], "MultiLS", _gid(gdf, idx, id_col)))
    return out


def _detect_duplicate_lines(gdf):
    out = []
    id_col = _find_id_col(gdf)
    for group in mod_dupline.detect_duplicate_lines(gdf):
        for idx in group[1:]:
            if idx in gdf.index:
                out.append((gdf.loc[idx, 'geometry'], "DupLines", _gid(gdf, idx, id_col)))
    return out


def _detect_duplicate_vertices(gdf):
    out = []
    id_col = _find_id_col(gdf)
    for det in mod_dupvert.detect_duplicate_vertices(gdf):
        idx = det["idx"]
        if idx in gdf.index:
            out.append((gdf.loc[idx, 'geometry'], "DupVtx", _gid(gdf, idx, id_col)))
    return out


def _detect_spike_lines(gdf):
    out = []
    id_col = _find_id_col(gdf)
    for det in mod_spike.detect_spike_lines(gdf):
        idx = det["idx"]
        if idx in gdf.index:
            out.append((gdf.loc[idx, 'geometry'], "Spike", _gid(gdf, idx, id_col)))
    return out


def _detect_overlap_lines(gdf):
    out = []
    id_col = _find_id_col(gdf)
    for det in mod_overlap.detect_overlap_lines(gdf):
        idx_short = det["idx_short"]
        if idx_short in gdf.index:
            out.append((gdf.loc[idx_short, 'geometry'], "Overlap", _gid(gdf, idx_short, id_col)))
    return out


def _detect_sliver_gap_lines(gdf, tolerance=None):
    out = []
    id_col = _find_id_col(gdf)
    errors = mod_sliver.detect_sliver_gap_lines(gdf, dist_threshold=tolerance) if tolerance is not None else mod_sliver.detect_sliver_gap_lines(gdf)
    for err in errors:
        short_idx = err['short_idx']
        if short_idx in gdf.index:
            out.append((gdf.loc[short_idx, 'geometry'], "Sliver", _gid(gdf, short_idx, id_col)))
    return out


def _detect_crossing(gdf, tolerance=None):
    out = []
    id_col = _find_id_col(gdf)
    detections = mod_crossing.detect_crossing(gdf, dist_threshold=tolerance) if tolerance is not None else mod_crossing.detect_crossing(gdf)
    for det in detections:
        idx = det["idx1"]
        if idx in gdf.index:
            out.append((gdf.loc[idx, 'geometry'], "Crossing", _gid(gdf, idx, id_col)))
    return out


def _detect_t_junction(gdf, tolerance=None):
    out = []
    tjunction_errors, gdf_lines = mod_tjunction.detect_t_junction(gdf, dist_min=tolerance) if tolerance is not None else mod_tjunction.detect_t_junction(gdf)
    id_col_l = _find_id_col(gdf_lines)
    for tgt_idx in (tjunction_errors or []):
        if tgt_idx < len(gdf_lines):
            out.append((gdf_lines.iloc[tgt_idx].geometry, "T-Junction",
                        _gid_iloc(gdf_lines, tgt_idx, id_col_l)))
    return out


def _detect_overshoot(gdf, tolerance=None):
    out = []
    id_col = _find_id_col(gdf)
    detections = mod_overshoot.detect_overshoot(gdf, dist_threshold=tolerance) if tolerance is not None else mod_overshoot.detect_overshoot(gdf)
    for det in detections:
        si = det["src_idx"]
        if si in gdf.index:
            out.append((gdf.loc[si, 'geometry'], "Overshoot", _gid(gdf, si, id_col)))
    return out


def _detect_undershoot(gdf, tolerance=None):
    out = []
    id_col = _find_id_col(gdf)
    detections = mod_undershoot.detect_undershoot(gdf, dist_threshold=tolerance) if tolerance is not None else mod_undershoot.detect_undershoot(gdf)
    for det in detections:
        si = det["src_idx"]
        if si in gdf.index:
            out.append((gdf.loc[si, 'geometry'], "Undershoot", _gid(gdf, si, id_col)))
    return out


def _detect_dangling(gdf, tolerance=None):
    out = []
    id_col = _find_id_col(gdf)
    detections = mod_dangling.detect_dangling(gdf, dist_threshold=tolerance) if tolerance is not None else mod_dangling.detect_dangling(gdf)
    for det in detections:
        si = det["src_idx"]
        if si in gdf.index:
            out.append((gdf.loc[si, 'geometry'], "Dangling", _gid(gdf, si, id_col)))
    return out


# Tra hàm PHÁT HIỆN theo tên bước (khớp tên trong PIPELINE bên dưới).
DETECT_FUNCS = {
    "Multi LineString":   _detect_multi_linestring,
    "Duplicate Lines":    _detect_duplicate_lines,
    "Duplicate Vertices": _detect_duplicate_vertices,
    "Spike Lines":        _detect_spike_lines,
    "Overlap Lines":      _detect_overlap_lines,
    "Sliver Gap Lines":   _detect_sliver_gap_lines,
    "Crossing":           _detect_crossing,
    "T-Junction":         _detect_t_junction,
    "Overshoot":          _detect_overshoot,
    "Undershoot":         _detect_undershoot,
    "Dangling Lines":     _detect_dangling,
}


# ══════════════════════════════════════════════════════════════
# ⚠️ CẤU HÌNH THỨ TỰ PIPELINE — Thay đổi thứ tự tại đây
# ══════════════════════════════════════════════════════════════
PIPELINE = [
    # (Tên hiển thị,       Tên file output,       Hàm xử lý)
    ("Multi LineString",   "multilinestring",     _step_multi_linestring),
    ("Duplicate Lines",    "duplicate-lines",     _step_duplicate_lines),
    ("Duplicate Vertices", "duplicate-vertices",  _step_duplicate_vertices),
    ("Spike Lines",        "spike-lines",         _step_spike_lines),
    ("Overlap Lines",      "overlap-lines",       _step_overlap_lines),
    ("Sliver Gap Lines",   "sliver-gap-lines",    _step_sliver_gap_lines),
    ("Crossing",           "crossing",            _step_crossing),
    ("T-Junction",         "t-junction",          _step_t_junction),
    ("Overshoot",          "overshoot",           _step_overshoot),
    ("Undershoot",         "undershoot",          _step_undershoot),
    ("Dangling Lines",     "dangling-lines",      _step_dangling),
]


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    run = config.RUN_NAME
    result_dir = OUTPUT_PATH
    result_dir.mkdir(parents=True, exist_ok=True)

    # Nạp & gộp các lớp ống (truyền dẫn + phân phối) để phân tích topology toàn mạng.
    # gdf có thêm cột config.SRC_FIELD đánh dấu lớp nguồn; layer_meta để tách lại khi xuất.
    gdf, layer_meta = load_layers()
    print(f"→ Đã nạp {len(gdf)} ống từ {len(layer_meta)} lớp: {', '.join(layer_meta)}")

    topo_report = []
    network_report = []
    error_records = []

    network_report.append(("Bắt đầu", _calc_network_stats(gdf)))

    # ── Chạy pipeline ──────────────────────────────────────────
    # Không lưu file trung gian từng bước; chỉ tính thống kê để báo cáo.
    for step_name, _file_suffix, step_fn in PIPELINE:
        gdf, detect_cnt, fixed_cnt, skip_cnt, step_errors = step_fn(gdf)
        error_records.extend(step_errors)
        topo_report.append((step_name, detect_cnt, fixed_cnt, skip_cnt))
        network_report.append((step_name, _calc_network_stats(gdf)))

    # ── Output cuối cùng — TÁCH RIÊNG từng lớp ─────────────────
    # (1) File gộp toàn mạng (kèm cột SRC_LAYER) — tiện kiểm tra/đối chiếu liên thông
    save_shp(gdf, result_dir / f"{run}-atopo.shp")
    # (2) File riêng cho từng lớp ống — đã bỏ cột đánh dấu nguồn
    layer_parts = split_layers(gdf, layer_meta)
    for lname, lgdf in layer_parts.items():
        save_shp(lgdf, result_dir / f"{run}-{lname}-atopo.shp")
        print(f"→ Saved {len(lgdf)} features to {run}-{lname}-atopo.shp")

    if error_records:
        gdf_error = _error_records_to_gdf(error_records, gdf.crs)
        save_shp(gdf_error, result_dir / f"{run}_error.shp")
        print(f"\n→ Saved {len(gdf_error)} error features to {run}_error.shp")
        for ek, col_code, col_desc in ERROR_COLUMNS_ORDER:
            if col_code in gdf_error.columns:
                cnt = int(gdf_error[col_code].sum())
                if cnt > 0:
                    print(f"   {col_desc}: {cnt}")
    else:
        print(f"\n→ Không phát hiện lỗi topology nào.")

    # ── Output bổ sung ② — {run}_disconnected.shp ─────────────
    G_final = build_graph(gdf)
    cc_list = sorted(nx.connected_components(G_final), key=len, reverse=True)

    if len(cc_list) > 1:
        small_cc_nodes = set()
        node_to_group = {}
        for grp_id, comp_nodes in enumerate(cc_list[1:], start=2):
            small_cc_nodes.update(comp_nodes)
            for nd in comp_nodes:
                node_to_group[nd] = grp_id

        disconnected_indices = []
        disconnected_groups = []
        for u, v, d in G_final.edges(data=True):
            if u in small_cc_nodes or v in small_cc_nodes:
                idx = d.get("idx")
                if idx is not None and idx in gdf.index:
                    disconnected_indices.append(idx)
                    grp = node_to_group.get(u) or node_to_group.get(v, 0)
                    disconnected_groups.append(grp)

        if disconnected_indices:
            gdf_disc = gdf.loc[disconnected_indices].copy()
            gdf_disc["cc_group"] = disconnected_groups
            gdf_disc = gdf_disc[~gdf_disc.index.duplicated(keep='first')]
            save_shp(gdf_disc, result_dir / f"{run}_disconnected.shp")
            print(f"→ Saved {len(gdf_disc)} disconnected features ({len(cc_list)-1} groups) to {run}_disconnected.shp")
        else:
            print(f"→ Không có thành phần liên thông rời rạc.")
    else:
        print(f"→ Mạng lưới chỉ có 1 thành phần liên thông, không có disconnected.")

    # ── Báo cáo ① — Lỗi topology ──────────────────────────────
    print("========================================================")
    print("1. Báo cáo lỗi topology")
    print("========================================================")
    print()
    print(f"| {'STT':<3} | {'Type':<18} | {'Detect':>6} | {'Correct':>7} | {'Skip':>4} |")
    print(f"| {'---':<3} | {'------------------':<18} | {'------':>6} | {'-------':>7} | {'----':>4} |")
    total_d, total_c, total_s = 0, 0, 0
    for i, (name, d, c, s) in enumerate(topo_report, 1):
        print(f"| {i:<3} | {name:<18} | {d:>6} | {c:>7} | {s:>4} |")
        total_d += d
        total_c += c
        total_s += s
    print(f"| {'':3} | {'Total':<18} | {total_d:>6} | {total_c:>7} | {total_s:>4} |")

    # ── Báo cáo ② — Sự thay đổi thành phần liên thông lớn ───────
    print()
    print("=" * 167)
    print("2. Sự thay đổi thành phần liên thông lớn")
    print("=" * 167)
    print(f"| {'STT':<3} | {'Bước':<18} | {'Tổng cạnh':>9} | {'Tổng cạnh trong':>19} | {'Thành phần liên thông':>21} | {'CC Δ%':>7} | {'Tổng chiều dài':>16} | {'Tổng chiều dài trong':>23} | {'Tỷ lệ chiều dài của':>23} |")
    print(f"| {'':<3} | {'':<18} | {'':>9} | {'thành phần lớn':>19} | {'':>21} | {'':>7} | {'':>16} | {'thành phần lớn':>23} | {'thành phần lớn':>23} |")
    print(f"| {'---':<3} | {'------------------':<18} | {'---------':>9} | {'-------------------':>19} | {'---------------------':>21} | {'-------':>7} | {'----------------':>16} | {'-----------------------':>23} | {'-----------------------':>23} |")
    cc_start = network_report[0][1]['CC']
    for i, (name, s) in enumerate(network_report, 1):
        if i == 1:
            cc_delta = "—"
        else:
            pct = (cc_start - s['CC']) / cc_start * 100 if cc_start else 0
            cc_delta = f"{pct:+.1f}%"
        print(f"| {i:<3} | {name:<18} | {s['Edges']:>9} | {s['LargestEdges']:>19} | {s['CC']:>21} | {cc_delta:>7} | {s['TotalLen']:>16.2f} | {s['LargestLen']:>23.2f} | {s['LenRatio']:>22.2f}% |")
    final = network_report[-1][1]
    pct_final = (cc_start - final['CC']) / cc_start * 100 if cc_start else 0
    cc_delta_final = f"{pct_final:+.1f}%"
    print(f"| {'':3} | {'Final':<18} | {final['Edges']:>9} | {final['LargestEdges']:>19} | {final['CC']:>21} | {cc_delta_final:>7} | {final['TotalLen']:>16.2f} | {final['LargestLen']:>23.2f} | {final['LenRatio']:>22.2f}% |")

    print()
    components = _get_cc_detail(gdf)
    print(f"Nhóm thành phần liên thông (sắp xếp giảm dần theo số edges):")
    if components:
        print(f"- Nhóm 1: {components[0][0]} edges")
        group_num = 2
        single_edge_ids = []
        for edges_count, ids in components[1:]:
            if edges_count >= 2:
                print(f"- Nhóm {group_num}: {edges_count} edges — {', '.join(ids)}")
                group_num += 1
            else:
                single_edge_ids.extend(ids)
        if single_edge_ids:
            num_single = sum(1 for ec, _ in components[1:] if ec == 1)
            print(f"- Nhóm 1 edges ({num_single}): {', '.join(single_edge_ids)}")


if __name__ == "__main__":
    main()
