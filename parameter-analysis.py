# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path

# Thêm đường dẫn thư mục scripts, function và thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "function"))
sys.path.insert(0, os.path.dirname(__file__))

import setup_console
import importlib
import pandas as pd
import geopandas as gpd

from load_layers import load_layers
import config

# Import các module sửa lỗi
mod_spike       = importlib.import_module("spike-lines")
mod_overlap     = importlib.import_module("overlap-lines")
mod_sliver      = importlib.import_module("sliver-gap-lines")
mod_crossing    = importlib.import_module("crossing")
mod_tjunction   = importlib.import_module("t-junction")
mod_overshoot   = importlib.import_module("overshoot")
mod_undershoot  = importlib.import_module("undershoot")
mod_dangling    = importlib.import_module("dangling-lines")

# ── Cấu hình thư mục và danh sách tham số phân tích ───────────
OUTPUT_DIR = config.OUTPUT_DIR / "parameter-analysis"

SPIKE_EXCEL      = OUTPUT_DIR / "spike.xlsx"
RATIO_EXCEL      = OUTPUT_DIR / "ratio.xlsx"
SLIVER_EXCEL     = OUTPUT_DIR / "tolerance-sliver-gap-lines.xlsx"
CROSSING_EXCEL   = OUTPUT_DIR / "tolerance-crossing.xlsx"
TJUNCTION_EXCEL  = OUTPUT_DIR / "tolerance-t-junction.xlsx"
OVERSHOOT_EXCEL  = OUTPUT_DIR / "tolerance-overshoot.xlsx"
UNDERSHOOT_EXCEL = OUTPUT_DIR / "tolerance-undershoot.xlsx"
DANGLING_EXCEL   = OUTPUT_DIR / "tolerance-dangling-lines.xlsx"

ANGLE_VALUES     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.25]
RATIO_VALUES     = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
TOLERANCE_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# TOLERANCE_VALUES = [0.001, 0.012, 0.023, 0.034, 0.045, 0.056, 0.067, 0.078, 0.089, 0.1]


# ══════════════════════════════════════════════════════════════
# 1. Phân tích tham số Spike Lines (angle_max)
# ══════════════════════════════════════════════════════════════
def run_spike_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[1/8] Bắt đầu phân tích tham số Spike Lines (angle_max): {ANGLE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(SPIKE_EXCEL, engine="openpyxl") as writer:
        for angle in ANGLE_VALUES:
            sheet_name = str(angle)

            # Sao chép dữ liệu độc lập cho mỗi giá trị tham số
            gdf = gdf_orig.copy(deep=True)

            detections = mod_spike.detect_spike_lines(gdf, angle_min=0, angle_max=angle)
            results = mod_spike.correct_spike_lines(gdf, detections, angle_min=0, angle_max=angle)

            detect_cnt = len(results)
            fixed_cnt = sum(1 for r in results if r["Status"] == "FIXED")

            if results:
                df = pd.DataFrame(results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID", "No-Vertices", "Angle", "No-Spike",
                    "Spike-at-Vertex", "OldLen", "NewLen", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Spike Lines vào: {SPIKE_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 2. Phân tích tham số Overlap Lines (ratio)
# ══════════════════════════════════════════════════════════════
def run_overlap_ratio_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[2/8] Bắt đầu phân tích tham số Overlap Lines (ratio): {RATIO_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(RATIO_EXCEL, engine="openpyxl") as writer:
        for ratio in RATIO_VALUES:
            sheet_name = str(ratio)

            gdf = gdf_orig.copy(deep=True)
            all_results = []

            for iteration in range(1, mod_overlap.ITERATIONS_MAX + 1):
                detections = mod_overlap.detect_overlap_lines(gdf, ratio_delete_threshold=ratio)
                if not detections:
                    break
                results = mod_overlap.correct_overlap_lines(gdf, detections, ratio_delete_threshold=ratio)
                all_results.extend(results)

            detect_cnt = len(all_results)
            fixed_cnt = sum(1 for r in all_results if r["Status"] in ("DELETE", "CUT", "FIXED"))

            if all_results:
                df = pd.DataFrame(all_results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = ["STT", "ID-Keep", "Len-Keep", "ID-Removed", "Len-Removed", "Ratio", "Status"]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Overlap Lines (ratio) vào: {RATIO_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 3. Phân tích tham số Sliver Gap Lines (tolerance)
# ══════════════════════════════════════════════════════════════
def run_sliver_gap_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[3/8] Bắt đầu phân tích tham số Sliver Gap Lines (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(SLIVER_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)

            errors = mod_sliver.detect_sliver_gap_lines(gdf, dist_threshold=tol)
            errors, gdf_result = mod_sliver.correct_sliver_gap_lines(gdf, errors)

            detect_cnt = len(errors)
            fixed_cnt = sum(1 for e in errors if e.get("status", "").startswith("FIXED"))

            results = []
            for err in errors:
                results.append({
                    "ID-Short": err["id_short"],
                    "OldLen-Short": round(err["old_len"], config.COORD_PRECISION),
                    "NewLen-Short": round(err["new_len"], config.COORD_PRECISION),
                    "ID-Long": err["id_long"],
                    "No-Intersect": err["no_intersect"],
                    "Mid-Point-Dist": round(err["dist"], config.COORD_PRECISION),
                    "Status": err.get("status", "SKIPPED"),
                })

            if results:
                df = pd.DataFrame(results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID-Short", "OldLen-Short", "NewLen-Short",
                    "ID-Long", "No-Intersect", "Mid-Point-Dist", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Sliver Gap Lines (tolerance) vào: {SLIVER_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 4. Phân tích tham số Crossing (tolerance)
# ══════════════════════════════════════════════════════════════
def run_crossing_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[4/8] Bắt đầu phân tích tham số Crossing (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(CROSSING_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)
            all_results = []

            for iteration in range(1, mod_crossing.ITERATIONS_MAX + 1):
                detections = mod_crossing.detect_crossing(gdf, dist_threshold=tol)
                if not detections:
                    break
                results = mod_crossing.correct_crossing(gdf, detections)
                all_results.extend(results)

            detect_cnt = len(all_results)
            fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")

            if all_results:
                df = pd.DataFrame(all_results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID-Source", "OldLen-Source", "NewLen-Source",
                    "ID-Target", "OldLen-Target", "NewLen-Target",
                    "No-Intersect", "Angle", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Crossing (tolerance) vào: {CROSSING_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 5. Phân tích tham số T-Junction (tolerance)
# ══════════════════════════════════════════════════════════════
def run_tjunction_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[5/8] Bắt đầu phân tích tham số T-Junction (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(TJUNCTION_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)

            tjunction_errors, gdf_lines = mod_tjunction.detect_t_junction(gdf, dist_min=tol)
            if tjunction_errors:
                result_gdf, results, detected, fixed = mod_tjunction.correct_t_junction(gdf_lines, tjunction_errors)
            else:
                results = []
                detected = 0
                fixed = 0

            detect_cnt = detected
            fixed_cnt = fixed

            if results:
                df = pd.DataFrame(results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID-Target", "OldLen-Target", "NewLen-Target",
                    "ID-Source", "Parts", "Angle", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả T-Junction (tolerance) vào: {TJUNCTION_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 6. Phân tích tham số Overshoot (tolerance)
# ══════════════════════════════════════════════════════════════
def run_overshoot_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[6/8] Bắt đầu phân tích tham số Overshoot (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(OVERSHOOT_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)
            all_results = []

            for iteration in range(1, mod_overshoot.ITERATIONS_MAX + 1):
                detections = mod_overshoot.detect_overshoot(gdf, dist_threshold=tol)
                if not detections:
                    break
                results, gdf = mod_overshoot.correct_overshoot(gdf, detections, dist_threshold=tol)
                all_results.extend(results)

            detect_cnt = len(all_results)
            fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")

            if all_results:
                df = pd.DataFrame(all_results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID-Source", "OldLen-Source", "NewLen-Source",
                    "ID-Target", "OldLen-Target", "NewLen-Target",
                    "Angle", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Overshoot (tolerance) vào: {OVERSHOOT_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 7. Phân tích tham số Undershoot (tolerance)
# ══════════════════════════════════════════════════════════════
def run_undershoot_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[7/8] Bắt đầu phân tích tham số Undershoot (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(UNDERSHOOT_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)
            all_results = []
            skipped_pairs = set()
            id_col = mod_undershoot._find_id_col(gdf)

            for iteration in range(1, mod_undershoot.ITERATIONS_MAX + 1):
                detections = mod_undershoot.detect_undershoot(gdf, dist_threshold=tol)

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

                results, gdf = mod_undershoot.correct_undershoot(gdf, detections, dist_threshold=tol)

                for r in results:
                    if r["Status"] == "SKIPPED":
                        for tgt in r["ID-Target"].split(","):
                            skipped_pairs.add(frozenset({r["ID-Source"], tgt.strip()}))

                all_results.extend(results)

            detect_cnt = len(all_results)
            fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")

            if all_results:
                df = pd.DataFrame(all_results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = [
                    "STT", "ID-Source", "OldLen-Source", "NewLen-Source",
                    "ID-Target", "Angle", "Status"
                ]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Undershoot (tolerance) vào: {UNDERSHOOT_EXCEL}")


# ══════════════════════════════════════════════════════════════
# 8. Phân tích tham số Dangling Lines (tolerance)
# ══════════════════════════════════════════════════════════════
def run_dangling_tolerance_analysis(gdf_orig: gpd.GeoDataFrame):
    print(f"\n[8/8] Bắt đầu phân tích tham số Dangling Lines (tolerance): {TOLERANCE_VALUES}")
    print("=" * 70)

    with pd.ExcelWriter(DANGLING_EXCEL, engine="openpyxl") as writer:
        for tol in TOLERANCE_VALUES:
            sheet_name = str(tol)

            gdf = gdf_orig.copy(deep=True)
            all_results = []

            for iteration in range(1, mod_dangling.ITERATIONS_MAX + 1):
                detections = mod_dangling.detect_dangling(gdf, dist_threshold=tol)
                if not detections:
                    break
                results, gdf = mod_dangling.correct_dangling(gdf, detections)
                all_results.extend(results)

            detect_cnt = len(all_results)
            fixed_cnt = sum(1 for r in all_results if r["Status"] == "FIXED")

            if all_results:
                df = pd.DataFrame(all_results)
                df.insert(0, "STT", range(1, len(df) + 1))
            else:
                columns = ["STT", "ID-Source", "OldLen-Source", "NewLen-Source", "ID-Target", "Status"]
                df = pd.DataFrame(columns=columns)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  • Sheet [{sheet_name:<6}]: Phát hiện {detect_cnt:>3} lỗi | Sửa thành công {fixed_cnt:>3} lỗi")

    print("=" * 70)
    print(f"✅ Đã lưu kết quả Dangling Lines (tolerance) vào: {DANGLING_EXCEL}")


# ── Hàm chạy chính ────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang nạp dữ liệu lớp ống...")
    gdf_orig, layer_meta = load_layers()
    print(f"→ Đã nạp {len(gdf_orig)} ống từ {len(layer_meta)} lớp.")

    # 1. Spike Lines (angle_max)
    run_spike_analysis(gdf_orig)

    # 2. Overlap Lines (ratio)
    run_overlap_ratio_analysis(gdf_orig)

    # 3. Sliver Gap Lines (tolerance)
    run_sliver_gap_tolerance_analysis(gdf_orig)

    # 4. Crossing (tolerance)
    run_crossing_tolerance_analysis(gdf_orig)

    # 5. T-Junction (tolerance)
    run_tjunction_tolerance_analysis(gdf_orig)

    # 6. Overshoot (tolerance)
    run_overshoot_tolerance_analysis(gdf_orig)

    # 7. Undershoot (tolerance)
    run_undershoot_tolerance_analysis(gdf_orig)

    # 8. Dangling Lines (tolerance)
    run_dangling_tolerance_analysis(gdf_orig)

    print("\n" + "═" * 70)
    print("🎉 ĐÃ HOÀN THÀNH TẤT CẢ PHÂN TÍCH THAM SỐ TOPOLOGY!")
    print(f"📁 Thư mục kết quả: {OUTPUT_DIR}")
    print("═" * 70)


if __name__ == "__main__":
    main()
