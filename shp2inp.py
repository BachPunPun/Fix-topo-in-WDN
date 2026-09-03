# -*- coding: utf-8 -*-
"""
Chuyển đổi dữ liệu mạng lưới đường ống từ Shapefile (.shp) sang EPANET (.inp).

Hỗ trợ chuyển đổi cả 2 tập dữ liệu:
- data_before: data/GiaDinh/OPP_2510.shp (Dữ liệu gốc trước khi sửa topology)
- data_after : result/GD_TD-atopo-manual.shp (Dữ liệu sau khi xử lý topology)

Tính năng chính:
- Đọc Shapefile đường ống (hỗ trợ cả LineString và MultiLineString).
- Tự động xây dựng đồ thị topology, gán nhãn start-node và end-node cho từng đoạn ống.
- Khởi tạo 4 nguồn nước (Reservoir: R_1..R_4) tại các vị trí tọa độ chỉ định kèm cột nước.
- Tính lại chiều dài hình học (Len) cho toàn bộ các cạnh.
- Trích xuất tọa độ nút ([COORDINATES]) và các đỉnh uốn cong của ống ([VERTICES]).
- Xuất ra file EPANET *.inp chuẩn EPANET 2.2, không tạo file phụ *_with_nodes.shp.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Thiết lập đường dẫn thư mục dùng chung
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULT_DIR = BASE_DIR / "result"
DATA_DIR = BASE_DIR / "data" / "GiaDinh"

# Thiết lập tham số đường dẫn mặc định
DEFAULT_DATA_BEFORE = DATA_DIR / "OPP_2510.shp"
DEFAULT_DATA_AFTER = RESULT_DIR / "GD_TD-atopo-manual.shp"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Đảm bảo in UTF-8 không lỗi trên Windows console
try:
    import setup_console
except ImportError:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

# Nạp read_shp / pyogrio để tối ưu tốc độ đọc shapefile
try:
    from read_file import read_shp
except ImportError:
    import pyogrio

    def read_shp(path):
        return pyogrio.read_dataframe(str(path))


# ── TỌA ĐỘ VÀ CỘT NƯỚC 4 NGUỒN NƯỚC CỐ ĐỊNH ──────────────────────────────────
DEFAULT_SOURCES = [
    {"name": "R_1", "coord": (606964.85, 1194561.83), "head": 20.0},
    {"name": "R_2", "coord": (605186.68, 1197050.66), "head": 20.0},
    {"name": "R_3", "coord": (604982.88, 1197438.42), "head": 30.0},
    {"name": "R_4", "coord": (601350.31, 1192605.80), "head": 15.0},
]

# ── HỆ SỐ NHÁM HAZEN-WILLIAMS THEO VẬT LIỆU ──────────────────────────────────
ROUGHNESS_MAP = {
    "UPVC": 140,
    "PVC": 140,
    "HDPE": 140,
    "DI": 130,          # Gang cầu
    "CI": 100,          # Gang xám
    "STEEL": 120,       # Thép
    "BTCT": 120,        # Bê tông cốt thép
    "BTCT-TA-KNT": 120,
}
DEFAULT_ROUGHNESS = 130
DEFAULT_DIAMETER = 100.0  # mm khi không có cỡ ống hoặc cỡ ống = 0


def convert_shp_to_inp(
    shp_path: str | Path = DEFAULT_DATA_AFTER,
    output_inp_path: str | Path | None = None,
    sources: list[dict | tuple] = DEFAULT_SOURCES,
    precision: int = 3,
    label: str | None = None,
) -> dict:
    """
    Đọc Shapefile mạng lưới ống, xây dựng topology nút, gán nguồn nước,
    tính lại chiều dài và xuất file EPANET .inp duy nhất (không tạo file phụ *_with_nodes).

    Args:
        shp_path: Đường dẫn tới file Shapefile đầu vào (.shp).
        output_inp_path: Đường dẫn file .inp đầu ra (mặc định result/<tên_shp>.inp).
        sources: Danh sách thông tin các nguồn nước [{"name": "R_1", "coord": (x,y), "head": h}, ...]
        precision: Số chữ số làm tròn sau dấu phẩy cho tọa độ (mặc định 3 chữ số ~ 1mm).
        label: Nhãn phân biệt (ví dụ 'BEFORE' hoặc 'AFTER').

    Returns:
        dict chứa thông tin tóm tắt kết quả chuyển đổi.
    """
    start_time = time.time()
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file shapefile: {shp_path}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if output_inp_path is None:
        output_inp_path = RESULT_DIR / f"{shp_path.stem}.inp"
    else:
        output_inp_path = Path(output_inp_path)
        output_inp_path.parent.mkdir(parents=True, exist_ok=True)

    if label is None:
        label_upper = shp_path.stem.upper()
        if "BEFORE" in label_upper or "2510" in label_upper:
            label = "BEFORE"
        elif "MANUAL" in label_upper or "AFTER" in label_upper or "GD_TD" in label_upper:
            label = "AFTER"
        else:
            label = shp_path.stem

    # Chuẩn hóa cấu trúc sources
    norm_sources = []
    for idx, s in enumerate(sources, 1):
        if isinstance(s, dict):
            name = s.get("name", f"R_{idx}")
            coord = s["coord"]
            head = float(s.get("head", 30.0))
        elif isinstance(s, (list, tuple)):
            name = f"R_{idx}"
            coord = (float(s[0]), float(s[1]))
            head = float(s[2]) if len(s) > 2 else 30.0
        else:
            continue
        norm_sources.append({"name": name, "coord": coord, "head": head})

    print("\n" + "=" * 70, flush=True)
    print(f"BẮT ĐẦU CHUYỂN ĐỔI SHAPEFILE SANG EPANET (.INP) [{label}]", flush=True)
    print(f"  - File đầu vào : {shp_path}", flush=True)
    print(f"  - File đầu ra  : {output_inp_path}", flush=True)
    print(f"  - Số nguồn nước: {len(norm_sources)} vị trí:", flush=True)
    for s in norm_sources:
        print(f"      + {s['name']}: Cột nước = {s['head']:.1f}m tại tọa độ {s['coord']}", flush=True)
    print("=" * 70, flush=True)

    # 1. Đọc dữ liệu shapefile
    print(f"[{label}] [1/5] Đang đọc file shapefile...", flush=True)
    gdf = read_shp(shp_path)
    num_pipes = len(gdf)
    print(f"      Đã nạp {num_pipes:,} đoạn ống.", flush=True)

    # 2. Tính lại chiều dài hình học (Len)
    print(f"[{label}] [2/5] Đang tính lại chiều dài hình học (Len)...", flush=True)
    geoms = gdf.geometry.values
    recalc_lens = np.zeros(num_pipes, dtype=np.float64)

    for i, geom in enumerate(geoms):
        if geom is not None and not geom.is_empty:
            recalc_lens[i] = round(float(geom.length), precision)
        else:
            recalc_lens[i] = 0.1

    gdf["Len"] = recalc_lens
    total_len = float(np.sum(recalc_lens))
    print(f"      Tổng chiều dài mạng lưới: {total_len:,.2f} m ({total_len/1000:,.2f} km).", flush=True)

    # 3. Trích xuất tọa độ điểm đầu/cuối và các điểm uốn khúc (Vertices)
    print(f"[{label}] [3/5] Đang xây dựng topology nút (start-node & end-node)...", flush=True)
    start_coords = []
    end_coords = []
    pipe_vertices_list = []  # Danh sách (pipe_idx, [(vx, vy), ...])

    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            p_start = (0.0, 0.0)
            p_end = (0.0, 0.0)
            vertices = []
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
            p_start = (round(coords[0][0], precision), round(coords[0][1], precision))
            p_end = (round(coords[-1][0], precision), round(coords[-1][1], precision))
            vertices = [
                (round(vx, precision), round(vy, precision))
                for vx, vy in coords[1:-1]
            ]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
            c_first = list(lines[0].coords)
            c_last = list(lines[-1].coords)
            p_start = (round(c_first[0][0], precision), round(c_first[0][1], precision))
            p_end = (round(c_last[-1][0], precision), round(c_last[-1][1], precision))

            # Trích xuất tất cả đỉnh trung gian
            all_pts = []
            for line in lines:
                all_pts.extend(list(line.coords))
            vertices = [
                (round(vx, precision), round(vy, precision))
                for vx, vy in all_pts[1:-1]
            ]
        else:
            p_start = (0.0, 0.0)
            p_end = (0.0, 0.0)
            vertices = []

        start_coords.append(p_start)
        end_coords.append(p_end)
        pipe_vertices_list.append(vertices)

    unique_nodes = list(dict.fromkeys(start_coords + end_coords))
    node_coords_arr = np.array(unique_nodes)
    print(f"      Tổng số nút duy nhất tìm thấy: {len(unique_nodes):,} nút.", flush=True)

    # 4. Tìm và ghép nối 4 vị trí nguồn nước với nút mạng gần nhất
    print(f"[{label}] [4/5] Đang xác định vị trí các nguồn nước (Reservoirs)...", flush=True)
    source_node_map = {}
    for s in norm_sources:
        s_name = s["name"]
        sx, sy = s["coord"]
        s_head = s["head"]

        dists = np.hypot(node_coords_arr[:, 0] - sx, node_coords_arr[:, 1] - sy)
        min_idx = int(np.argmin(dists))
        best_coord = unique_nodes[min_idx]
        best_dist = dists[min_idx]

        source_node_map[best_coord] = s_name
        print(
            f"      Nguồn {s_name}: Mục tiêu ({sx:.2f}, {sy:.2f}) -> {s_name} tại "
            f"({best_coord[0]:.3f}, {best_coord[1]:.3f}) (kc: {best_dist:.4f} m, Head = {s_head:.1f}m)",
            flush=True,
        )

    # Đặt ID cho toàn bộ các nút
    node_id_lookup = {}
    junction_idx = 0
    for coord in unique_nodes:
        if coord in source_node_map:
            node_id_lookup[coord] = source_node_map[coord]
        else:
            junction_idx += 1
            node_id_lookup[coord] = f"J_{junction_idx}"

    start_node_ids = [node_id_lookup[c] for c in start_coords]
    end_node_ids = [node_id_lookup[c] for c in end_coords]

    # 5. Sinh nội dung file EPANET .inp
    print(f"[{label}] [5/5] Đang tạo file EPANET .inp...", flush=True)

    junction_lines = []
    reservoir_lines = []
    coord_lines = []

    # Ghi danh sách nguồn nước (R_1..R_4)
    for s in norm_sources:
        r_id = s["name"]
        r_head = s["head"]
        reservoir_lines.append(f"{r_id:<16} {r_head:<10.2f}")

    # Ghi danh sách nút (Junctions) và Tọa độ (Coordinates)
    for coord, n_id in node_id_lookup.items():
        x, y = coord
        coord_lines.append(f"{n_id:<16} {x:<14.3f} {y:<14.3f}")
        if not n_id.startswith("R_"):
            junction_lines.append(f"{n_id:<16} {0.0:<10.2f} {0.0:<10.2f}")

    # Chuẩn hóa cột đường kính và vật liệu
    col_map = {c.lower().strip(): c for c in gdf.columns}
    dia_col = next((col_map[k] for k in ["coong", "diameter", "dn", "size"] if k in col_map), None)
    mat_col = next((col_map[k] for k in ["vatlieu", "material", "chatlieu"] if k in col_map), None)

    dia_values = gdf[dia_col].values if dia_col else [DEFAULT_DIAMETER] * num_pipes
    mat_values = gdf[mat_col].values if mat_col else ["UPVC"] * num_pipes

    pipe_lines = []
    vertices_lines = []

    for idx in range(num_pipes):
        pipe_id = f"P_{idx + 1}"
        u = start_node_ids[idx]
        v = end_node_ids[idx]
        length = max(0.1, float(recalc_lens[idx]))

        # Đường kính
        raw_dia = dia_values[idx]
        try:
            dia = float(raw_dia) if pd.notna(raw_dia) and float(raw_dia) > 0 else DEFAULT_DIAMETER
        except (ValueError, TypeError):
            dia = DEFAULT_DIAMETER

        # Độ nhám
        raw_mat = str(mat_values[idx]).strip().upper() if pd.notna(mat_values[idx]) else "UPVC"
        roughness = ROUGHNESS_MAP.get(raw_mat, DEFAULT_ROUGHNESS)

        pipe_lines.append(
            f"{pipe_id:<16} {u:<16} {v:<16} {length:<10.2f} {dia:<10.1f} {roughness:<10} {0:<10} OPEN"
        )

        # Ghi các đỉnh uốn khúc
        for vx, vy in pipe_vertices_list[idx]:
            vertices_lines.append(f"{pipe_id:<16} {vx:<14.3f} {vy:<14.3f}")

    # Bounding box cho backdrop
    bounds = gdf.total_bounds
    margin = 200.0
    backdrop_info = (
        f" DIMENSIONS           {bounds[0] - margin:.2f} {bounds[1] - margin:.2f} "
        f"{bounds[2] + margin:.2f} {bounds[3] + margin:.2f}\n"
        f" UNITS                Meters"
    )

    inp_sections = [
        ";EPANET Water Distribution Network Model",
        f";Source File : {shp_path.name} ({label})",
        f";Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f";Summary     : {len(pipe_lines):,} Pipes, {len(junction_lines):,} Junctions, {len(reservoir_lines)} Reservoirs",
        "",
        "[TITLE]",
        f"Water Network Model - {shp_path.stem} ({label})",
        "",
        "[JUNCTIONS]",
        ";ID               Elevation  Demand     Pattern",
        "\n".join(junction_lines),
        "",
        "[RESERVOIRS]",
        ";ID               Head       Pattern",
        "\n".join(reservoir_lines),
        "",
        "[TANKS]",
        ";ID               Elevation  InitLevel  MinLevel   MaxLevel   Diameter   MinVol     VolCurve",
        "",
        "[PIPES]",
        ";ID               Node1            Node2            Length     Diameter   Roughness  MinorLoss  Status",
        "\n".join(pipe_lines),
        "",
        "[PUMPS]",
        ";ID               Node1            Node2            Parameters",
        "",
        "[VALVES]",
        ";ID               Node1            Node2            Diameter   Type       Setting    MinorLoss",
        "",
        "[TAGS]",
        "",
        "[DEMANDS]",
        ";ID               Demand     Pattern    Category",
        "",
        "[STATUS]",
        ";ID               Status/Setting",
        "",
        "[PATTERNS]",
        ";ID               Multipliers",
        "",
        "[CURVES]",
        ";ID               X-Value    Y-Value",
        "",
        "[CONTROLS]",
        "",
        "[RULES]",
        "",
        "[ENERGY]",
        " Global Efficiency    75",
        " Global Price         0.0",
        " Demand Charge        0.0",
        "",
        "[EMITTERS]",
        ";ID               FlowCoeff",
        "",
        "[QUALITY]",
        ";Node             InitQual",
        "",
        "[SOURCES]",
        ";Node             Type       Quality    Pattern",
        "",
        "[REACTIONS]",
        ";Type             Pipe/Tank  Coeff",
        "",
        "[REACTIONS]",
        " Order Bulk           1",
        " Order Wall           1",
        " Global Bulk          0.0",
        " Global Wall          0.0",
        " Limiting Potential   0.0",
        " Roughness Correlation 0.0",
        "",
        "[MIXING]",
        ";Tank             Model",
        "",
        "[TIMES]",
        " Duration             0:00",
        " Hydraulic Timestep   1:00",
        " Quality Timestep     0:05",
        " Pattern Timestep     1:00",
        " Pattern Start        0:00",
        " Report Timestep      1:00",
        " Report Start         0:00",
        " Start ClockTime      12 am",
        " Statistic            None",
        "",
        "[REPORT]",
        " Status               Yes",
        " Summary              Yes",
        " Page                 0",
        "",
        "[OPTIONS]",
        " Units                LPS",
        " Headloss             H-W",
        " Specific Gravity     1.0",
        " Viscosity            1.0",
        " Trials               40",
        " Accuracy             0.001",
        " CHECKFREQ            2",
        " MAXCHECK             10",
        " DAMPLIMIT            0",
        " Unbalanced           Continue 10",
        " Pattern              1",
        " Demand Multiplier    1.0",
        " Emitter Exponent     0.5",
        " Quality              None mg/L",
        " Diffusivity          1.0",
        " Tolerance            0.01",
        "",
        "[COORDINATES]",
        ";Node             X-Coord        Y-Coord",
        "\n".join(coord_lines),
        "",
        "[VERTICES]",
        ";Link             X-Coord        Y-Coord",
        "\n".join(vertices_lines),
        "",
        "[LABELS]",
        ";X-Coord          Y-Coord        Label & Anchor Node",
        "",
        "[BACKDROP]",
        backdrop_info,
        "",
        "[END]",
        "",
    ]

    with open(output_inp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(inp_sections))

    file_size_mb = output_inp_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print("=" * 70, flush=True)
    print(f"XUẤT FILE THÀNH CÔNG [{label}]:", flush=True)
    print(f"  - File INP     : {output_inp_path} ({file_size_mb:.2f} MB)", flush=True)
    print(f"  - Tổng số ống  : {len(pipe_lines):,}", flush=True)
    print(f"  - Số Junctions : {len(junction_lines):,}", flush=True)
    sources_summary = ", ".join([f"{s['name']}={s['head']}m" for s in norm_sources])
    print(f"  - Số Nguồn     : {len(reservoir_lines)} ({sources_summary})", flush=True)
    print(f"  - Số Vertices  : {len(vertices_lines):,}", flush=True)
    print(f"  - Thời gian    : {elapsed:.2f}s", flush=True)
    print("=" * 70, flush=True)

    return {
        "label": label,
        "input_shp": shp_path,
        "output_inp": output_inp_path,
        "pipes": len(pipe_lines),
        "junctions": len(junction_lines),
        "reservoirs": len(reservoir_lines),
        "vertices": len(vertices_lines),
        "file_size_mb": file_size_mb,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi dữ liệu mạng lưới đường ống từ Shapefile (.shp) sang EPANET (.inp)."
    )
    parser.add_argument(
        "--data_before",
        type=Path,
        default=DEFAULT_DATA_BEFORE,
        help=f"Đường dẫn dữ liệu trước sửa topology (mặc định: {DEFAULT_DATA_BEFORE})",
    )
    parser.add_argument(
        "--data_after",
        type=Path,
        default=DEFAULT_DATA_AFTER,
        help=f"Đường dẫn dữ liệu sau sửa topology (mặc định: {DEFAULT_DATA_AFTER})",
    )
    parser.add_argument(
        "--target",
        choices=["all", "after", "before"],
        default="all",
        help="Chọn tập dữ liệu chuyển đổi: 'all' (cả 2), 'after', hoặc 'before' (mặc định: all)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Đường dẫn file .shp tùy chỉnh (ghi đè target)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Đường dẫn file .inp đầu ra (tùy chọn)",
    )

    args = parser.parse_args()

    results = []

    # 1. Nếu chỉ định file đầu vào tùy chỉnh
    if args.input is not None:
        res = convert_shp_to_inp(shp_path=args.input, output_inp_path=args.output)
        results.append(res)

    # 2. Nếu chọn target = 'before'
    elif args.target == "before":
        res = convert_shp_to_inp(
            shp_path=args.data_before,
            output_inp_path=args.output,
            label="BEFORE",
        )
        results.append(res)

    # 3. Nếu chọn target = 'after'
    elif args.target == "after":
        res = convert_shp_to_inp(
            shp_path=args.data_after,
            output_inp_path=args.output,
            label="AFTER",
        )
        results.append(res)

    # 4. Mặc định 'all': Chuyển đổi cả Before và After
    else:
        # Chuyển đổi Before
        res_before = convert_shp_to_inp(
            shp_path=args.data_before,
            output_inp_path=RESULT_DIR / f"{args.data_before.stem}.inp",
            label="BEFORE",
        )
        results.append(res_before)

        # Chuyển đổi After
        res_after = convert_shp_to_inp(
            shp_path=args.data_after,
            output_inp_path=RESULT_DIR / f"{args.data_after.stem}.inp",
            label="AFTER",
        )
        results.append(res_after)

    # In bảng tổng kết cuối cùng nếu có nhiều kết quả
    if len(results) > 1:
        print("\n" + "=" * 70, flush=True)
        print("TỔNG KẾT QUÁ TRÌNH CHUYỂN ĐỔI SHP SANG INP:", flush=True)
        for res in results:
            print(
                f"  ✓ [{res['label']}] {res['output_inp'].name} ({res['file_size_mb']:.2f} MB) | "
                f"{res['pipes']:,} ống | {res['junctions']:,} nút | {res['reservoirs']} nguồn | {res['time']:.2f}s",
                flush=True,
            )
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
