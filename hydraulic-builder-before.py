# -*- coding: utf-8 -*-
"""
Mô phỏng thủy lực và Trực quan hóa bản đồ Áp lực Mạng lưới Gốc (Before - result/OPP_2510.inp)
bằng WNTR & Matplotlib.

Đặc điểm bản đồ:
- Dữ liệu: result/OPP_2510.inp (Mạng lưới trước khi sửa topology).
- Phân loại áp lực thành 2 nhóm chính:
    + P = 0    : Các nút bị cô lập / mất áp lực (Isolated Nodes).
    + P ≠ 0    : Các nút có áp lực được cấp nước từ nguồn (Active Nodes).
- Bảng Legend gồm:
    1. Reservoir
    2. Pipe
    3. P = 0
    4. P ≠ 0
- Chuẩn xuất ảnh:
    + Độ phân giải: 300 DPI.
    + Định dạng màu: 24-bit True Color RGB.
    + Không viền khung ngoài, không chữ N chỉ hướng, nền trắng liền mạch.
"""

import os
import sys
import time
from pathlib import Path

# Thiết lập đường dẫn thư mục dùng chung
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULT_DIR = BASE_DIR / "result"
DATA_DIR = BASE_DIR / "data" / "GiaDinh"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import setup_console
except ImportError:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import copy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần GUI window
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

import wntr


# ── ĐỊNH NGHĨA 2 MỨC ÁP LỰC CHO MẠNG LƯỚI BEFORE ───────────────────────────
PRESSURE_CATEGORIES = [
    {"type": "zero",     "label": "P = 0", "color": "#EF4444", "name": "Không có áp lực"},
    {"type": "non-zero", "label": "P ≠ 0", "color": "#22C55E", "name": "Có áp lực"},
]


def run_simulation_before(inp_path: str | Path) -> tuple[wntr.network.WaterNetworkModel, pd.Series]:
    """
    Chạy mô phỏng thủy lực cho mạng lưới trước khi sửa topology (OPP_2510.inp).
    Tự động cô lập và gán áp lực P = 0 cho các phân vùng bị ngắt kết nối khỏi nguồn.

    Args:
        inp_path: Đường dẫn tới file .inp

    Returns:
        tuple (wn: WaterNetworkModel, node_pressure: pd.Series)
    """
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file INP: {inp_path}")

    print(f"[1/3] Đang tải mô hình gốc từ: {inp_path.name}...")
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    print(f"      - Tổng số đoạn ống   : {wn.num_pipes:,}")
    print(f"      - Tổng số nút        : {wn.num_junctions:,}")
    print(f"      - Tổng số nguồn nước : {wn.num_reservoirs}")

    print(f"[2/3] Đang phân tích đồ thị liên thông và chạy mô phỏng thủy lực...")
    # Phân tích các thành phần liên thông
    G = wn.get_graph().to_undirected()
    components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    main_nodes = components[0]
    isolated_nodes = set().union(*components[1:]) if len(components) > 1 else set()

    print(f"      - Số thành phần liên thông : {len(components)}")
    print(f"      - Số nút thuộc vùng cấp nước chính : {len(main_nodes):,}")
    print(f"      - Số nút bị cô lập (P = 0)         : {len(isolated_nodes):,}")

    # Nạp mô hình để chạy mô phỏng cho thành phần liên thông chính
    wn_sim = wntr.network.WaterNetworkModel(str(inp_path))
    if isolated_nodes:
        # Xóa các ống thuộc nút cô lập
        pipes_to_remove = [
            p_name for p_name, p in wn_sim.pipes()
            if p.start_node_name in isolated_nodes or p.end_node_name in isolated_nodes
        ]
        for p_name in pipes_to_remove:
            wn_sim.remove_link(p_name)
        for n_name in isolated_nodes:
            wn_sim.remove_node(n_name)

    sim = wntr.sim.EpanetSimulator(wn_sim)
    results = sim.run_sim()
    sim_pressure = results.node["pressure"].iloc[0]

    # Tổng hợp kết quả áp lực cho toàn bộ mạng lưới gốc (gồm cả nút cô lập P = 0)
    all_node_names = wn.node_name_list
    full_pressure = pd.Series(0.0, index=all_node_names)
    for n in sim_pressure.index:
        full_pressure[n] = sim_pressure[n]

    print(f"      - Mô phỏng hoàn tất.")
    print(f"      - Nút có áp lực (P > 0) : {(full_pressure > 0.001).sum():,}")
    print(f"      - Nút mất áp (P = 0)    : {(full_pressure <= 0.001).sum():,}")

    return wn, full_pressure


def plot_pressure_map_before(
    wn: wntr.network.WaterNetworkModel,
    pressure: pd.Series,
    output_image_path: str | Path = RESULT_DIR / "OPP_2510_pressure_map_300dpi.png",
    dpi: int = 300,
    title: str = "BẢN ĐỒ ÁP LỰC MẠNG LƯỚI GỐC (BEFORE - OPP_2510)",
) -> Path:
    """
    Vẽ bản đồ phân bố áp lực mạng lưới gốc và lưu ảnh 24-bit True Color với DPI >= 300.
    Legend bao gồm: Reservoir, Pipe, P = 0, P ≠ 0.

    Args:
        wn: Đối tượng mô hình mạng lưới WNTR
        pressure: Series áp lực của toàn bộ các nút
        output_image_path: Đường dẫn lưu file ảnh output (.png)
        dpi: Độ phân giải ảnh (mặc định 300 DPI)
        title: Tiêu đề bản đồ

    Returns:
        Path của file ảnh đã lưu.
    """
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] Đang vẽ bản đồ phân bố áp lực Before (DPI={dpi}, 24-bit True Color)...")

    # 1. Thu thập tọa độ các đường ống để vẽ LineCollection
    pipe_lines = []
    for link_name, link in wn.pipes():
        start_coord = link.start_node.coordinates
        end_coord = link.end_node.coordinates
        
        coords = [start_coord]
        if hasattr(link, "vertices") and link.vertices:
            coords.extend(link.vertices)
        coords.append(end_coord)
        pipe_lines.append(coords)

    # 2. Thu thập tọa độ và áp lực của các nút (Junctions & Reservoirs)
    node_x = []
    node_y = []
    node_p = []
    node_names = []

    res_x = []
    res_y = []
    res_names = []
    res_heads = []

    for name, node in wn.nodes():
        x, y = node.coordinates
        p = pressure.get(name, 0.0)
        if node.node_type == "Reservoir":
            res_x.append(x)
            res_y.append(y)
            res_names.append(name)
            head_val = getattr(node.head_timeseries, "base_value", 30.0)
            res_heads.append(head_val)
        else:
            node_x.append(x)
            node_y.append(y)
            node_p.append(p)
            node_names.append(name)

    node_x = np.array(node_x)
    node_y = np.array(node_y)
    node_p = np.array(node_p)

    # 3. Phân loại 2 nhóm áp lực: P = 0 và P ≠ 0
    mask_zero = node_p <= 0.001
    mask_nonzero = node_p > 0.001

    count_zero = int(mask_zero.sum())
    count_nonzero = int(mask_nonzero.sum())

    # 4. Khởi tạo khung vẽ kích thước lớn cho chất lượng cao
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Danh sách handles và labels cho Legend theo đúng yêu cầu
    legend_handles = []
    legend_labels = []

    # 4.1. Vẽ 4 vị trí nguồn nước (Reservoir) - Mục 1 trong Legend
    if res_x:
        ax.scatter(
            res_x,
            res_y,
            c="#1E3A8A",
            s=160,
            marker="^",
            edgecolors="#FFFFFF",
            linewidths=2.0,
            zorder=4,
        )
        legend_handles.append(
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#1E3A8A", markersize=14, markeredgewidth=1.5, markeredgecolor="#FFFFFF", linestyle="None")
        )
        legend_labels.append("Reservoir")

        # Gắn nhãn tên nguồn nước và cột nước thực tế
        for rx, ry, rname, rhead in zip(res_x, res_y, res_names, res_heads):
            ax.annotate(
                f"  {rname} ({rhead:.0f}m)",
                (rx, ry),
                fontsize=11.5,
                fontweight="bold",
                color="#0F172A",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="#EFF6FF", ec="#3B82F6", alpha=0.9, lw=1.2),
            )

    # 4.2. Vẽ nền đường ống (Pipe) - Mục 2 trong Legend
    pipe_collection = LineCollection(
        pipe_lines,
        colors="#94A3B8",
        linewidths=0.55,
        alpha=0.55,
        zorder=1,
    )
    ax.add_collection(pipe_collection)
    legend_handles.append(
        plt.Line2D([0, 1], [0, 0], color="#94A3B8", lw=2.5)
    )
    legend_labels.append("Pipe")

    # 4.3. Vẽ nhóm P = 0 (Màu Đỏ #EF4444) - Mục 3 trong Legend
    if count_zero > 0:
        ax.scatter(
            node_x[mask_zero],
            node_y[mask_zero],
            c="#EF4444",
            s=18,
            alpha=0.95,
            edgecolors="none",
            zorder=3,
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#EF4444", markersize=11, linestyle="None")
    )
    legend_labels.append("P = 0")

    # 4.4. Vẽ nhóm P ≠ 0 (Màu Xanh lá cây #22C55E) - Mục 4 trong Legend
    if count_nonzero > 0:
        ax.scatter(
            node_x[mask_nonzero],
            node_y[mask_nonzero],
            c="#22C55E",
            s=15,
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#22C55E", markersize=11, linestyle="None")
    )
    legend_labels.append("P ≠ 0")

    # 5. Tinh chỉnh khung nhìn: Bỏ trục tọa độ, bỏ khung viền, giữ lại mạng lưới và chú giải
    margin_x = (node_x.max() - node_x.min()) * 0.02
    margin_y = (node_y.max() - node_y.min()) * 0.02
    ax.set_xlim(node_x.min() - margin_x, node_x.max() + margin_x)
    ax.set_ylim(node_y.min() - margin_y, node_y.max() + margin_y)
    ax.set_aspect("equal", adjustable="datalim")

    # Bỏ toàn bộ trục, vạch chia, nhãn và lưới tọa độ
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    # Bỏ khung viền ngoài (Border Frame)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Chú giải (Legend) kích thước lớn, rõ ràng
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.95,
        fontsize=14,
        title="Legend",
        title_fontsize=16,
        borderpad=1.0,
        labelspacing=0.8,
        handletextpad=0.8,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#0F172A")

    # 6. Lưu file ảnh với chuẩn 300 DPI và đảm bảo 24-bit True Color
    plt.tight_layout()
    temp_save_path = output_image_path.with_suffix(".tmp.png")
    fig.savefig(temp_save_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Chuyển đổi định dạng chuẩn 24-bit True Color RGB bằng Pillow
    with Image.open(temp_save_path) as img:
        rgb_img = img.convert("RGB")  # 24-bit True Color (8 bit x 3 channels: R, G, B)
        rgb_img.save(output_image_path, dpi=(dpi, dpi), format="PNG")

    if temp_save_path.exists():
        temp_save_path.unlink()

    file_size_mb = output_image_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"XUẤT ẢNH BẢN ĐỒ ÁP LỰC BEFORE THÀNH CÔNG:")
    print(f"  - Đường dẫn file : {output_image_path}")
    print(f"  - Dung lượng     : {file_size_mb:.2f} MB")
    print(f"  - Độ phân giải   : {dpi} DPI ({rgb_img.width} x {rgb_img.height} px)")
    print(f"  - Định dạng màu  : 24-bit True Color (RGB)")
    print(f"  - Thống kê nút   : P ≠ 0 ({count_nonzero:,} nút), P = 0 ({count_zero:,} nút)")
    print("=" * 70)

    return output_image_path


def main():
    inp_file = RESULT_DIR / "OPP_2510.inp"
    output_image = RESULT_DIR / "OPP_2510_pressure_map_300dpi.png"

    # 1. Chạy mô phỏng Before
    wn, pressure = run_simulation_before(inp_file)

    # 2. Vẽ và xuất bản đồ áp lực Before
    plot_pressure_map_before(
        wn=wn,
        pressure=pressure,
        output_image_path=output_image,
        dpi=300,
        title="BẢN ĐỒ ÁP LỰC MẠNG LƯỚI GỐC (BEFORE)",
    )


if __name__ == "__main__":
    main()
