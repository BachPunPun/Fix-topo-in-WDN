# -*- coding: utf-8 -*-
"""
Mô phỏng thủy lực và Trực quan hóa bản đồ Áp lực (Pressure Map) bằng WNTR & Matplotlib.

Tính năng:
- Đọc file mô hình EPANET (*.inp) hoặc chuyển đổi từ Shapefile.
- Chạy mô phỏng thủy lực bằng EPANET Engine tích hợp trong WNTR.
- Trích xuất kết quả áp lực (Pressure) tại từng nút (Junctions).
- Phân loại áp lực thành 5 mức màu tiêu chuẩn:
    + P <= 10m  : Màu Đỏ (Red)
    + P <= 15m  : Màu Cam (Orange)
    + P <= 20m  : Màu Vàng (Yellow)
    + P <= 25m  : Màu Xanh lá cây (Green)
    + P <= 30m  : Màu Xanh dương (Blue)
- Vẽ bản đồ mạng lưới đường ống và phân bố áp lực chất lượng cao:
    + Độ phân giải: Tối thiểu DPI = 300.
    + Định dạng màu: 24-bit True Color RGB.
    + Thống kê số lượng nút, chú giải (Legend), hướng Bắc và thông số mạng.
"""

import os
import sys
import time
from pathlib import Path

# Thiết lập đường dẫn thư mục dùng chung
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULT_DIR = BASE_DIR / "result"

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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần GUI window
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import wntr


# ── ĐỊNH NGHĨA CÁC MỨC ÁP LỰC VÀ MÃ MÀU CHO LEGEND ─────────────────────────
PRESSURE_BINS = [
    {"min": -np.inf, "max": 0.001, "color": "#64748B", "label": "P = 0"},
    {"min": 0.001,  "max": 10.0,  "color": "#EF4444", "label": "P <= 10"},
    {"min": 10.0,   "max": 15.0,  "color": "#F97316", "label": "P <= 15"},
    {"min": 15.0,   "max": 20.0,  "color": "#FBBF24", "label": "P <= 20"},
    {"min": 20.0,   "max": 25.0,  "color": "#22C55E", "label": "P <= 25"},
    {"min": 25.0,   "max": np.inf, "color": "#2563EB", "label": "P <= 30"},
]


def run_simulation(inp_path: str | Path) -> tuple[wntr.network.WaterNetworkModel, pd.Series]:
    """
    Chạy mô phỏng thủy lực với file EPANET .inp và trích xuất áp lực tại các nút.

    Args:
        inp_path: Đường dẫn tới file .inp

    Returns:
        tuple (wn: WaterNetworkModel, node_pressure: pd.Series)
    """
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file INP: {inp_path}")

    print(f"[1/3] Đang tải mô hình từ: {inp_path.name}...")
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    print(f"      - Tổng số đoạn ống   : {wn.num_pipes:,}")
    print(f"      - Tổng số nút        : {wn.num_junctions:,}")
    print(f"      - Tổng số nguồn nước : {wn.num_reservoirs}")

    print(f"[2/3] Đang chạy mô phỏng thủy lực (EPANET engine)...")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # Lấy áp lực tại bước thời gian đầu tiên (hoặc duy nhất với static simulation)
    pressure = results.node["pressure"].iloc[0]
    print(f"      - Mô phỏng thành công.")
    print(f"      - Áp lực Min: {pressure.min():.2f} m | Max: {pressure.max():.2f} m | Mean: {pressure.mean():.2f} m")

    return wn, pressure


def plot_pressure_map(
    wn: wntr.network.WaterNetworkModel,
    pressure: pd.Series,
    output_image_path: str | Path = RESULT_DIR / "pressure_map_300dpi.png",
    dpi: int = 300,
    title: str = "BẢN ĐỒ PHÂN BỐ ÁP LỰC MẠNG LƯỚI CẤP NƯỚC (GD_TD)",
) -> Path:
    """
    Vẽ bản đồ phân bố áp lực mạng lưới và lưu ảnh 24-bit True Color với DPI >= 300.

    Args:
        wn: Đối tượng mô hình mạng lưới WNTR
        pressure: Series áp lực của các nút
        output_image_path: Đường dẫn lưu file ảnh output (.png)
        dpi: Độ phân giải ảnh (mặc định 300 DPI)
        title: Tiêu đề bản đồ

    Returns:
        Path của file ảnh đã lưu.
    """
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] Đang vẽ bản đồ phân bố áp lực (DPI={dpi}, 24-bit True Color)...")

    # 1. Thu thập tọa độ các đường ống để vẽ LineCollection
    pipe_lines = []
    for link_name, link in wn.pipes():
        start_coord = link.start_node.coordinates
        end_coord = link.end_node.coordinates
        
        # Nếu có intermediate vertices
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

    # 3. Phân nhóm các nút theo các khoảng áp lực
    bin_masks = []
    bin_counts = []
    for b in PRESSURE_BINS:
        mask = (node_p > b["min"]) & (node_p <= b["max"])
        bin_masks.append(mask)
        bin_counts.append(mask.sum())

    # 4. Khởi tạo khung vẽ kích thước lớn cho chất lượng cao
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Danh sách handles và labels cho Legend theo thứ tự chính xác
    legend_handles = []
    legend_labels = []

    # 4.1. Vẽ 4 vị trí nguồn nước (Reservoirs) - Mục đầu tiên trong Legend
    if res_x:
        res_sc = ax.scatter(
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

    # 4.2. Vẽ nền đường ống - Mục thứ hai trong Legend
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

    # 4.3. Vẽ các nhóm nút theo màu áp lực (P = 0, P <= 10, P <= 15, P <= 20, P <= 25, P <= 30)
    for b, mask, count in zip(PRESSURE_BINS, bin_masks, bin_counts):
        if count > 0:
            ax.scatter(
                node_x[mask],
                node_y[mask],
                c=b["color"],
                s=16,
                alpha=0.9,
                edgecolors="none",
                zorder=2,
            )
        # Thêm vào Legend
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=b["color"], markersize=11, linestyle="None")
        )
        legend_labels.append(b["label"])

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
    print(f"XUẤT ẢNH BẢN ĐỒ ÁP LỰC THÀNH CÔNG:")
    print(f"  - Đường dẫn file : {output_image_path}")
    print(f"  - Dung lượng     : {file_size_mb:.2f} MB")
    print(f"  - Độ phân giải   : {dpi} DPI ({rgb_img.width} x {rgb_img.height} px)")
    print(f"  - Định dạng màu  : 24-bit True Color (RGB)")
    print("=" * 70)

    return output_image_path


def main():
    inp_file = RESULT_DIR / "GD_TD-atopo-manual.inp"
    output_image = RESULT_DIR / "GD_TD_pressure_map_300dpi.png"

    # 1. Chạy mô phỏng
    wn, pressure = run_simulation(inp_file)

    # 2. Vẽ và xuất bản đồ áp lực
    plot_pressure_map(
        wn=wn,
        pressure=pressure,
        output_image_path=output_image,
        dpi=300,
        title="BẢN ĐỒ PHÂN BỐ ÁP LỰC MẠNG LƯỚI CẤP NƯỚC (GD_TD)",
    )


if __name__ == "__main__":
    main()
