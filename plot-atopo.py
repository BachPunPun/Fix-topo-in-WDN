# -*- coding: utf-8 -*-
"""
plot-atopo.py
Trực quan hóa sự thay đổi chiều dài mạng lưới và tỷ lệ thành phần liên thông lớn qua các bước xử lý topology.
Cấu trúc biểu đồ gồm 2 panel (2 subplots):
  (a) Network length (km): Chiều dài toàn mạng (Total length) và Chiều dài thành phần lớn nhất (Length of largest component)
  (b) Largest-component length share (%): Tỷ lệ chiều dài của thành phần liên thông lớn nhất
Dữ liệu nguồn: result/atopo-analysis/atopo.xlsx (hoặc dữ liệu truyền trực tiếp từ pipeline atopo.py)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl

# Thêm đường dẫn thư mục scripts, function và thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "function"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import setup_console
except ImportError:
    pass

import config

BASE_DIR = Path(__file__).parent
ANALYSIS_DIR = config.OUTPUT_DIR / "atopo-analysis"
PLOTS_DIR = ANALYSIS_DIR
EXCEL_PATH = ANALYSIS_DIR / "atopo.xlsx"


def load_atopo_network_data(excel_path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Đọc dữ liệu từ bảng '2. Sự thay đổi thành phần liên thông lớn' trong file Excel atopo.xlsx."""
    if not excel_path.exists():
        fallback = config.OUTPUT_DIR / "atopo.xlsx"
        if fallback.exists():
            excel_path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {excel_path}")

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    data = []
    capture = False

    for r in rows:
        if not r or not any(r):
            continue
        first_cell = str(r[0] or "").strip()
        second_cell = str(r[1] or "").strip()

        # Tìm dòng tiêu đề cột của bảng 2
        if first_cell == "STT" and "Bước" in second_cell:
            capture = True
            continue

        if capture:
            if first_cell.lower() == "final" or not first_cell or first_cell.startswith("3."):
                break
            try:
                stt = int(first_cell)
                step_raw = second_cell
                step_name = "Initial" if step_raw in ("Bắt đầu", "Initial") else step_raw
                tot_len_m = float(r[6])
                lg_len_m = float(r[7])
                ratio = float(r[8])

                data.append({
                    "stt": stt,
                    "step": step_name,
                    "tot_len_km": tot_len_m / 1000.0,
                    "lg_len_km": lg_len_m / 1000.0,
                    "share_pct": ratio,
                })
            except (ValueError, TypeError, IndexError):
                continue

    return pd.DataFrame(data)


def parse_network_report(network_report: list) -> pd.DataFrame:
    """Chuyển đổi danh sách network_report trực tiếp từ atopo.py thành DataFrame."""
    data = []
    for i, (name, s) in enumerate(network_report, start=1):
        step_name = "Initial" if name in ("Bắt đầu", "Initial") else name
        tot_len_m = float(s["TotalLen"])
        lg_len_m = float(s["LargestLen"])
        ratio = float(s["LenRatio"])

        data.append({
            "stt": i,
            "step": step_name,
            "tot_len_km": tot_len_m / 1000.0,
            "lg_len_km": lg_len_m / 1000.0,
            "share_pct": ratio,
        })
    return pd.DataFrame(data)


def plot_length_comparison(
    data: pd.DataFrame | list = None,
    excel_path: Path = EXCEL_PATH,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ so sánh chiều dài và tỷ lệ thành phần lớn gồm 2 panel chuẩn bài báo:
      (a) Network length (km): Đường Total length và Length of largest component
      (b) Largest-component length share (%): Tỷ lệ chiều dài thành phần liên thông lớn nhất
    Định dạng số làm tròn 2 chữ số, căn giữa trên marker và đồng bộ với plot-ratio.py, plot-spike.py.
    """
    if data is not None:
        if isinstance(data, list):
            df = parse_network_report(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Tham số data phải là list hoặc pd.DataFrame.")
    else:
        df = load_atopo_network_data(excel_path)

    if df.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ.")
        return

    x_indices = np.arange(len(df))
    x_labels = df["step"].tolist()

    # Bảng màu chuẩn từ plot-ratio.py và các file plot-*.py
    color_total = "#1f77b4"      # Xanh dương chuẩn (Delete/Primary từ plot-ratio.py)
    color_largest = "#157a74"    # Xanh mòng két thẫm
    color_share = "#b86e1e"      # Màu đồng / hổ phách đất

    # Xác định vị trí các mốc: Initial, Crossing, T-Junction, Dangling Lines (cuối)
    crossing_idx = df.index[df["step"] == "Crossing"].tolist()
    crossing_idx = crossing_idx[0] if crossing_idx else 7

    tjunction_idx = df.index[df["step"] == "T-Junction"].tolist()
    tjunction_idx = tjunction_idx[0] if tjunction_idx else 8

    annot_steps = [0, crossing_idx, tjunction_idx, len(df) - 1]

    # Thiết lập kích thước đồ thị đồng bộ với plot-spike.py dạng 2x1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.2), dpi=300)

    # ══════════════════════════════════════════════════════════
    # ── Đồ thị (a): Network length (km) ────────────────────────
    # ══════════════════════════════════════════════════════════
    ax1.plot(
        x_indices,
        df["tot_len_km"],
        color=color_total,
        linewidth=2.2,
        marker="s",
        markersize=6.0,
        markerfacecolor=color_total,
        markeredgecolor=color_total,
        label="Total length",
        zorder=3,
    )

    ax1.plot(
        x_indices,
        df["lg_len_km"],
        color=color_largest,
        linewidth=2.2,
        marker="o",
        markersize=6.0,
        markerfacecolor=color_largest,
        markeredgecolor=color_largest,
        label="Length of largest component",
        zorder=3,
    )

    # Chú thích số liệu làm tròn 2 chữ số, căn chính giữa marker
    for idx in annot_steps:
        # 1. Total length
        tot_val = df["tot_len_km"].iloc[idx]
        ax1.annotate(
            f"{tot_val:.2f}",
            (idx, tot_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2c3e50",
            zorder=5,
        )

        # 2. Length of largest component (mốc 0, 7 đặt trên marker; mốc 8, 11 đặt dưới marker)
        lg_val = df["lg_len_km"].iloc[idx]
        offset_y = 8 if idx in (0, crossing_idx) else -14
        ax1.annotate(
            f"{lg_val:.2f}",
            (idx, lg_val),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2c3e50",
            zorder=5,
        )

    # Tiêu đề & nhãn trục lấy định dạng theo plot-spike.py
    ax1.set_title("(a) Network length (km)", fontsize=12.0, pad=10)
    ax1.set_xlabel("Processing step", fontsize=11, labelpad=8)
    ax1.set_ylabel("Network length (km)", fontsize=11, labelpad=8)

    # Giới hạn trục & ticks
    ax1.set_xlim(-0.6, len(df) - 0.4)
    ax1.set_ylim(670, 705)
    ax1.set_yticks([675, 685, 695, 700])
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=40, ha="right")
    ax1.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)

    # Dòng kẻ ô lưới đồng bộ với các file plot-*.py
    ax1.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax1.set_axisbelow(False)

    # Chú thích Legend: định dạng và vị trí ở góc dưới bên phải theo plot-ratio.py
    legend1 = ax1.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#d9d9d9",
        fontsize=9.5,
        handlelength=2.2,
        borderpad=0.6,
        labelspacing=0.4,
    )
    legend1.set_zorder(5)

    # ══════════════════════════════════════════════════════════
    # ── Đồ thị (b): Largest-component length share (%) ─────────
    # ══════════════════════════════════════════════════════════
    ax2.plot(
        x_indices,
        df["share_pct"],
        color=color_share,
        linewidth=2.2,
        marker="o",
        markersize=6.0,
        markerfacecolor=color_share,
        markeredgecolor=color_share,
        zorder=3,
    )

    # Chú thích các mốc: Initial (0), Crossing (7), T-Junction (8), Dangling Lines (11)
    # Đã bỏ Duplicate Vertices, tất cả số liệu căn giữa chính xác trên marker
    for idx in annot_steps:
        val = df["share_pct"].iloc[idx]
        ax2.annotate(
            f"{val:.2f}%",
            (idx, val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2c3e50",
            zorder=5,
        )

    # Tiêu đề & nhãn trục lấy định dạng theo plot-spike.py
    ax2.set_title("(b) Largest-component length share (%)", fontsize=12.0, pad=10)
    ax2.set_xlabel("Processing step", fontsize=11, labelpad=8)
    ax2.set_ylabel("Largest-component length share (%)", fontsize=11, labelpad=8)

    # Giới hạn trục & ticks
    ax2.set_xlim(-0.6, len(df) - 0.4)
    ax2.set_ylim(95.5, 100.5)
    ax2.set_yticks([96, 97, 98, 99, 100])
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=40, ha="right")
    ax2.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)

    # Dòng kẻ ô lưới đồng bộ với các file plot-*.py
    ax2.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax2.set_axisbelow(False)

    plt.tight_layout()

    # Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_atopo_length_comparison.png"
    out_pdf = save_dir / "fig_atopo_length_comparison.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ so sánh chiều dài thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


def plot_length_share_combined(
    data: pd.DataFrame | list = None,
    excel_path: Path = EXCEL_PATH,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ tích hợp một panel với 2 trục Y (twinx):
      • Trục trái: Chiều dài mạng lưới (km) - Total length & Length of largest component
      • Trục phải: Tỷ lệ thành phần liên thông lớn nhất (%) - Largest-component length share
      • Bỏ tiêu đề (title), số liệu chiều dài ở trên marker, tỷ lệ ở dưới marker.
    """
    if data is not None:
        if isinstance(data, list):
            df = parse_network_report(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Tham số data phải là list hoặc pd.DataFrame.")
    else:
        df = load_atopo_network_data(excel_path)

    if df.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ.")
        return

    x_indices = np.arange(len(df))
    x_labels = df["step"].tolist()

    color_total = "#1f77b4"
    color_largest = "#157a74"
    color_share = "#b86e1e"

    fig, ax1 = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax2 = ax1.twinx()

    # 1. Trục bên trái: Chiều dài (km)
    line1 = ax1.plot(
        x_indices,
        df["tot_len_km"],
        color=color_total,
        linewidth=2.2,
        marker="s",
        markersize=5.5,
        markerfacecolor=color_total,
        markeredgecolor=color_total,
        label="Total length",
        zorder=3,
    )
    line2 = ax1.plot(
        x_indices,
        df["lg_len_km"],
        color=color_largest,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor=color_largest,
        markeredgecolor=color_largest,
        label="Length of largest component",
        zorder=3,
    )

    # 2. Trục bên phải: Tỷ lệ (%)
    line3 = ax2.plot(
        x_indices,
        df["share_pct"],
        color=color_share,
        linewidth=2.0,
        linestyle="--",
        marker="^",
        markersize=5.0,
        markerfacecolor=color_share,
        markeredgecolor=color_share,
        label="Largest-component length share",
        zorder=4,
    )

    # Cân đối tỷ lệ hai trục để các đường và mốc số liệu hài hòa
    y1_min, y1_max = 665, 708
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_yticks([670, 680, 690, 700])

    y2_min = y1_min / 7.0
    y2_max = y1_max / 7.0
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_yticks([95, 96, 97, 98, 99, 100])

    crossing_idx = df.index[df["step"] == "Crossing"].tolist()
    crossing_idx = crossing_idx[0] if crossing_idx else 7

    tjunction_idx = df.index[df["step"] == "T-Junction"].tolist()
    tjunction_idx = tjunction_idx[0] if tjunction_idx else 8

    annot_steps = [0, crossing_idx, tjunction_idx, len(df) - 1]

    # Chú thích số liệu: chiều dài ở trên marker, tỷ lệ ở dưới marker
    for idx in annot_steps:
        tot_val = df["tot_len_km"].iloc[idx]
        lg_val = df["lg_len_km"].iloc[idx]
        share_val = df["share_pct"].iloc[idx]

        # Total length: chiều dài ở trên
        ax1.annotate(
            f"{tot_val:.2f}",
            (idx, tot_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2c3e50",
            zorder=5,
        )

        # Largest component length & Share %
        if idx == len(df) - 1:
            # Tại mốc cuối Dangling Lines: đưa 698.52 xuống dưới để không bị cấn vào 699.80, tỷ lệ 99.82% ở dưới tiếp
            ax1.annotate(
                f"{lg_val:.2f}",
                (idx, lg_val),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2c3e50",
                zorder=5,
            )
            ax2.annotate(
                f"{share_val:.2f}%",
                (idx, share_val),
                textcoords="offset points",
                xytext=(0, -26),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2c3e50",
                zorder=5,
            )
        else:
            # Các mốc khác: chiều dài ở trên marker, tỷ lệ ở dưới marker
            ax1.annotate(
                f"{lg_val:.2f}",
                (idx, lg_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2c3e50",
                zorder=5,
            )
            ax2.annotate(
                f"{share_val:.2f}%",
                (idx, share_val),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2c3e50",
                zorder=5,
            )

    ax1.set_xlabel("Processing step", fontsize=11, labelpad=8)
    ax1.set_ylabel("Network length (km)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Largest-component length share (%)", fontsize=11, labelpad=8)

    ax1.set_xlim(-0.6, len(df) - 0.4)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=40, ha="right")
    ax1.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)
    ax2.tick_params(axis="y", which="major", labelsize=9.5, length=4, width=0.8)

    ax1.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=1)
    ax1.set_axisbelow(False)
    ax2.grid(False)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(
        lines,
        labels,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#d9d9d9",
        fontsize=9.5,
        handlelength=2.2,
        borderpad=0.6,
        labelspacing=0.4,
    )
    legend.set_zorder(5)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_atopo_length_share_combined.png"
    out_pdf = save_dir / "fig_atopo_length_share_combined.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ tích hợp chiều dài & tỷ lệ thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_length_comparison(show=False)
    plot_length_share_combined(show=False)
