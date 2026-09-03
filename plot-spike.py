# -*- coding: utf-8 -*-
"""
plot-spike.py
Trực quan hóa phân tích độ nhạy của ngưỡng góc Spike Lines (Sharp Angle Threshold)
Dữ liệu nguồn: result/parameter-analysis/spike.xlsx
Bao gồm 2 biểu đồ:
  1. Biểu đồ đường tích lũy (Cumulative Sensitivity & Stable response Region)
  2. Biểu đồ cột phân bố gia tăng (Marginal Error Distribution)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thêm đường dẫn thư mục scripts, function và thư mục gốc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "function"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import setup_console
except ImportError:
    pass

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "result" / "parameter-analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"
SPIKE_EXCEL = OUTPUT_DIR / "spike.xlsx"


def load_spike_data(excel_path: Path = SPIKE_EXCEL) -> pd.DataFrame:
    """Đọc dữ liệu từ file Excel spike.xlsx và tổng hợp số lượng lỗi theo từng ngưỡng góc."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    records = []

    for sheet in xls.sheet_names:
        try:
            angle_val = float(sheet)
        except ValueError:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        total_cnt = len(df)

        records.append({
            "angle_str": sheet,
            "angle": angle_val,
            "label": f"{sheet}°",
            "count": total_cnt,
        })

    result_df = pd.DataFrame(records).sort_values(by="angle").reset_index(drop=True)
    # Tính số lượng lỗi phát hiện mới tại mỗi bước ngưỡng (marginal count)
    result_df["marginal"] = result_df["count"].diff().fillna(result_df["count"].iloc[0]).astype(int)
    return result_df


# ══════════════════════════════════════════════════════════════
# 1. Biểu đồ 1: Đường nhạy cảm tích lũy (Cumulative Sensitivity)
# ══════════════════════════════════════════════════════════════
def plot_spike_cumulative(
    excel_path: Path = SPIKE_EXCEL,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ đường độ nhạy của ngưỡng góc Spike Lines (Cumulative Sensitivity Curve).
    Đánh dấu vùng quyết định ổn định từ 7° đến 11.25°.
    """
    df = load_spike_data(excel_path)
    if df.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ.")
        return

    x_indices = np.arange(len(df))
    x_labels = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # 1. Vùng Stable response Region (từ 7° - index 6 đến 11.25° - index 11)
    stable_start_idx = 6  # Tương ứng với mốc 7°
    stable_end_idx = len(df) - 1  # Tương ứng với mốc 11.25°

    ax.axvspan(
        stable_start_idx,
        stable_end_idx,
        color="#e8f1f8",
        alpha=0.9,
        zorder=1,
    )
    # Đặt chữ "Stable response region" vào chính giữa vùng ổn định
    ax.text(
        (stable_start_idx + stable_end_idx) / 2,
        51.5,
        "Stable response region",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#2c3e50",
        zorder=4,
    )

    # 2. Vẽ đường tích lũy lỗi phát hiện (Nét liền, màu xanh dương, marker tròn)
    color_primary = "#1f77b4"
    ax.plot(
        x_indices,
        df["count"],
        color=color_primary,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markerfacecolor=color_primary,
        markeredgecolor=color_primary,
        zorder=3,
    )

    # 3. Hiển thị nhãn giá trị số lượng tại các điểm quan trọng
    annotations = {
        0: (50, (0, -14)),    # 1°
        1: (54, (0, 8)),      # 2° (đưa lên trên)
        2: (55, (0, 8)),      # 3° (bổ sung)
        3: (56, (0, 8)),      # 4°
        5: (56, (0, 8)),      # 6° (bổ sung)
        6: (57, (0, 8)),      # 7°
        11: (57, (0, 8)),     # 11.25°
    }
    for idx, (val, offset) in annotations.items():
        if idx < len(df):
            is_bold = idx in (0, 1, 2, 3, 5, 6, 11)
            ax.annotate(
                f"{val}",
                (idx, val),
                textcoords="offset points",
                xytext=offset,
                ha="center",
                fontsize=8.5,
                color=color_primary,
                fontweight="bold" if is_bold else "normal",
                zorder=5,
            )

    # 4. Cấu hình nhãn trục
    ax.set_xlabel("Sharp angle threshold (°)", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of automatically detected errors (count)", fontsize=11, labelpad=8)

    # 5. Cấu hình giới hạn trục và bước tick (Cột Y từ 45 đến 60, bước nhảy 2.5)
    ax.set_xlim(-0.4, len(df) - 0.6)
    ax.set_ylim(45, 60)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    
    y_ticks = np.arange(45, 62.5, 2.5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:g}" for y in y_ticks])

    # Định dạng ticks
    ax.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)

    # 6. Grid nền mờ nhẹ
    ax.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax.set_axisbelow(False)

    plt.tight_layout()

    # 8. Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_spike_sensitivity_cumulative.png"
    out_pdf = save_dir / "fig_spike_sensitivity_cumulative.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ tích lũy thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 2. Biểu đồ 2: Cột phân bố gia tăng (Marginal Error Distribution)
# ══════════════════════════════════════════════════════════════
def plot_spike_marginal(
    excel_path: Path = SPIKE_EXCEL,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ cột phân bố lỗi mới phát hiện theo từng khoảng ngưỡng góc Spike Lines.
    Cột Y tối đa là 55.
    """
    df = load_spike_data(excel_path)
    if df.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ.")
        return

    x_indices = np.arange(len(df))
    x_labels = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # 1. Vẽ các cột phân bố (Màu xanh dương đồng bộ)
    color_primary = "#1f77b4"
    bars = ax.bar(
        x_indices,
        df["marginal"],
        width=0.48,
        color=color_primary,
        edgecolor=color_primary,
        linewidth=1.0,
        zorder=3,
    )

    # 2. Đính kèm giá trị lên trên mỗi cột (chỉ hiện cho các cột > 0)
    for bar, val in zip(bars, df["marginal"]):
        if val > 0:
            ax.annotate(
                f"{val}",
                (bar.get_x() + bar.get_width() / 2, val),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9.0,
                fontweight="bold",
                color=color_primary,
                zorder=5,
            )

    # 3. Cấu hình nhãn trục
    ax.set_xlabel("Sharp angle threshold (°)", fontsize=11, labelpad=8)
    ax.set_ylabel("Newly detected spike-line errors (count)", fontsize=11, labelpad=8)

    # 4. Cấu hình giới hạn trục và bước tick (Cột Y tối đa 55)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0, 55)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 55])

    # Định dạng ticks
    ax.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)

    # 5. Grid nền mờ nhẹ
    ax.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax.set_axisbelow(False)

    plt.tight_layout()

    # 6. Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_spike_marginal_distribution.png"
    out_pdf = save_dir / "fig_spike_marginal_distribution.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ phân bố gia tăng thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 3. Biểu đồ 3: Ghép cả 2 biểu đồ vào cùng 1 khung hình (2x1)
# ══════════════════════════════════════════════════════════════
def plot_spike_combined(
    excel_path: Path = SPIKE_EXCEL,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ hình tổng hợp 2 trong 1 gồm:
      (a) Biểu đồ tích lũy & vùng ổn định (Y từ 45 đến 60, bước nhảy 2.5)
      (b) Biểu đồ phân bố lỗi gia tăng (Y tối đa 55)
    """
    df = load_spike_data(excel_path)
    if df.empty:
        return

    x_indices = np.arange(len(df))
    x_labels = df["label"].tolist()
    color_primary = "#1f77b4"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.2), dpi=300)

    # ── Đồ thị (a): Tích lũy ──
    stable_start_idx = 6
    stable_end_idx = len(df) - 1

    ax1.axvspan(stable_start_idx, stable_end_idx, color="#e8f1f8", alpha=0.9, zorder=1)
    ax1.text((stable_start_idx + stable_end_idx) / 2, 51.5, "Stable response region", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#2c3e50", zorder=4)

    ax1.plot(x_indices, df["count"], color=color_primary, linewidth=2.0, marker="o", markersize=5.5, label="Total detected errors", zorder=3)

    annotations = {
        0: (50, (0, -13)),
        1: (54, (0, 7)),
        2: (55, (0, 7)),
        3: (56, (0, 7)),
        5: (56, (0, 7)),
        6: (57, (0, 7)),
        11: (57, (0, 7)),
    }
    for idx, (val, offset) in annotations.items():
        if idx < len(df):
            ax1.annotate(f"{val}", (idx, val), textcoords="offset points", xytext=offset, ha="center", fontsize=8.5, color=color_primary, fontweight="bold", zorder=5)

    ax1.set_title("(a) Total number of detected spike-line errors across threshold values", fontsize=12.0, pad=10)
    ax1.set_xlabel("Sharp angle threshold (°)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Total detected spike line errors (count)", fontsize=11, labelpad=8)
    ax1.set_xlim(-0.5, len(df) - 0.5)
    ax1.set_ylim(45, 60)
    
    y_ticks_1 = np.arange(45, 62.5, 2.5)
    ax1.set_yticks(y_ticks_1)
    ax1.set_yticklabels([f"{y:g}" for y in y_ticks_1])
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)
    
    ax1.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax1.set_axisbelow(False)

    # ── Đồ thị (b): Phân bố gia tăng ──
    bars = ax2.bar(x_indices, df["marginal"], width=0.48, color=color_primary, edgecolor=color_primary, zorder=3)
    for bar, val in zip(bars, df["marginal"]):
        if val > 0:
            ax2.annotate(f"{val}", (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9.0, fontweight="bold", color=color_primary, zorder=5)

    ax2.set_title("(b) Incremental number of spike-line errors detected at each threshold", fontsize=12.0, pad=10)
    ax2.set_xlabel("Sharp angle threshold (°)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Newly detected spike-line errors (count)", fontsize=11, labelpad=8)
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(0, 55)
    ax2.set_yticks([0, 10, 20, 30, 40, 50, 55])
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)
    ax2.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax2.set_axisbelow(False)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_spike_sensitivity_combined.png"
    out_pdf = save_dir / "fig_spike_sensitivity_combined.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ ghép 2-trong-1 thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


def plot_all():
    """Chạy toàn bộ các hàm vẽ đồ thị Spike Lines."""
    plot_spike_cumulative(show=False)
    plot_spike_marginal(show=False)
    plot_spike_combined(show=False)


if __name__ == "__main__":
    plot_all()
