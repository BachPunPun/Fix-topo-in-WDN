# -*- coding: utf-8 -*-
"""
plot-ratio.py
Trực quan hóa phân tích độ nhạy của ngưỡng Overlap Ratio Threshold
Dữ liệu nguồn: result/parameter-analysis/ratio.xlsx
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
RATIO_EXCEL = OUTPUT_DIR / "ratio.xlsx"


def load_ratio_data(excel_path: Path = RATIO_EXCEL) -> pd.DataFrame:
    """Đọc dữ liệu từ file Excel ratio.xlsx và đếm số lượng DELETE, CUT theo từng ngưỡng."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    records = []

    for sheet in xls.sheet_names:
        try:
            ratio_val = float(sheet)
        except ValueError:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        if "Status" in df.columns:
            status_counts = df["Status"].astype(str).str.upper().value_counts()
            delete_cnt = status_counts.get("DELETE", 0)
            cut_cnt = status_counts.get("CUT", 0)
        else:
            delete_cnt = 0
            cut_cnt = 0

        records.append({
            "ratio": ratio_val,
            "delete": delete_cnt,
            "cut": cut_cnt,
            "total": delete_cnt + cut_cnt,
        })

    result_df = pd.DataFrame(records).sort_values(by="ratio").reset_index(drop=True)
    return result_df


def plot_overlap_ratio_sensitivity(
    excel_path: Path = RATIO_EXCEL,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ đồ thị Sensitivity of overlap ratio threshold: classification stability.
    Chỉ hiển thị vùng kết quả ổn định từ 0.8 trở lên (Stable plateau >= 0.8).
    """
    df = load_ratio_data(excel_path)
    if df.empty:
        print("❌ Không có dữ liệu để vẽ đồ thị.")
        return

    # Thiết lập kích thước và tỷ lệ đồ thị chuẩn bài báo
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)

    # 1. Vẽ vùng Stable Decision Region (từ ngưỡng 0.8 đến hết dải dữ liệu trên trục x)
    plateau_start = 0.80
    plateau_end = float(df["ratio"].max())  # 0.95
    ax.axvspan(
        plateau_start,
        plateau_end,
        color="#e8f1f8",
        alpha=0.9,
        zorder=1,
    )
    # Đặt chữ "Stable decision region" vào chính giữa vùng ổn định
    ax.text(
        (plateau_start + plateau_end) / 2,
        27.0,
        "Stable decision region",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#2c3e50",
        zorder=4,
    )

    # 2. Vẽ đường Delete (Nét liền, màu xanh dương, marker tròn)
    color_delete = "#1f77b4"
    ax.plot(
        df["ratio"],
        df["delete"],
        color=color_delete,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markerfacecolor=color_delete,
        markeredgecolor=color_delete,
        label="Delete decision",
        zorder=3,
    )

    # 3. Vẽ đường Cut (Nét liền, màu cam, marker vuông)
    color_cut = "#ff7f0e"
    ax.plot(
        df["ratio"],
        df["cut"],
        color=color_cut,
        linewidth=2.2,
        marker="s",
        markersize=6.5,
        markerfacecolor=color_cut,
        markeredgecolor=color_cut,
        label="Cut decision",
        zorder=3,
    )

    # 4. Hiển thị nhãn giá trị số lượng tại các điểm chuyển tiếp và ổn định chính
    # Delete annotations
    ax.annotate("25", (0.5, 25), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=color_delete, fontweight="bold")
    ax.annotate("25", (0.65, 25), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=color_delete)
    ax.annotate("23", (0.7, 23), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.0, color=color_delete)
    ax.annotate("22", (0.75, 22), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.0, color=color_delete)
    ax.annotate("20", (0.8, 20), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=color_delete, fontweight="bold")
    ax.annotate("20", (0.95, 20), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color=color_delete, fontweight="bold")

    # Cut annotations
    ax.annotate("29", (0.5, 29), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=color_cut, fontweight="bold")
    ax.annotate("29", (0.65, 29), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=color_cut)
    ax.annotate("31", (0.7, 31), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.0, color=color_cut)
    ax.annotate("32", (0.75, 32), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.0, color=color_cut)
    ax.annotate("34", (0.8, 34), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=color_cut, fontweight="bold")
    ax.annotate("34", (0.95, 34), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=color_cut, fontweight="bold")

    # 5. Cấu hình nhãn trục
    ax.set_xlabel("Overlap ratio threshold", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of corrected errors (count)", fontsize=11, labelpad=8)

    # 6. Cấu hình giới hạn trục và bước tick (Y: 15->40 bước 5, X: 0.5->1.0 bước 0.05)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(15, 40)

    ax.set_xticks(np.arange(0.5, 1.01, 0.05))
    ax.set_yticks(np.arange(15, 45, 5))

    # Định dạng ticks
    ax.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=0.8)

    # 7. Grid nền mờ nhẹ
    ax.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax.set_axisbelow(False)

    # 8. Chú thích Legend ở góc dưới bên phải (lower right)
    legend = ax.legend(
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

    # 8. Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_ratio_sensitivity.png"
    out_pdf = save_dir / "fig_ratio_sensitivity.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo và lưu đồ thị thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_overlap_ratio_sensitivity(show=False)
