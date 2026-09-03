# -*- coding: utf-8 -*-
"""
plot-tolerance.py
Trực quan hóa phân tích độ nhạy của tham số Geometric Tolerance
Dữ liệu nguồn: result/parameter-analysis/tolerance-*.xlsx
Bao gồm 3 biểu đồ:
  1. Biểu đồ đường quỹ đạo phát hiện lỗi (Trajectory of Topological Defect Detection)
  2. Biểu đồ nhiệt ma trận độ nhạy dạng 2x2 (Sensitivity Heatmap Matrix 2x2 across Baselines τ ∈ [0.10, 0.25])
  3. Biểu đồ ghép 2-trong-1 kết hợp quỹ đạo và ma trận nhiệt (Combined Sensitivity Analysis)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Cấu hình danh sách các loại lỗi topology và bảng màu chuẩn bài báo
DEFECT_CONFIG = {
    "Crossing": {
        "file": OUTPUT_DIR / "tolerance-crossing.xlsx",
        "color": "#20558a",
        "marker": "o",
    },
    "Dangling Lines": {
        "file": OUTPUT_DIR / "tolerance-dangling-lines.xlsx",
        "color": "#d95f02",
        "marker": "o",
    },
    "Overshoot": {
        "file": OUTPUT_DIR / "tolerance-overshoot.xlsx",
        "color": "#736fad",
        "marker": "o",
    },
    "Sliver Gap Lines": {
        "file": OUTPUT_DIR / "tolerance-sliver-gap-lines.xlsx",
        "color": "#e7298a",
        "marker": "o",
    },
    "T-Junction": {
        "file": OUTPUT_DIR / "tolerance-t-junction.xlsx",
        "color": "#0fa370",
        "marker": "o",
    },
    "Undershoot": {
        "file": OUTPUT_DIR / "tolerance-undershoot.xlsx",
        "color": "#e5a500",
        "marker": "o",
    },
}

CATEGORIES = list(DEFECT_CONFIG.keys())
BASELINE_TOLERANCE = 0.10


def load_tolerance_data(output_dir: Path = OUTPUT_DIR):
    """
    Đọc dữ liệu từ các file Excel tolerance-*.xlsx và tính toán:
      1. df_counts: Ma trận số lượng lỗi phát hiện theo từng ngưỡng tolerance
      2. df_deviation: Ma trận độ lệch tương đối (%) so với mốc baseline τ = 0.10
    """
    counts_data = {}

    for category, cfg in DEFECT_CONFIG.items():
        file_path = cfg["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file kết quả: {file_path}")

        xls = pd.ExcelFile(file_path)
        cat_counts = {}
        for sheet_name in xls.sheet_names:
            try:
                tol_val = float(sheet_name)
            except ValueError:
                continue

            df = pd.read_excel(xls, sheet_name=sheet_name)
            cat_counts[tol_val] = len(df)

        counts_data[category] = cat_counts

    # Tạo DataFrame số lượng lỗi, sắp xếp cột theo thứ tự tolerance tăng dần
    df_counts = pd.DataFrame(counts_data).T
    df_counts = df_counts.reindex(CATEGORIES)
    df_counts = df_counts.reindex(sorted(df_counts.columns), axis=1)

    # Tính độ lệch tương đối (%) so với baseline τ = 0.10
    if BASELINE_TOLERANCE not in df_counts.columns:
        raise ValueError(f"Không tìm thấy cột baseline τ = {BASELINE_TOLERANCE} trong dữ liệu.")

    baseline_series = df_counts[BASELINE_TOLERANCE]
    df_deviation = df_counts.apply(
        lambda col: ((col - baseline_series) / baseline_series.replace(0, np.nan)) * 100,
        axis=0
    ).fillna(0.0)

    return df_counts, df_deviation


# ══════════════════════════════════════════════════════════════
# 1. Biểu đồ 1: Quỹ đạo phát hiện lỗi (Trajectory Plot)
# ══════════════════════════════════════════════════════════════
def plot_tolerance_trajectory(
    output_dir: Path = OUTPUT_DIR,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ Figure 1: Trajectory of Topological Defect Detection across Tolerance Continuum.
    Đánh dấu vùng ổn định tối ưu Stable response region từ 0.1 đến 0.25.
    """
    df_counts, df_dev = load_tolerance_data(output_dir)
    if df_counts.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ.")
        return

    tol_values = list(df_counts.columns)
    tol_labels = [f"{val:g}" for val in tol_values]
    x_indices = np.arange(len(tol_labels))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # 1. Vẽ vùng Stable response region (từ 0.1 đến 0.25 - không có đường viền nét đứt)
    plateau_start_val = 0.10
    plateau_end_val = 0.25
    p_start_idx = tol_values.index(plateau_start_val)  # index 4
    p_end_idx = tol_values.index(plateau_end_val)      # index 7

    ax.axvspan(
        p_start_idx,
        p_end_idx,
        color="#e8f1f8",
        alpha=0.9,
        zorder=1,
    )

    # Đặt nhãn "Stable response region" ở đỉnh vùng ổn định
    ax.text(
        (p_start_idx + p_end_idx) / 2,
        235,
        "Stable response region",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#2c3e50",
        zorder=4,
    )

    # 2. Vẽ đường quỹ đạo phát hiện lỗi cho từng danh mục topology
    for cat in CATEGORIES:
        cfg = DEFECT_CONFIG[cat]
        ax.plot(
            x_indices,
            df_counts.loc[cat],
            color=cfg["color"],
            linewidth=2.2,
            marker=cfg["marker"],
            markersize=6.0,
            markerfacecolor=cfg["color"],
            markeredgecolor=cfg["color"],
            label=cat,
            zorder=3,
        )

    # 3. Cấu hình nhãn trục
    ax.set_xlabel("Tolerance threshold (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Total of corrected errors (count)", fontsize=11, labelpad=8)

    # 4. Cấu hình giới hạn trục và bước tick
    ax.set_xlim(-0.4, len(tol_labels) - 0.6)
    ax.set_ylim(-10, 255)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(tol_labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(np.arange(0, 260, 50))
    ax.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=1.0, color="black")

    # 5. Grid nền mờ nhẹ và viền khung đồ thị (trục x, y là line màu đen)
    ax.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax.set_axisbelow(False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # 6. Chú thích Legend (chỉ chứa 6 danh mục lỗi) ở góc trên bên trái
    legend = ax.legend(
        loc="upper left",
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

    # 7. Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_tolerance_sensitivity_trajectory.png"
    out_pdf = save_dir / "fig_tolerance_sensitivity_trajectory.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ tích lũy quỹ đạo thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 2. Biểu đồ 2: Ma trận nhiệt độ nhạy 2x2 (Sensitivity Heatmap Matrix 2x2)
# ══════════════════════════════════════════════════════════════
def plot_tolerance_heatmap(
    output_dir: Path = OUTPUT_DIR,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ biểu đồ Sensitivity Heatmap Matrix dạng subplot 2x2:
      (a) Baseline ε = 0.10
      (b) Baseline ε = 0.15
      (c) Baseline ε = 0.20
      (d) Baseline ε = 0.25
    Biểu diễn độ nhạy của từng loại lỗi theo ma trận nhiệt hai chiều tương ứng với từng mốc baseline trong vùng ổn định.
    """
    df_counts, _ = load_tolerance_data(output_dir)
    if df_counts.empty:
        print("❌ Không có dữ liệu để vẽ biểu đồ nhiệt.")
        return

    baselines = [0.10, 0.15, 0.20, 0.25]
    titles = [
        "(a) Baseline ε = 0.10",
        "(b) Baseline ε = 0.15",
        "(c) Baseline ε = 0.20",
        "(d) Baseline ε = 0.25",
    ]

    tol_values = list(df_counts.columns)
    tol_labels = [f"{val:g}" for val in tol_values]

    # 1. Tạo bảng màu phân kỳ (Diverging Colormap) và chuẩn hóa dải giá trị chung
    c_neg = ["#9ab2c7", "#b5c6d5", "#d4dfe8", "#edf2f7", "#f6f6f7"]
    c_pos = ["#f6f6f7", "#fae8e8", "#e2adad", "#c46464", "#9e2a2b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("tolerance_div_cmap", c_neg[:-1] + c_pos, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-82.0, vcenter=0.0, vmax=175.0)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), dpi=300)

    im_last = None
    for idx, (b_val, title) in enumerate(zip(baselines, titles)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        if b_val not in df_counts.columns:
            continue

        b_series = df_counts[b_val]
        df_dev = df_counts.apply(
            lambda c: ((c - b_series) / b_series.replace(0, np.nan)) * 100,
            axis=0,
        ).fillna(0.0)
        matrix_vals = df_dev.values

        # 2. Vẽ ma trận nhiệt cho từng subplot
        im = ax.imshow(matrix_vals, cmap=cmap, norm=norm, aspect="auto")
        im_last = im

        # 3. Cấu hình trục x và trục y
        ax.set_xticks(np.arange(len(tol_labels)))
        ax.set_yticks(np.arange(len(CATEGORIES)))
        ax.set_xticklabels(tol_labels, fontsize=8.5)

        if col == 0:
            ax.set_yticklabels(CATEGORIES, fontsize=9.0)
            ax.set_ylabel("Topological Error Category", fontsize=10.0, labelpad=6)
        else:
            ax.set_yticklabels([])

        # 4. Hiển thị giá trị phần trăm tương đối trong từng ô
        for i in range(len(CATEGORIES)):
            for j in range(len(tol_labels)):
                val = matrix_vals[i, j]
                text_color = "#ffffff" if (val >= 90.0 or val <= -78.0) else "#2c3e50"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8.0, color=text_color)

        # 5. Kẻ viền trắng phân cách giữa các ô
        for i in range(len(CATEGORIES) + 1):
            ax.axhline(i - 0.5, color="white", linewidth=1.8)
        for j in range(len(tol_labels) + 1):
            ax.axvline(j - 0.5, color="white", linewidth=1.8)

        # 6. Ẩn viền khung ngoài của axes
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis="both", which="both", length=0)
        ax.set_title(title, fontsize=11.5, pad=7)

        if row == 1:
            ax.set_xlabel("Tolerance threshold (m)", fontsize=9.5, labelpad=6)

    # 7. Căn chỉnh khoảng cách và thêm thanh màu Colorbar chung
    fig.subplots_adjust(right=0.91, hspace=0.26, wspace=0.08, top=0.94, bottom=0.08, left=0.10)
    if im_last is not None:
        cbar_ax = fig.add_axes([0.925, 0.12, 0.014, 0.76])
        cbar = fig.colorbar(im_last, cax=cbar_ax)
        cbar.set_label("Relative Deviation from Baseline (%)", fontsize=9.5, labelpad=8)
        cbar.set_ticks([-50, 0, 50, 100, 150])
        cbar.ax.tick_params(labelsize=8.5)
        cbar.outline.set_visible(False)

    # 8. Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_tolerance_sensitivity_heatmap.png"
    out_pdf = save_dir / "fig_tolerance_sensitivity_heatmap.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ ma trận nhiệt 2x2 thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 3. Biểu đồ 3: Ghép cả 2 biểu đồ vào cùng 1 khung hình (2x1)
# ══════════════════════════════════════════════════════════════
def plot_tolerance_combined(
    output_dir: Path = OUTPUT_DIR,
    save_dir: Path = PLOTS_DIR,
    show: bool = False,
):
    """
    Vẽ hình tổng hợp 2 trong 1 gồm:
      (a) Biểu đồ quỹ đạo phát hiện lỗi across tolerance continuum (vùng ổn định từ 0.1 đến 0.25, trục x/y màu đen)
      (b) Biểu đồ ma trận nhiệt độ lệch tương đối (%) từ baseline ε = 0.10
    """
    df_counts, df_dev = load_tolerance_data(output_dir)
    if df_counts.empty or df_dev.empty:
        return

    tol_values = list(df_counts.columns)
    tol_labels = [f"{val:g}" for val in tol_values]
    x_indices = np.arange(len(tol_labels))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 9.2),
        dpi=300,
        gridspec_kw={"height_ratios": [1.2, 1.0]}
    )

    # ── Đồ thị (a): Quỹ đạo phát hiện lỗi ──
    plateau_start_val = 0.10
    plateau_end_val = 0.25
    p_start_idx = tol_values.index(plateau_start_val)  # index 4
    p_end_idx = tol_values.index(plateau_end_val)      # index 7

    # Vùng ổn định (không có đường nét đứt)
    ax1.axvspan(
        p_start_idx,
        p_end_idx,
        color="#e8f1f8",
        alpha=0.9,
        zorder=1,
    )

    # Đặt nhãn "Stable response region" ở đỉnh vùng ổn định
    ax1.text(
        (p_start_idx + p_end_idx) / 2,
        235,
        "Stable response region",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#2c3e50",
        zorder=4,
    )

    for cat in CATEGORIES:
        cfg = DEFECT_CONFIG[cat]
        ax1.plot(
            x_indices,
            df_counts.loc[cat],
            color=cfg["color"],
            linewidth=2.2,
            marker=cfg["marker"],
            markersize=6.0,
            markerfacecolor=cfg["color"],
            markeredgecolor=cfg["color"],
            label=cat,
            zorder=3,
        )

    ax1.set_title("(a) Trajectory of topological defect detection across tolerance continuum", fontsize=12.0, pad=10)
    ax1.set_ylabel("Total of corrected errors (count)", fontsize=11, labelpad=8)
    ax1.set_xlim(-0.4, len(tol_labels) - 0.6)
    ax1.set_ylim(-10, 255)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(tol_labels, rotation=45, ha="right", fontsize=9.5)
    ax1.set_yticks(np.arange(0, 260, 50))
    ax1.tick_params(axis="both", which="major", labelsize=9.5, length=4, width=1.0, color="black")
    ax1.grid(True, linestyle="-", linewidth=0.8, color="#e5e5e5", alpha=0.7, zorder=2)
    ax1.set_axisbelow(False)

    # Trục x, y là line màu đen
    for spine in ax1.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # Chú thích Legend (chỉ chứa 6 danh mục lỗi) ở góc trên bên trái
    legend = ax1.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#d9d9d9",
        fontsize=9.5,
        handlelength=2.2,
        borderpad=0.6,
        labelspacing=0.4,
    )
    legend.set_zorder(5)

    # ── Đồ thị (b): Ma trận nhiệt độ nhạy ──
    c_neg = ["#9ab2c7", "#b5c6d5", "#d4dfe8", "#edf2f7", "#f6f6f7"]
    c_pos = ["#f6f6f7", "#fae8e8", "#e2adad", "#c46464", "#9e2a2b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("tolerance_div_cmap", c_neg[:-1] + c_pos, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-81.0, vcenter=0.0, vmax=174.2)

    matrix_vals = df_dev.values
    im = ax2.imshow(matrix_vals, cmap=cmap, norm=norm, aspect="auto")

    ax2.set_xticks(np.arange(len(tol_labels)))
    ax2.set_yticks(np.arange(len(CATEGORIES)))
    ax2.set_xticklabels(tol_labels, fontsize=9.5)
    ax2.set_yticklabels(CATEGORIES, fontsize=9.5)

    for i in range(len(CATEGORIES)):
        for j in range(len(tol_labels)):
            val = matrix_vals[i, j]
            text_color = "#ffffff" if (val >= 95.0 or val <= -80.0) else "#2c3e50"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9.0, color=text_color)

    for i in range(len(CATEGORIES) + 1):
        ax2.axhline(i - 0.5, color="white", linewidth=2.0)
    for j in range(len(tol_labels) + 1):
        ax2.axvline(j - 0.5, color="white", linewidth=2.0)

    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax2.tick_params(axis="both", which="both", length=0)
    ax2.set_title("(b) Sensitivity heatmap matrix of relative deviations (%) from baseline ε = 0.10", fontsize=12.0, pad=10)
    ax2.set_xlabel("Tolerance threshold (m)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Topological Error Category", fontsize=11, labelpad=8)

    cbar = fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.022, pad=0.025)
    cbar.set_label("Relative Deviation from Baseline ε=0.10 (%)", fontsize=9.5, labelpad=8)
    cbar.set_ticks([-50, 0, 50, 100, 150])
    cbar.ax.tick_params(labelsize=9.0)
    cbar.outline.set_visible(False)

    plt.tight_layout()

    # Lưu kết quả ra file PNG và PDF
    save_dir.mkdir(parents=True, exist_ok=True)
    out_png = save_dir / "fig_tolerance_sensitivity_combined.png"
    out_pdf = save_dir / "fig_tolerance_sensitivity_combined.pdf"

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Đã tạo biểu đồ ghép 2-trong-1 thành công:\n  • PNG: {out_png}\n  • PDF: {out_pdf}")

    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 4. Hàm thực thi vẽ toàn bộ biểu đồ Tolerance
# ══════════════════════════════════════════════════════════════
def plot_all():
    """Chạy toàn bộ các hàm vẽ đồ thị Geometric Tolerance."""
    plot_tolerance_trajectory(show=False)
    plot_tolerance_heatmap(show=False)
    plot_tolerance_combined(show=False)


if __name__ == "__main__":
    plot_all()
