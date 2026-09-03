# -*- coding: utf-8 -*-
"""
Cấu hình trung tâm cho atopo — quản lý đường dẫn data ở MỘT nơi duy nhất.

Chỉnh sửa file này khi đổi bộ dữ liệu đầu vào, không cần sửa từng script.

Mô hình xử lý:
    - Nạp NHIỀU lớp ống (truyền dẫn + phân phối) vào chung 1 GeoDataFrame,
      gắn cột đánh dấu lớp nguồn (SRC_FIELD) để phân tích topology TRÊN TOÀN MẠNG
      (bắt đúng lỗi tại điểm nối giữa các lớp, báo cáo liên thông chính xác).
    - Cuối pipeline, TÁCH kết quả trở lại từng lớp và xuất ra file riêng.
"""

from pathlib import Path

# ── Thư mục gốc ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent                     # .../atopo
PROJECT_DIR = BASE_DIR.parent                          # .../NhaBe-GIS-WaterGEMS

# ── Thư mục dữ liệu đầu vào ───────────────────────────────────
DATA_DIR = BASE_DIR / "data"

# ── Thư mục kết quả (data đã chuẩn hóa) ───────────────────────
OUTPUT_DIR = BASE_DIR / "result"

# ── Tên lần chạy (tiền tố cho file output) ────────────────────
RUN_NAME = "GD_TD"

# ── Các lớp ống đầu vào ───────────────────────────────────────
# Mỗi lớp: name (dùng làm hậu tố file output) + path (.shp).
# Thêm/bớt lớp tại đây là đủ — pipeline tự gộp để phân tích chung rồi tách xuất riêng.
# LƯU Ý: `name` phải trùng tên feature class trong GDB_PATH thì mới tra được domain.
INPUT_LAYERS = [
    # {"name": "OTD_2601", "path": DATA_DIR / "TruyenDan" / "OTD_2601.shp"},
    {"name": "OPP_2510", "path": DATA_DIR / "GiaDinh" / "OPP_2510.shp"},
]

# ── Cột đánh dấu lớp nguồn ────────────────────────────────────
# ≤10 ký tự để không bị truncate khi lưu ESRI Shapefile.
SRC_FIELD = "SRC_LAYER"

# ── Giải mã coded-value domain ────────────────────────────────
# Shapefile export từ .gdb không mang theo domain: VATLIEU chỉ còn mã số (10)
# thay vì nhãn (HDPE) — ArcGIS dịch mã→nhãn khi hiển thị, shapefile thì không.
# Trỏ tới .gdb gốc để atopo tự dịch lúc nạp.
#
# Cơ chế: chỉ đụng vào cột đang là kiểu SỐ. Lớp nào đã có nhãn text sẵn thì bỏ qua,
# nên an toàn với bộ dữ liệu hỗn hợp. Không có .gdb → cảnh báo và giữ nguyên mã số,
# pipeline vẫn chạy bình thường.
GDB_PATH = None   # None nếu không có .gdb

# Các field cần giải mã. Domain tra theo TỪNG LỚP, vì cùng tên field nhưng khác lớp
# có thể trỏ domain khác nhau (DMVatLieuOngNuoc: 10=HDPE / DMVatLieuOngBe: 3=HDPE).
# Để rỗng [] để tắt hẳn.
DOMAIN_DECODE_FIELDS = ["VATLIEU", "TIEUCHUAN", "TINHTRANG", "CAPONG"]

# Mã coi như "không có giá trị" → dịch thành ô trống thay vì nhãn.
# Shapefile không lưu được NULL cho cột số nên ArcGIS export ô trống thành 0
# (vd OngPhanPhoi.TIEUCHUAN: 2294 NULL trong .gdb → 2507 số 0 trong .shp). Mọi domain
# ở đây đều đánh mã từ 1 nên 0 không bao giờ hợp lệ — kiểm tra lại nếu đổi bộ dữ liệu.
DOMAIN_NULL_CODES = [0]

# ── Tham số chung ─────────────────────────────────────────────
COORD_PRECISION = 6
