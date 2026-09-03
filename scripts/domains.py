# -*- coding: utf-8 -*-
"""
Giải mã coded-value domain của ArcGIS File Geodatabase.

Vì sao cần: shapefile không mang theo domain. Sau khi export từ .gdb, cột VATLIEU
chỉ còn mã số (10) thay vì nhãn (HDPE) — ArcGIS dịch mã sang nhãn lúc hiển thị,
còn shapefile thì không có thông tin đó. Module này đọc bảng domain thẳng từ .gdb
và dịch mã → nhãn.

Domain nằm trong bảng hệ thống GDB_Items dưới dạng XML. Mỗi feature class tự khai
báo field → DomainName riêng, nên PHẢI tra theo đúng lớp: DMVatLieuOngNuoc có
10=HDPE, trong khi DMVatLieuOngBe có 3=HDPE. Dùng chung một bảng mã sẽ dịch sai.

Cách dùng:
    from domains import decode_domain_fields
    gdf = decode_domain_fields(gdf, "OngTruyenDan", gdb_path, ["VATLIEU"])
"""

import warnings
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyogrio

SHP_FIELD_MAXLEN = 10  # ESRI Shapefile cắt tên cột về 10 ký tự


def _strip_ns(el):
    """Bỏ namespace để findtext() dùng được tên thẻ trần."""
    for e in el.iter():
        if "}" in e.tag:
            e.tag = e.tag.split("}", 1)[1]
    return el


def _norm_code(v):
    """Chuẩn hóa mã domain: ưu tiên int để khớp với cột số trong shapefile."""
    s = str(v).strip()
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


@lru_cache(maxsize=4)
def _read_gdb_items(gdb_path: str) -> tuple:
    """Đọc bảng hệ thống GDB_Items → tuple((Name, Definition), ...).

    Cần OPENFILEGDB_LIST_ALL_TABLES để GDAL cho phép mở bảng hệ thống.
    """
    pyogrio.set_gdal_config_options({"OPENFILEGDB_LIST_ALL_TABLES": "YES"})
    df = pyogrio.read_dataframe(gdb_path, layer="GDB_Items", read_geometry=False)
    return tuple(zip(df["Name"], df["Definition"]))


@lru_cache(maxsize=4)
def load_domains(gdb_path: str) -> dict:
    """Trả về {tên_domain: {mã: nhãn}} cho mọi coded-value domain trong .gdb."""
    out = {}
    for name, defn in _read_gdb_items(gdb_path):
        if not defn or "CodedValueDomain" not in str(defn)[:300]:
            continue  # bỏ qua range domain / định nghĩa khác
        root = _strip_ns(ET.fromstring(str(defn)))
        codes = {}
        for cv in root.iter("CodedValue"):
            code, label = cv.findtext("Code"), cv.findtext("Name")
            if code is not None and label is not None:
                codes[_norm_code(code)] = label
        if codes:
            out[str(name)] = codes
    return out


@lru_cache(maxsize=32)
def field_domain_map(gdb_path: str, layer: str) -> dict:
    """Trả về {tên_field: tên_domain} khai báo trong feature class `layer`."""
    for name, defn in _read_gdb_items(gdb_path):
        if str(name) != layer or not defn:
            continue
        root = _strip_ns(ET.fromstring(str(defn)))
        return {
            f.findtext("Name"): f.findtext("DomainName")
            for f in root.iter("GPFieldInfoEx")
            if f.findtext("DomainName")
        }
    return {}


def _match_column(gdf, field: str) -> str:
    """Tìm cột trong gdf ứng với `field` của .gdb (shapefile cắt tên còn 10 ký tự)."""
    if field in gdf.columns:
        return field
    truncated = field[:SHP_FIELD_MAXLEN]
    for c in gdf.columns:
        if c == truncated or c.upper() == truncated.upper():
            return c
    return ""


def decode_domain_fields(gdf, layer: str, gdb_path=None, fields=None, null_codes=()):
    """
    Dịch mã → nhãn cho các cột domain, thay đổi TẠI CHỖ trên chính cột đó.

    Chỉ đụng vào cột đang là kiểu số. Cột đã có nhãn dạng text (shapefile export
    kèm domain description) được giữ nguyên — nên hàm này an toàn khi chạy trên
    bộ dữ liệu hỗn hợp.

    Args:
        gdf:        GeoDataFrame vừa nạp từ .shp
        layer:      Tên feature class trong .gdb (khớp name ở config.INPUT_LAYERS)
        gdb_path:   Đường dẫn .gdb. None/không tồn tại → bỏ qua, giữ nguyên mã số.
        fields:     Danh sách field cần giải mã, vd ["VATLIEU"]. Rỗng → không làm gì.
        null_codes: Các mã coi như "không có giá trị" → dịch thành ô trống.
                    Shapefile không lưu được NULL cho cột số nên ArcGIS ghi thành 0.

    Returns:
        GeoDataFrame (cùng object, đã sửa cột nếu có giải mã).
    """
    if not fields:
        return gdf

    # Bước 1 — lọc trước các cột thật sự cần giải mã (tồn tại + đang là số)
    targets = []
    for field in fields:
        col = _match_column(gdf, field)
        if not col:
            continue
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            continue  # đã là nhãn text → không cần dịch
        targets.append((field, col))

    if not targets:
        return gdf

    # Bước 2 — cần .gdb mới dịch được; không có thì giữ nguyên mã số
    if gdb_path is None or not Path(gdb_path).exists():
        warnings.warn(
            f"[{layer}] Cần giải mã {[c for _, c in targets]} nhưng không tìm thấy "
            f".gdb ({gdb_path}); giữ nguyên mã số."
        )
        return gdf

    gdb_path = str(gdb_path)
    try:
        fmap = field_domain_map(gdb_path, layer)
        domains = load_domains(gdb_path)
    except Exception as e:
        warnings.warn(f"[{layer}] Không đọc được domain từ .gdb: {e}; giữ nguyên mã số.")
        return gdf

    if not fmap:
        warnings.warn(f"[{layer}] Không có feature class tên '{layer}' trong .gdb; giữ nguyên mã số.")
        return gdf

    null_set = {_norm_code(c) for c in (null_codes or ())}

    # Bước 3 — dịch từng cột theo domain khai báo cho ĐÚNG lớp này
    for field, col in targets:
        dname = fmap.get(field)
        if not dname:
            warnings.warn(f"[{layer}] Field '{field}' không gắn domain nào; giữ nguyên mã số.")
            continue
        codes = domains.get(dname)
        if not codes:
            warnings.warn(f"[{layer}] Domain '{dname}' không phải coded-value; giữ nguyên mã số.")
            continue

        unknown = set()

        def lookup(v):
            if pd.isna(v):
                return None
            code = _norm_code(v)
            if code in null_set:
                return None
            label = codes.get(code)
            if label is None:
                unknown.add(code)
                return str(code)  # mã lạ → giữ dạng chuỗi, không vứt dữ liệu đi
            return label

        mapped = gdf[col].map(lookup)
        if unknown:
            warnings.warn(
                f"[{layer}] {col}: mã không có trong domain '{dname}': "
                f"{sorted(unknown, key=str)} — giữ nguyên dạng chuỗi"
            )

        gdf[col] = mapped
        n_ok = int(mapped.notna().sum())
        n_empty = len(mapped) - n_ok
        msg = f"   → [{layer}] {col}: giải mã {n_ok}/{len(gdf)} theo domain '{dname}'"
        if n_empty:
            msg += f" ({n_empty} trống)"
        print(msg)

    return gdf
