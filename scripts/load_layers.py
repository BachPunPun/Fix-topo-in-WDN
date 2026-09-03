# -*- coding: utf-8 -*-
"""
Nạp nhiều lớp ống vào CHUNG một GeoDataFrame để phân tích topology toàn mạng,
rồi TÁCH kết quả trở lại từng lớp khi xuất.

Đường dẫn các lớp được khai báo tập trung trong atopo/config.py (INPUT_LAYERS).

Cách dùng:
    from load_layers import load_layers, split_layers

    gdf, meta = load_layers()          # gộp tất cả lớp + gắn cột SRC_LAYER
    ...                                # chạy pipeline trên gdf chung
    parts = split_layers(gdf, meta)    # {tên_lớp: GeoDataFrame} đã bỏ cột đánh dấu
"""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Cho phép import config (ở thư mục gốc dự án) dù được gọi từ bất kỳ script nào.
# scripts/load_layers.py  →  parents[1] = thư mục gốc (chứa config.py).
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config
from read_file import read_shp
from domains import decode_domain_fields


def load_layers(layers: list | None = None, src_field: str | None = None):
    """
    Nạp & gộp các lớp ống thành một GeoDataFrame chung.

    Args:
        layers:    Danh sách lớp [{"name", "path"}]. Mặc định = config.INPUT_LAYERS.
        src_field: Tên cột đánh dấu lớp nguồn. Mặc định = config.SRC_FIELD.

    Returns:
        (gdf, meta)
          - gdf:  GeoDataFrame gộp, index reset 0..N-1, có thêm cột `src_field`.
          - meta: dict {tên_lớp: [cột gốc của lớp đó]} — dùng để tách lại khi xuất.
    """
    layers = layers if layers is not None else config.INPUT_LAYERS
    src_field = src_field or config.SRC_FIELD

    parts, meta, base_crs = [], {}, None
    for lyr in layers:
        name, path = lyr["name"], Path(lyr["path"])
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy lớp '{name}': {path}")

        g = read_shp(path)

        # Dịch mã domain → nhãn (VATLIEU: 10 → HDPE). Chỉ tác động khi cột đang là
        # số và có .gdb; ngược lại giữ nguyên. Chạy TRƯỚC khi chốt meta vì hàm này
        # sửa cột tại chỗ, không thêm cột mới.
        g = decode_domain_fields(
            g,
            layer=name,
            gdb_path=getattr(config, "GDB_PATH", None),
            fields=getattr(config, "DOMAIN_DECODE_FIELDS", None),
            null_codes=getattr(config, "DOMAIN_NULL_CODES", ()),
        )

        # Lưu lại danh sách cột gốc (trừ geometry & cột đánh dấu) để tách lại sau này
        meta[name] = [c for c in g.columns if c not in ("geometry", src_field)]

        if base_crs is None:
            base_crs = g.crs
        elif g.crs is not None and base_crs is not None and g.crs != base_crs:
            g = g.to_crs(base_crs)

        g[src_field] = name
        parts.append(g)

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        crs=base_crs,
    )
    combined = combined.set_geometry("geometry")
    return combined, meta


def split_layers(
    gdf: gpd.GeoDataFrame,
    meta: dict,
    src_field: str | None = None,
    keep_extra: tuple = ("ID", "Len"),
):
    """
    Tách GeoDataFrame chung trở lại từng lớp theo cột đánh dấu nguồn.

    Mỗi lớp chỉ giữ các cột gốc của nó + các cột bổ sung hữu ích (ID, Len) do
    pipeline tạo ra; cột đánh dấu nguồn và các cột riêng của lớp khác bị loại bỏ.

    Args:
        gdf:        GeoDataFrame chung sau khi chạy pipeline.
        meta:       dict {tên_lớp: [cột gốc]} trả về từ load_layers().
        src_field:  Tên cột đánh dấu lớp nguồn. Mặc định = config.SRC_FIELD.
        keep_extra: Các cột bổ sung muốn giữ nếu tồn tại (mặc định ID, Len).

    Returns:
        dict {tên_lớp: GeoDataFrame} đã reset index và bỏ cột đánh dấu.
    """
    src_field = src_field or config.SRC_FIELD
    out = {}
    for name, orig_cols in meta.items():
        sub = gdf[gdf[src_field] == name].copy()

        keep = [c for c in orig_cols if c in sub.columns]
        for ex in keep_extra:
            if ex in sub.columns and ex not in keep:
                keep.append(ex)
        if "geometry" not in keep:
            keep.append("geometry")

        out[name] = gpd.GeoDataFrame(
            sub[keep].reset_index(drop=True), geometry="geometry", crs=gdf.crs
        )
    return out
