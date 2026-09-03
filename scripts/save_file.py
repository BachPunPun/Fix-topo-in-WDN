from pathlib import Path
import warnings
import pyogrio
import geopandas as gpd


def save_shp(
    gdf: gpd.GeoDataFrame,
    path: str | Path = "",
    fallback_epsg: int = 9210,
) -> Path:
    """
    Lưu GeoDataFrame ra file .shp với tốc độ cao dùng pyogrio.

    Args:
        gdf:           GeoDataFrame cần lưu
        path:          Đường dẫn file đầu ra (.shp)
        fallback_epsg: EPSG dùng khi CRS gốc không xác định được (mặc định 3405)

    Returns:
        Path của file đã lưu

    Example:
        save_shp(gdf, "output/roads.shp")
        save_shp(gdf, "output/roads.shp", fallback_epsg=4326)
    """
    if not path:
        raise ValueError("Vui lòng truyền đường dẫn file, ví dụ: save_shp(gdf, 'output/roads.shp')")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Truncate tên cột > 10 ký tự (giới hạn của ESRI Shapefile)
    long_cols = [c for c in gdf.columns if c != "geometry" and len(c) > 10]
    if long_cols:
        rename_map = {}
        used_names = set(c for c in gdf.columns if len(c) <= 10)
        for c in long_cols:
            base = c[:10]
            counter = 1
            new_name = base
            while new_name in used_names:
                new_name = f"{base[:8]}_{counter}"
                counter += 1
            rename_map[c] = new_name
            used_names.add(new_name)
        gdf = gdf.rename(columns=rename_map)

    # Xử lý CRS không tương thích với WKT1_GDAL
    if gdf.crs is not None:
        try:
            gdf.crs.to_wkt("WKT1_GDAL")
        except Exception:
            epsg = gdf.crs.to_epsg()
            if epsg:
                gdf = gdf.set_crs(f"EPSG:{epsg}", allow_override=True)
            else:
                gdf = gdf.set_crs(f"EPSG:{fallback_epsg}", allow_override=True)

    pyogrio.write_dataframe(gdf, str(path), driver="ESRI Shapefile")
    return path