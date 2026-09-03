import pyogrio
import geopandas as gpd
from pathlib import Path


def read_shp(
    path: str | Path,
    columns: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    where: str | None = None,
    max_features: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Đọc file .shp với tốc độ cao dùng pyogrio (GDAL binding).

    Args:
        path:         Đường dẫn tới file .shp
        columns:      Chỉ đọc các cột thuộc tính này (None = đọc tất cả)
        bbox:         Lọc theo bounding box (xmin, ymin, xmax, ymax)
        where:        Bộ lọc SQL kiểu OGR, VD: "province = 'HCM'"
        max_features: Giới hạn số feature đọc vào

    Returns:
        GeoDataFrame chứa geometry và attributes

    Example:
        gdf = read_shp("data/roads.shp")
        gdf = read_shp("data/roads.shp", columns=["name", "type"], bbox=(106.5, 10.5, 107.0, 11.0))
    """
    try:
        return pyogrio.read_dataframe(
            str(path),
            columns=columns,
            bbox=bbox,
            where=where,
            max_features=max_features,
            use_arrow=True,   # Dùng Apache Arrow → nhanh hơn ~2-3x khi load vào DataFrame
        )
    except (RuntimeError, ImportError):
        return pyogrio.read_dataframe(
            str(path),
            columns=columns,
            bbox=bbox,
            where=where,
            max_features=max_features,
            use_arrow=False,
        )