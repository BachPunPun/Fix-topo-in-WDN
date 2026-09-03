import networkx as nx
import geopandas as gpd

COORD_PRECISION = 6


def build_graph(gdf: gpd.GeoDataFrame) -> nx.MultiGraph:
    """
    Chuyển GeoDataFrame (LineString) thành đồ thị networkx MultiGraph.

    Args:
        gdf: GeoDataFrame chứa geometry dạng LineString

    Returns:
        nx.MultiGraph với node = tọa độ (x, y), edge chứa attributes
    """
    G = nx.MultiGraph()

    valid_gdf = gdf[gdf.geometry.notna() & (gdf.geometry.type == 'LineString')].copy()
    if valid_gdf.empty:
        return G

    def find_col(*names):
        norm = {c.lower().strip(): c for c in valid_gdf.columns}
        return next((norm[n.lower()] for n in names if n.lower() in norm), None)

    mat_col = find_col('MATERIAL', 'vatlieu', 'chatlieu')
    dia_col = find_col('DIAMETER', 'duongkinh', 'dn', 'size')
    oid_col = find_col('OBJECTID', 'fid', 'id')

    def col_values(col, default):
        return valid_gdf[col].values if col else [default] * len(valid_gdf)

    def make_edge(idx, geom, mat, dia, length, oid):
        coords = geom.coords
        if len(coords) < 2:
            return None
        u = round(coords[0][0], COORD_PRECISION), round(coords[0][1], COORD_PRECISION)
        v = round(coords[-1][0], COORD_PRECISION), round(coords[-1][1], COORD_PRECISION)
        return u, v, {'idx': idx, 'objectid': oid, 'material': mat, 'diameter': dia, 'length': length}

    edges = filter(None, map(make_edge,
        valid_gdf.index.values,
        valid_gdf.geometry.values,
        col_values(mat_col, 'Unknown'),
        col_values(dia_col, 0),
        valid_gdf.geometry.length.values,
        col_values(oid_col, valid_gdf.index.values),
    ))

    G.add_edges_from(edges)
    return G
