# coding: utf-8
import itertools


def _clear_dupe_start_end_boundary(coords_list):
    """
    """
    return coords_list if len(coords_list) < 2 or coords_list[0] != coords_list[-1] else coords_list[:-1]


def _clear_dupe_start_end_multi_polygon(multi_polygon):
    """
    """
    return [[_clear_dupe_start_end_boundary(boundary) for boundary in polygon] for polygon in multi_polygon]


def shapely_figure_to_coords_list(mp) -> list:
    """
    """
    mp_type = mp['type']
    if mp_type == 'MultiLineString':
        return mp['coordinates']
    elif mp_type == 'LineString':
        return [mp['coordinates']]
    elif mp_type == 'Polygon':
        return _clear_dupe_start_end_multi_polygon([mp['coordinates']])
    elif mp_type == 'GeometryCollection':
        # Here we get a list of figures from every recursive call, so need to unwrap it
        # into a flat list of all the figures.
        return list(itertools.chain.from_iterable(
            shapely_figure_to_coords_list(geom_obj) for geom_obj in mp['geometries']))
    elif mp_type == 'MultiPolygon':
        return _clear_dupe_start_end_multi_polygon(mp['coordinates'])
    else:
        return []
