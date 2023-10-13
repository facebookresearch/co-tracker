# coding: utf-8
from supervisely.geometry.constants import EXTERIOR, INTERIOR, POINTS


def _is_2d_typed_coords_valid(coords, individual_coord_types):
    """
    """
    return isinstance(coords, (list, tuple)) and all(
        len(point) == 2 and
        all(isinstance(coord, individual_coord_types) for coord in point) for point in coords)


def _is_2d_numeric_coords_valid(coords) -> bool:
    """
    Float-type coordinates will be deprecated soon.
    Args:
        coords:  list or tuple of 2 numbers.
    Returns:
        True if validation successful, False otherwise.
    """
    return _is_2d_typed_coords_valid(coords, (int, float))


def is_2d_int_coords_valid(coords) -> bool:
    """
    Args:
        coords:  list (or tuple) of 2 integers.
    Returns:
        True if validation successful, False otherwise.
    """
    return _is_2d_typed_coords_valid(coords, (int,))


def validate_geometry_points_fields(json_obj: dict) -> None:
    """
    Validate json geometry points container structure, which presented as python dict.
    :param json_obj: example:
        {
            "points": {
                "exterior": [[1,2], [10, 20]],
                "interior": [[2,3]]
            }
        }
    :return: None
    """
    if POINTS not in json_obj:
        raise ValueError('Input data must contain {} field.'.format(POINTS))

    points_obj = json_obj[POINTS]
    if not isinstance(points_obj, dict):
        raise TypeError('Input data field "{}" must be dict object.'.format(POINTS))

    if EXTERIOR not in points_obj or INTERIOR not in points_obj:
        raise ValueError('"{}" field must contain {} and {} fields.'.format(POINTS, EXTERIOR, INTERIOR))

    if not _is_2d_numeric_coords_valid(points_obj[EXTERIOR]):
        raise TypeError('{} field must be a list of 2 numbers lists.'.format(EXTERIOR))

    interior = points_obj[INTERIOR]
    if not isinstance(interior, list) or not all(
            _is_2d_numeric_coords_valid(interior_component) for interior_component in interior):
        raise TypeError('{} field must be a list of lists of 2 numbers lists.'.format(INTERIOR))
