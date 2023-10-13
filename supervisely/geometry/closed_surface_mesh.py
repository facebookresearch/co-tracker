from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import (
    EXTERIOR,
    INTERIOR,
    POINTS,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
)


class ClosedSurfaceMesh(Geometry):
    """
    """

    @staticmethod
    def geometry_name():
        """
        """
        return "closed_surface_mesh"

    def draw(self, bitmap, color, thickness=1, config=None):
        """
        """
        raise NotImplementedError('Method "draw" is unavailable for this geometry')

    def draw_contour(self, bitmap, color, thickness=1, config=None):
        """
        """
        raise NotImplementedError(
            'Method "draw_contour" is unavailable for this geometry'
        )

    def convert(self, new_geometry, contour_radius=0, approx_epsilon=None):
        """
        """
        raise NotImplementedError('Method "convert" is unavailable for this geometry')

    def to_json(self):
        """
        """
        res = {}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        """
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)

        return cls(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
