# coding: utf-8

from supervisely.geometry.cuboid_3d import Vector3d
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID


class Point3d(Geometry):
    """
    """

    @staticmethod
    def geometry_name():
        """
        """
        return 'point_3d'

    def __init__(self, point: Vector3d,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        if type(point) is not Vector3d:
            raise TypeError("\"position\" param has to be of type {!r}".format(type(Vector3d)))

        self._point = point

    @property
    def x(self):
        """
        """
        return self._point._x

    @property
    def y(self):
        """
        """
        return self._point._y

    @property
    def z(self):
        """
        """
        return self._point._z

    def to_json(self):
        """
        """
        res = self._point.to_json()
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        """
        point = Vector3d.from_json(data)

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(point, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)