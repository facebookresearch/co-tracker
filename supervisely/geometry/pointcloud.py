# coding: utf-8

from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID, INDICES
from supervisely.geometry.geometry import Geometry

# pointcloud mask (segmentation)
class Pointcloud(Geometry):
    """
    """

    @staticmethod
    def geometry_name():
        """
        """
        return 'point_cloud'

    def __init__(self, indices, sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        if type(indices) is not list:
            raise TypeError("\"indices\" param has to be of type {!r}".format(type(list)))

        self._indices = indices

    @property
    def indices(self):
        """
        """
        return self._indices.copy()

    def to_json(self):
        """
        """
        res = {INDICES: self.indices}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        """
        indices = data[INDICES]

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(indices, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
