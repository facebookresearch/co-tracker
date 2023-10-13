# coding: utf-8

from copy import deepcopy

from supervisely.geometry.constants import X, Y, Z, \
    POSITION, ROTATION, DIMENTIONS, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely.geometry.geometry import Geometry


class Vector3d:
    """
    This is a class for creating and using Vector3d objects for Cuboid3d class objects
    """
    def __init__(self, x, y, z):
        """
        :param x: int
        :param y: int
        :param z: int
        """
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """
        """
        return self._x

    @property
    def y(self):
        """
        """
        return self._y

    @property
    def z(self):
        """
        """
        return self._z

    def to_json(self):
        """
        The function to_json convert Vector3d class object to json format(dict)
        :return: Vector3d in json format
        """
        return {X: self.x, Y: self.y, Z: self.z}

    @classmethod
    def from_json(cls, data):
        """
        The function from_json convert Vector3d from json format(dict) to Vector3d class object.
        :param data: Vector3d in json format(dict)
        :return: Vector3d class object
        """
        x = data[X]
        y = data[Y]
        z = data[Z]
        return cls(x, y, z)

    def clone(self):
        """
        """
        return deepcopy(self)


class Cuboid3d(Geometry):
    """
    This is a class for creating and using Cuboid3d objects for Labels
    """
    @staticmethod
    def geometry_name():
        """
        """
        return 'cuboid_3d'

    def __init__(self, position: Vector3d, rotation: Vector3d, dimensions: Vector3d,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):

        """

        :param position: Vector3d class object
        :param rotation: Vector3d class object
        :param dimensions: Vector3d class object
        """         
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        
        if type(position) is not Vector3d:
            raise TypeError("\"position\" param has to be of type {!r}".format(type(Vector3d)))
        if type(rotation) is not Vector3d:
            raise TypeError("\"rotation\" param has to be of type {!r}".format(type(Vector3d)))
        if type(dimensions) is not Vector3d:
            raise TypeError("\"dimensions\" param has to be of type {!r}".format(type(Vector3d)))

        self._position = position
        self._rotation = rotation
        self._dimensions = dimensions

    @property
    def position(self):
        """
        """
        return self._position.clone()

    @property
    def rotation(self):
        """
        """
        return self._rotation.clone()

    @property
    def dimensions(self):
        """
        """
        return self._dimensions.clone()

    def to_json(self):
        """
        The function to_json convert Cuboid3d class object to json format(dict)
        :return: Cuboid3d in json format
        """
        res = {POSITION: self.position.to_json(),
                ROTATION: self.rotation.to_json(),
                DIMENTIONS: self.dimensions.to_json()}

        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        The function from_json convert Cuboid3d from json format(dict) to Cuboid3d class object.
        :param data: Cuboid3d in json format(dict)
        :return: Cuboid3d class object
        """
        position = Vector3d.from_json(data[POSITION])
        rotation = Vector3d.from_json(data[ROTATION])
        dimentions = Vector3d.from_json(data[DIMENTIONS])

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(position, rotation, dimentions, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
