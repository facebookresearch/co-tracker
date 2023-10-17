# coding: utf-8

from __future__ import annotations
from copy import deepcopy
from re import L
from typing import List, Union
import uuid
from supervisely.volume_annotation.volume_figure import VolumeFigure

from supervisely.project.project_meta import ProjectMeta
from supervisely._utils import take_with_default
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.volume_annotation.plane import Plane
from supervisely.volume_annotation.constants import (
    NAME,
    TAGS,
    OBJECTS,
    KEY,
    VOLUME_ID,
    VOLUME_META,
    PLANES,
    SPATIAL_FIGURES,
)


class VolumeAnnotation:
    """
    VolumeAnnotation for a single volume. :class:`VolumeAnnotation<VolumeAnnotation>` object is immutable.

    :param volume_meta: Metadata of the volume.
    :type volume_meta: dict
    :param objects: VolumeObjectCollection object.
    :type objects: VolumeObjectCollection, optional
    :param plane_sagittal: Sagittal plane of the volume.
    :type plane_sagittal: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
    :param plane_coronal: Coronal plane of the volume.
    :type plane_coronal: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
    :param plane_axial: Axial plane of the volume.
    :type plane_axial: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
    :param tags: VolumeTagCollection object.
    :type tags: VolumeTagCollection, optional
    :param spatial_figures: List of spatial figures associated with the volume.
    :type spatial_figures: List[VolumeFigure], optional
    :param key: UUID object.
    :type key: UUID, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Simple VolumeAnnotation example
        path = "/home/admin/work/volumes/vol_01.nrrd"
        volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
        volume_ann = sly.VolumeAnnotation(volume_meta)
        print(volume_ann.to_json())
        # Output: {
        # {
        #     "key": "56107223943346e5900fc256b8dcd7f0",
        #     "objects": [],
        #     "planes": [
        #         { "name": "sagittal", "normal": { "x": 1, "y": 0, "z": 0 }, "slices": [] },
        #         { "name": "coronal", "normal": { "x": 0, "y": 1, "z": 0 }, "slices": [] },
        #         { "name": "axial", "normal": { "x": 0, "y": 0, "z": 1 }, "slices": [] }
        #     ],
        #     "spatialFigures": [],
        #     "tags": [],
        #     "volumeMeta": {
        #         "ACS": "RAS",
        #         "channelsCount": 1,
        #         "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
        #         "directions": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #         "intensity": { "max": 3071.0, "min": -3024.0 },
        #         "origin": [-194.238403081894, -217.5384061336518, -347.7500000000001],
        #         "rescaleIntercept": 0,
        #         "rescaleSlope": 1,
        #         "spacing": [0.7617189884185793, 0.7617189884185793, 2.5],
        #         "windowCenter": 23.5,
        #         "windowWidth": 6095.0
        #     }
        # }

        # More complex VolumeAnnotation example

        path = "/home/admin/work/volumes/vol_01.nrrd"
        volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
        # VolumeObjectCollection
        obj_class = sly.ObjClass('brain', sly.Rectangle)
        volume_obj = sly.VolumeObject(obj_class)
        objects = sly.VolumeObjectCollection([volume_obj])
        # VolumeTagCollection
        brain_meta = sly.TagMeta('brain_tag', sly.TagValueType.ANY_STRING)
        from supervisely.volume_annotation.volume_tag import VolumeTag
        vol_tag = VolumeTag(brain_meta, value='human')
        from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
        volume_tags = VolumeTagCollection([vol_tag])

        volume_ann = sly.VolumeAnnotation(volume_meta, objects, volume_tags)
        print(volume_ann.to_json())
        # Output:
        # {
        #     "key": "4d4bb69e6fcd40e1a1cb076c07769903",
        #     "objects": [
        #         {
        #         "classTitle": "brain",
        #         "key": "22e1082a17f74279b00eed0bfb0ba11d",
        #         "tags": []
        #         }
        #     ],
        #     "planes": [
        #         { "name": "sagittal", "normal": { "x": 1, "y": 0, "z": 0 }, "slices": [] },
        #         { "name": "coronal", "normal": { "x": 0, "y": 1, "z": 0 }, "slices": [] },
        #         { "name": "axial", "normal": { "x": 0, "y": 0, "z": 1 }, "slices": [] }
        #     ],
        #     "spatialFigures": [],
        #     "tags": [
        #         {
        #         "key": "b9de6631d328441796119b4b0039fc61",
        #         "name": "brain_tag",
        #         "value": "human"
        #         }
        #     ],
        #     "volumeMeta": {
        #         "ACS": "RAS",
        #         "channelsCount": 1,
        #         "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
        #         "directions": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #         "intensity": { "max": 3071.0, "min": -3024.0 },
        #         "origin": [-194.238403081894, -217.5384061336518, -347.7500000000001],
        #         "rescaleIntercept": 0,
        #         "rescaleSlope": 1,
        #         "spacing": [0.7617189884185793, 0.7617189884185793, 2.5],
        #         "windowCenter": 23.5,
        #         "windowWidth": 6095.0
        #     }
        # }
    """

    def __init__(
        self,
        volume_meta,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        spatial_figures=None,
        key=None,
    ):
        self._volume_meta = volume_meta
        self._tags = take_with_default(tags, VolumeTagCollection())
        self._objects = take_with_default(objects, VolumeObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

        self._plane_sagittal = take_with_default(
            plane_sagittal,
            Plane(Plane.SAGITTAL, volume_meta=volume_meta),
        )
        self._plane_coronal = take_with_default(
            plane_coronal,
            Plane(Plane.CORONAL, volume_meta=volume_meta),
        )
        self._plane_axial = take_with_default(
            plane_axial,
            Plane(Plane.AXIAL, volume_meta=volume_meta),
        )

        self._spatial_figures = take_with_default(spatial_figures, [])
        self.validate_figures_bounds()

    @property
    def volume_meta(self) -> dict:
        """
        Volume meta data.

        :returns: Sagittal plane of the volume.
        :rtype: dict

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            volume_meta = vol_ann.volume_meta
        """

        return deepcopy(self._volume_meta)

    @property
    def plane_sagittal(self) -> Plane:
        """
        Sagital plane of the volume.

        :returns: Sagittal plane of the volume.
        :rtype: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            plane_sagittal = vol_ann.plane_sagittal
        """

        return self._plane_sagittal

    @property
    def plane_coronal(self) -> Plane:
        """
        Coronal plane of the volume.

        :returns: Coronal plane of the volume.
        :rtype: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            plane_coronal = vol_ann.plane_coronal
        """

        return self._plane_coronal

    @property
    def plane_axial(self) -> Plane:
        """
        Axial plane of the volume.

        :returns: Axial plane of the volume.
        :rtype: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            plane_axial = vol_ann.plane_axial
        """

        return self._plane_axial

    @property
    def objects(self) -> VolumeObjectCollection:
        """
        VolumeAnnotation objects.

        :return: VolumeObjectCollection object
        :rtype: :class:`VolumeObjectCollection<supervisely.volume_annotation.volume_object_collection.VolumeObjectCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            path = "/Users/Downloads/volumes/Demo volumes_ds1_CTChest.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)

            # VolumeObjectCollection
            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            objects = sly.VolumeObjectCollection([volume_obj_heart])
            volume_ann = sly.VolumeAnnotation(volume_meta, objects)

            print(volume_ann.objects.to_json())
            # Output: [
            #     {
            #         "key": "2b5d70baa5a74d06a525b950b5f2b756",
            #         "classTitle": "heart",
            #         "tags": []
            #     }
            # ]
        """

        return self._objects

    @property
    def tags(self) -> VolumeTagCollection:
        """
        VolumeTag objects.

        :returns: VolumeTagCollection
        :rtype: :class:`VolumeTagCollection<supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection>`

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            tags = vol_ann.tags
        """

        return self._tags

    @property
    def spatial_figures(self) -> List[VolumeFigure]:
        """
        Get a list of spatial figures.

        :returns: List of spatial figures from VolumeAnnotation object.
        :rtype: List[VolumeFigure]

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            spatial_figures = vol_ann.spatial_figures
        """

        return self._spatial_figures

    @property
    def figures(self) -> List[VolumeFigure]:
        """
        VolumeFigure objects.

        :returns: List of VolumeFigure objects from VolumeAnnotation object.
        :rtype: List[VolumeFigure]

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            figures = vol_ann.figures
        """

        all_figures = []
        for plane in [self.plane_sagittal, self.plane_coronal, self.plane_axial]:
            all_figures.extend(plane.figures)
        return all_figures

    def key(self) -> str:
        """
        Volume annotation key value.

        :returns: Key value of VolumeAnnotation object.
        :rtype: str

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)
            key = vol_ann.key()
        """
        return self._key

    def validate_figures_bounds(self):
        """
        Checks if all slices in each plane contains figures.

        :raises: :class:`OutOfImageBoundsException<supervisely.video_annotation.video_figure.OutOfImageBoundsException>`, if figure is out of slices images bounds
        :return: None
        :rtype: :class:`NoneType`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            plane_axial = sly.Plane(sly.Plane.AXIAL, frames, volume_meta=volume_meta)
            volume_ann = sly.VolumeAnnotation(volume_meta, objects, plane_axial=plane_axial)
            volume_ann.validate_figures_bounds()
        """

        self.plane_sagittal.validate_figures_bounds()
        self.plane_coronal.validate_figures_bounds()
        self.plane_axial.validate_figures_bounds()

    def is_empty(self) -> bool:
        """
        Check whether volume annotation contains objects or tags, or not.

        :returns: True if volume annotation is empty, False otherwise.
        :rtype: :class:`bool`

        :Usage exmaple:

         .. code-block:: python

            import supervisely as sly
            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            vol_ann = sly.VolumeAnnotation(volume_meta)

            is_empty = vol_ann.is_empty()
        """

        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False

    def clone(
        self,
        volume_meta=None,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        spatial_figures=None,
    ):
        """
        Makes a copy of VolumeAnnotation with new fields, if fields are given, otherwise it will use fields of the original VolumeAnnotation.

        :param volume_meta: Metadata of the volume.
        :type volume_meta: dict
        :param objects: VolumeObjectCollection object.
        :type objects: VolumeObjectCollection, optional
        :param plane_sagittal: Sagittal plane of the volume.
        :type plane_sagittal: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
        :param plane_coronal: Coronal plane of the volume.
        :type plane_coronal: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
        :param plane_axial: Axial plane of the volume.
        :type plane_axial: :py:class:`Plane<supervisely.volume_annotation.plane.Plane>`, optional
        :param tags: VolumeTagCollection object.
        :type tags: VolumeTagCollection, optional
        :param spatial_figures: List of spatial figures associated with the volume.
        :type spatial_figures: List[VolumeFigure], optional

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            volume_ann = sly.VolumeAnnotation(volume_meta)

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            video_obj_heart = sly.VolumeObject(obj_class_heart)
            new_objects = sly.VolumeObjectCollection([volume_obj_heart])
            new_volume_ann = volume_ann.clone(objects=new_objects)
        """

        return VolumeAnnotation(
            volume_meta=take_with_default(volume_meta, self.volume_meta),
            objects=take_with_default(objects, self.objects),
            plane_sagittal=take_with_default(plane_sagittal, self.plane_sagittal),
            plane_coronal=take_with_default(plane_coronal, self.plane_coronal),
            plane_axial=take_with_default(plane_axial, self.plane_axial),
            tags=take_with_default(tags, self.tags),
            spatial_figures=take_with_default(spatial_figures, self.spatial_figures),
        )

    @classmethod
    def from_json(cls, data: dict, project_meta: ProjectMeta, key_id_map: KeyIdMap = None):
        """
        Convert a json dict to VolumeAnnotation.

        :param data: Volume annotation in json format as a dict.
        :type data: dict
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: VolumeAnnotation object
        :rtype: :class:`VolumeAnnotation<VolumeAnnotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.video_annotation.key_id_map import KeyIdMap

            meta = sly.ProjectMeta()
            key_id_map = KeyIdMap()

            ann_json = {
                "key": "56107223943346e5900fc256b8dcd7f0",
                "objects": [],
                "planes": [
                    { "name": "sagittal", "normal": { "x": 1, "y": 0, "z": 0 }, "slices": [] },
                    { "name": "coronal", "normal": { "x": 0, "y": 1, "z": 0 }, "slices": [] },
                    { "name": "axial", "normal": { "x": 0, "y": 0, "z": 1 }, "slices": [] }
                ],
                "spatialFigures": [],
                "tags": [],
                "volumeMeta": {
                    "ACS": "RAS",
                    "channelsCount": 1,
                    "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
                    "directions": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "intensity": { "max": 3071.0, "min": -3024.0 },
                    "origin": [-194.238403081894, -217.5384061336518, -347.7500000000001],
                    "rescaleIntercept": 0,
                    "rescaleSlope": 1,
                    "spacing": [0.7617189884185793, 0.7617189884185793, 2.5],
                    "windowCenter": 23.5,
                    "windowWidth": 6095.0
                }
            }


            ann = sly.VolumeAnnotation.from_json(ann_json, meta, key_id_map)
        """

        volume_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()
        if key_id_map is not None:
            key_id_map.add_video(volume_key, data.get(VOLUME_ID, None))

        volume_meta = data[VOLUME_META]

        tags = VolumeTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = VolumeObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)

        plane_sagittal = None
        plane_coronal = None
        plane_axial = None
        for plane_json in data[PLANES]:
            if plane_json[NAME] == Plane.SAGITTAL:
                plane_sagittal = Plane.from_json(
                    plane_json,
                    Plane.SAGITTAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            elif plane_json[NAME] == Plane.CORONAL:
                plane_coronal = Plane.from_json(
                    plane_json,
                    Plane.CORONAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            elif plane_json[NAME] == Plane.AXIAL:
                plane_axial = Plane.from_json(
                    plane_json,
                    Plane.AXIAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            else:
                raise RuntimeError(f"Unknown plane name {plane_json[NAME]}")

        spatial_figures = []
        for figure_json in data.get(SPATIAL_FIGURES, []):
            figure = VolumeFigure.from_json(
                figure_json,
                objects,
                plane_name=None,
                slice_index=None,
                key_id_map=key_id_map,
            )
            spatial_figures.append(figure)

        return cls(
            volume_meta=volume_meta,
            objects=objects,
            plane_sagittal=plane_sagittal,
            plane_coronal=plane_coronal,
            plane_axial=plane_axial,
            tags=tags,
            spatial_figures=spatial_figures,
            key=volume_key,
        )

    def to_json(self, key_id_map: KeyIdMap = None) -> dict:
        """
        Convert the VolumeAnnotation to a json dict.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: Volume annotation in json format as a dict.
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            from supervisely.video_annotation.key_id_map import KeyIdMap

            path = "/home/admin/work/volumes/vol_01.nrrd"
            volume, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            volume_ann = sly.VolumeAnnotation(volume_meta)

            print(volume_ann.to_json())
            # Output: {
            # {
            #     "key": "56107223943346e5900fc256b8dcd7f0",
            #     "objects": [],
            #     "planes": [
            #         { "name": "sagittal", "normal": { "x": 1, "y": 0, "z": 0 }, "slices": [] },
            #         { "name": "coronal", "normal": { "x": 0, "y": 1, "z": 0 }, "slices": [] },
            #         { "name": "axial", "normal": { "x": 0, "y": 0, "z": 1 }, "slices": [] }
            #     ],
            #     "spatialFigures": [],
            #     "tags": [],
            #     "volumeMeta": {
            #         "ACS": "RAS",
            #         "channelsCount": 1,
            #         "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
            #         "directions": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            #         "intensity": { "max": 3071.0, "min": -3024.0 },
            #         "origin": [-194.238403081894, -217.5384061336518, -347.7500000000001],
            #         "rescaleIntercept": 0,
            #         "rescaleSlope": 1,
            #         "spacing": [0.7617189884185793, 0.7617189884185793, 2.5],
            #         "windowCenter": 23.5,
            #         "windowWidth": 6095.0
            #     }
            # }
        """

        res_json = {
            VOLUME_META: self.volume_meta,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            PLANES: [
                self.plane_sagittal.to_json(),
                self.plane_coronal.to_json(),
                self.plane_axial.to_json(),
            ],
            SPATIAL_FIGURES: [figure.to_json(key_id_map) for figure in self.spatial_figures],
        }

        if key_id_map is not None:
            volume_id = key_id_map.get_video_id(self.key())
            if volume_id is not None:
                res_json[VOLUME_ID] = volume_id

        return res_json

    def add_objects(
        self, objects: Union[List[VolumeObject], VolumeObjectCollection]
    ) -> VolumeAnnotation:
        """
        Add new objects to a VolumeAnnotation object.

        :param objects: New volume objects.
        :type objects: List[VolumeObject] or VolumeObjectCollection
        :return: A VolumeAnnotation object containing the original and new volume objects.
        :rtype: VolumeAnnotation
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            path = "/vol_01.nrrd"
            _, volume_meta = sly.volume.read_nrrd_serie_volume_np(path)
            volume_ann = sly.VolumeAnnotation(volume_meta)
            obj_class_heart = sly.ObjClass('heart', sly.Mask3D)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_ann = volume_ann.add_objects([volume_obj_heart])

        """

        sf_figures = []
        for volume_object in objects:
            if volume_object.obj_class.geometry_type in (Mask3D, AnyGeometry):
                if isinstance(volume_object.figure.geometry, Mask3D):
                    sf_figures.append(volume_object.figure)

        collection = self.objects.add_items(objects)
        new_ann = self.clone(objects=collection)
        new_ann.spatial_figures.extend(sf_figures)
        return new_ann
