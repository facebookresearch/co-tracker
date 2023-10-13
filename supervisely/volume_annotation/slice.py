# coding: utf-8

from typing import Optional

from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.constants import FIGURES, INDEX


class Slice(Frame):
    """
    A class representing a single slice of a medical image.
    Slice object for :class:`VolumeAnnotation<supervisely.volume_annotation.volume_annotation.VolumeAnnotation>`. :class:`Frame<Frame>` object is immutable.

    :param index: Index of the Slice.
    :type index: int
    :param figures: List of :class:`VolumeFigures<supervisely.volume_annotation.volume_figure.VolumeFigure>`.
    :type figures: Optional[List[VolumeFigure]]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        frame_index = 7
        geometry = sly.Rectangle(0, 0, 100, 100)
        class_car = sly.ObjClass('car', sly.Rectangle)
        object_car = sly.VolumeObject(class_car)
        figure_car = sly.VolumeFigure(object_car, geometry, frame_index)

        frame = sly.Slice(frame_index, figures=[figure_car])
        print(frame.to_json())
        # Output: {
        #     "figures": [
        #         {
        #         "geometry": {
        #             "points": {
        #             "exterior": [
        #                 [0, 0],
        #                 [100, 100]
        #             ],
        #             "interior": []
        #             }
        #         },
        #         "geometryType": "rectangle",
        #         "key": "eb0ab5f772054f70b6a9f5b583a47287",
        #         "meta": {
        #             "normal": { "x": 0, "y": 0, "z": 1 },
        #             "planeName": "axial",
        #             "sliceIndex": 7
        #         },
        #         "objectKey": "dbd236a6a6f440139fd0299905fcc46e"
        #         }
        #     ],
        #     "index": 7
        # }

    """

    figure_type = VolumeFigure

    # @classmethod
    # def from_json(cls, data, objects, slices_count=None, key_id_map=None):
    #     raise NotImplementedError()
    #     _frame = super().from_json(data, objects, slices_count, key_id_map)
    #     return cls(index=_frame.index, figures=_frame.figures)

    @classmethod
    def from_json(
        cls,
        data: dict,
        objects: VolumeObjectCollection,
        plane_name: str,
        slices_count: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ):
        """
        Deserialize a `Slice` object from a JSON representation.

        :param data: The JSON representation of the `Slice`.
        :type data: dict
        :param objects: A collection of objects in volume.
        :type objects: VolumeObjectCollection
        :param plane_name: The name of the plane.
        :type plane_name: str
        :param slices_count: The total number of slices in the volume, if known.
        :type slices_count: Optional[int]
        :param key_id_map: A mapping of keys to IDs used for referencing objects.
        :type key_id_map: Optional[KeyIdMap]
        :return: The deserialized `Slice<Slice>` object.
        :rtype: Slice
        :raises ValueError: If the slice index is negative or greater than the total number of slices.

        :Usage example:

         .. code-block:: python

            import supervisely as sly
            slice_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            class_car = sly.ObjClass('car', sly.Rectangle)
            object_car = sly.VolumeObject(class_car)
            objects = sly.VolumeObjectCollection([object_car])
            figure_car = sly.VolumeFigure(object_car, geometry, sly.Plane.AXIAL, slice_index)

            slice = sly.Slice(slice_index, figures=[figure_car])
            slice_json = slice.to_json()

            new_slice = sly.Slice.from_json(slice_json, objects, sly.Plane.AXIAL)
        """

        index = data[INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        if slices_count is not None:
            if index > slices_count:
                raise ValueError(
                    "Item contains {} frames. Frame index is {}".format(slices_count, index)
                )

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = cls.figure_type.from_json(figure_json, objects, plane_name, index, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures)
