# coding: utf-8
"""Labeling object for :class:`Annotation<supervisely.annotation.annotation.Annotation>`"""

# docs
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Union
from PIL.ImageFont import FreeTypeFont
import numpy as np
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.image_rotator import ImageRotator

from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.annotation.tag import Tag
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.imaging import image as sly_image
from supervisely.imaging import font as sly_font
from supervisely.project.project_meta import ProjectMeta
from supervisely._utils import take_with_default
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.geometry.constants import GEOMETRY_TYPE, GEOMETRY_SHAPE


class LabelJsonFields:
    """Json fields for :class:`Annotation<supervisely.annotation.label.Label>`"""

    OBJ_CLASS_NAME = "classTitle"
    """"""
    OBJ_CLASS_ID = "classId"
    """"""
    DESCRIPTION = "description"
    """"""
    TAGS = "tags"
    """"""
    INSTANCE_KEY = "instance"
    """"""


class LabelBase:
    """
    Labeling object for :class:`Annotation<supervisely.annotation.annotation.Annotation>`. :class:`Label<Label>` object is immutable.

    :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
    :type geometry: Geometry
    :param obj_class: Label :class:`class<supervisely.annotation.obj_class.ObjClass>`.
    :type obj_class: ObjClass
    :param tags: Label :class:`tags<supervisely.annotation.tag_collection.TagCollection>`.
    :type tags: TagCollection or List[Tag]
    :param description: Label description.
    :type description: str, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Simple Label example
        class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)
        figure = sly.Rectangle(0, 0, 300, 300)
        label_kiwi = sly.Label(figure, class_kiwi)

        # More complex Label example
        # Tag
        meta_kiwi = sly.TagMeta('kiwi_tag', sly.TagValueType.ANY_STRING)
        tag_kiwi = sly.Tag(meta_kiwi, 'Hello')
        # ObjClass
        class_kiwi = sly.ObjClass('kiwi', sly.Rectangle)

        # Label
        geometry_figure = sly.Rectangle(0, 0, 300, 300)
        label = sly.Label(figure, class_kiwi, sly.TagCollection([tag_kiwi]), 'Label description')
        # or sly.Label(figure, class_kiwi, [tag_kiwi], 'Label description')
    """

    def __init__(
        self,
        geometry: Geometry,
        obj_class: ObjClass,
        tags: Optional[Union[TagCollection, List[Tag]]] = None,
        description: Optional[str] = "",
        binding_key=None,
    ):
        self._geometry = geometry
        self._obj_class = obj_class
        self._tags = take_with_default(tags, TagCollection())
        self._description = description
        self._validate_geometry_type()
        self._validate_geometry()

        if not isinstance(tags, TagCollection):
            self._tags = TagCollection(tags)

        self._binding_key = binding_key

    def _validate_geometry(self):
        """
        The function checks the name of the Object for compliance.
        :return: generate ValueError error if name is mismatch
        """
        self._geometry.validate(
            self._obj_class.geometry_type.geometry_name(), self.obj_class.geometry_config
        )

    def _validate_geometry_type(self):
        raise NotImplementedError()

    @property
    def obj_class(self) -> ObjClass:
        """
        ObjClass of the current Label.

        :return: ObjClass object
        :rtype: :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>`
        :Usage example:

         .. code-block:: python

            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(150, 150, 400, 500), class_dog)

            label_dog_json = label_dog.obj_class.to_json()
            print(label_dog_json)
            # Output: {
            #    "title": "dog",
            #    "shape": "rectangle",
            #    "color": "#0F8A12",
            #    "geometry_config": {},
            #    "hotkey": ""
            # }
        """
        return self._obj_class

    @property
    def description(self) -> str:
        """
        Description of the current Label.

        :return: Description
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(150, 150, 400, 500), class_dog, description="Insert Label description here")

            print(label_dog.description)
            # Output: 'Insert Label description here'
        """
        return self._description

    @property
    def geometry(self) -> Geometry:
        """
        Geometry of the current Label.

        :return: Geometry object
        :rtype: :class:`Geometry<supervisely.geometry>`
        :Usage example:

         .. code-block:: python

            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(150, 150, 400, 500), class_dog)

            label_json = label_dog.geometry.to_json()
            print(label_json)
            # Output: {
            #    "points": {
            #        "exterior": [
            #            [
            #                150,
            #                150
            #            ],
            #            [
            #                500,
            #                400
            #            ]
            #        ],
            #        "interior": []
            #    }
            # }
        """
        return self._geometry

    @property
    def tags(self) -> TagCollection:
        """
        TagCollection of the current Label.

        :return: TagCollection object
        :rtype: :class:`TagCollection<supervisely.annotation.tag.TagCollection>`
        :Usage example:

         .. code-block:: python

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta_dog, 'Woof')
            class_dog = sly.ObjClass('dog', sly.Rectangle)

            label_dog = sly.Label(sly.Rectangle(100, 100, 700, 900), class_dog, sly.TagCollection([tag_dog]))

            label_dog_json = label_dog.tags.to_json()
            print(label_dog_json)
            # Output: [
            #    {
            #        "name": "dog",
            #        "value": "Woof"
            #    }
            # ]
        """
        return self._tags.clone()

    def to_json(self) -> Dict:
        """
        Convert the Label to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_dog = sly.TagMeta('dog', sly.TagValueType.ANY_STRING)
            tag_dog = sly.Tag(meta_dog, 'Woof')
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(100, 100, 700, 900), class_dog, sly.TagCollection([tag_dog]), description='Insert Label description here')

            label_dog_json = label_dog.to_json()
            print(label_dog_json)
            # Output: {
            #    "classTitle": "dog",
            #    "description": "",
            #    "tags": [
            #        {
            #            "name": "dog",
            #            "value": "Woof"
            #        }
            #    ],
            #    "points": {
            #        "exterior": [[100, 100],[900, 700]],
            #        "interior": []
            #    },
            #    "geometryType": "rectangle",
            #    "shape": "rectangle"
            # }
        """
        res = {
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.DESCRIPTION: self.description,
            LabelJsonFields.TAGS: self.tags.to_json(),
            **self.geometry.to_json(),
            GEOMETRY_TYPE: self.geometry.geometry_name(),
            GEOMETRY_SHAPE: self.geometry.geometry_name(),
        }

        if self.obj_class.sly_id is not None:
            res[LabelJsonFields.OBJ_CLASS_ID] = self.obj_class.sly_id

        if self.binding_key is not None:
            res[LabelJsonFields.INSTANCE_KEY] = self.binding_key

        return res

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta) -> LabelBase:
        """
        Convert a json dict to Label. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Label in json format as a dict.
        :type data: dict
        :param project_meta: Input ProjectMeta.
        :type project_meta: ProjectMeta
        :return: Label object
        :rtype: :class:`Label<LabelBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            meta = api.project.get_meta(PROJECT_ID)

            data = {
                "classTitle": "dog",
                "tags": [
                    {
                        "name": "dog",
                        "value": "Woof"
                    }
                ],
                "points": {
                    "exterior": [[100, 100], [900, 700]],
                    "interior": []
                }
            }

            label_dog = sly.Label.from_json(data, meta)
        """
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(
                f"Failed to deserialize a Label object from JSON: label class name {obj_class_name!r} "
                f"was not found in the given project meta."
            )

        if obj_class.geometry_type is AnyGeometry:
            geometry_type_actual = GET_GEOMETRY_FROM_STR(
                data[GEOMETRY_TYPE] if GEOMETRY_TYPE in data else data[GEOMETRY_SHAPE]
            )
            geometry = geometry_type_actual.from_json(data)
        else:
            geometry = obj_class.geometry_type.from_json(data)

        binding_key = data.get(LabelJsonFields.INSTANCE_KEY)
        return cls(
            geometry=geometry,
            obj_class=obj_class,
            tags=TagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas),
            description=data.get(LabelJsonFields.DESCRIPTION, ""),
            binding_key=binding_key,
        )

    def add_tag(self, tag: Tag) -> LabelBase:
        """
        Adds Tag to the current Label.

        :param tag: Tag to be added.
        :type tag: Tag
        :return: Label object
        :rtype: :class:`Label<LabelBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create label
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(0, 0, 500, 600), class_dog)

            # Create tag
            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            tag_dog = sly.Tag(meta_dog)

            # Add Tag
            # Remember that Label object is immutable, and we need to assign new instance of Label to a new variable
            new_label = label_dog.add_tag(tag_dog)
        """
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: List[Tag]) -> LabelBase:
        """
        Adds multiple Tags to the current Label.

        :param tags: List of Tags to be added.
        :type tags: List[Tag]
        :return: Label object
        :rtype: :class:`Label<LabelBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create label
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(0, 0, 500, 600), class_dog)

            # Create tags
            meta_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            tag_dog = sly.Tag(meta_dog)

            meta_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            tag_cat = sly.Tag(meta_cat)

            tags_arr = [tag_dog, tag_cat]

            # Add Tags
            # Remember that Label object is immutable, and we need to assign new instance of Label to a new variable
            new_label = label_dog.add_tags(tags_arr)
        """
        return self.clone(tags=self._tags.add_items(tags))

    def clone(
        self,
        geometry: Optional[Geometry] = None,
        obj_class: Optional[ObjClass] = None,
        tags: Optional[Union[TagCollection, List[Tag]]] = None,
        description: Optional[str] = None,
        binding_key: Optional[str] = None,
    ) -> LabelBase:
        """
        Makes a copy of Label with new fields, if fields are given, otherwise it will use fields of the original Label.

        :param geometry: Label :class:`geometry<supervisely.geometry.geometry.Geometry>`.
        :type geometry: Geometry
        :param obj_class: Label :class:`class<supervisely.annotation.obj_class.ObjClass>`.
        :type obj_class: ObjClass
        :param tags: Label :class:`tags<supervisely.annotation.tag.TagCollection>`.
        :type tags: TagCollection or List[Tag]
        :param description: Label description.
        :type description: str, optional
        :return: New instance of Label
        :rtype: :class:`Label<LabelBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            import numpy as np

            # Original Label
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(150, 150, 500, 400), class_dog)

            # Let's clone our Label, but with different Geometry coordinates
            # Remember that Label object is immutable, and we need to assign new instance of Label to a new variable
            clone_label_dog = label_dog.clone(sly.Rectangle(100, 100, 500, 500), class_dog)

            # Let's clone our Label with new TagCollection and description
            meta_breed = sly.TagMeta('breed', sly.TagValueType.ANY_STRING)
            tag_breed = sly.Tag(meta_breed, 'German Shepherd')
            tags = sly.TagCollection([tag_breed])

            # Remember that Label object is immutable, and we need to assign new instance of Label to a new variable
            clone_label_dog_2 = label_dog.clone(tags=tags, description='Dog breed german shepherd')

            # Note that you can't clone Label if ObjClass geometry type differ from the new given geometry
            mask = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]], dtype=np.uint8)

            mask_bool = mask==1

            clone_label_dog = label_dog.clone(sly.Label(sly.Bitmap(mask_bool), class_dog))
            # In this case RuntimeError will be raised
        """
        return self.__class__(
            geometry=take_with_default(geometry, self.geometry),
            obj_class=take_with_default(obj_class, self.obj_class),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
            binding_key=take_with_default(binding_key, self.binding_key),
        )

    def crop(self, rect: Rectangle) -> List[LabelBase]:
        """
        Clones the current Label and crops it. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.crop_labels>`.

        :param rect: Rectangle object.
        :type rect: Rectangle
        :return: List of Labels with new geometries
        :rtype: :class:`List[Label]<LabelBase>`
        """
        if rect.contains(self.geometry.to_bbox()):
            return [self]
        else:
            # for compatibility of old slightly invalid annotations, some of them may be out of image bounds.
            # will correct it automatically
            result_geometries = self.geometry.crop(rect)
            if len(result_geometries) == 1:
                result_geometries[0]._copy_creation_info_inplace(self.geometry)
                return [self.clone(geometry=result_geometries[0])]
            else:
                return [self.clone(geometry=g) for g in self.geometry.crop(rect)]

    def relative_crop(self, rect: Rectangle) -> List[LabelBase]:
        """
        Clones the current Label and crops it, but return results with coordinates relative to the given Rectangle. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.relative_crop>`.

        :param rect: Rectangle object.
        :type rect: Rectangle
        :return: List of Labels with new geometries
        :rtype: :class:`List[Label]<LabelBase>`
        """
        return [self.clone(geometry=g) for g in self.geometry.relative_crop(rect)]

    def rotate(self, rotator: ImageRotator) -> LabelBase:
        """
        Clones the current Label and rotates it. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.rotate>`.

        :param rotator: ImageRotator object.
        :type rotator: ImageRotator
        :return: New instance of Label with rotated geometry
        :rtype: :class:`Label<LabelBase>`
        """
        return self.clone(geometry=self.geometry.rotate(rotator))

    def resize(self, in_size: Tuple[int, int], out_size: Tuple[int, int]) -> LabelBase:
        """
        Clones the current Label and resizes it. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.resize>`.

        :param in_size: Input image size (height, width) of the Annotation to which Label belongs.
        :type in_size: Tuple[int, int]
        :param out_size: Desired output image size (height, width) of the Annotation to which Label belongs.
        :type out_size: Tuple[int, int]
        :return: New instance of Label with resized geometry
        :rtype: :class:`Label<LabelBase>`
        """
        return self.clone(geometry=self.geometry.resize(in_size, out_size))

    def scale(self, factor: float) -> LabelBase:
        """
        Clones the current Label and scales it. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.scale>`.

        :param factor: Scale factor.
        :type factor: float
        :return: New instance of Label with scaled geometry
        :rtype: :class:`Label<LabelBase>`
        """
        return self.clone(geometry=self.geometry.scale(factor))

    def translate(self, drow: int, dcol: int) -> LabelBase:
        """
        Clones the current Label and shifts it by a certain number of pixels. Mostly used for internal implementation.

        :param drow: Horizontal shift.
        :type drow: int
        :param dcol: Vertical shift.
        :type dcol: int
        :return: New instance of Label with translated geometry
        :rtype: :class:`Label<LabelBase>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            # Get image and annotation from API
            project_id = 117139
            image_id = 194190568
            meta_json = api.project.get_meta(project_id)
            meta = sly.ProjectMeta.from_json(meta_json)
            ann_info = api.annotation.download(image_id)
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            img = api.image.download_np(image_id)
            new_img = copy.deepcopy(img)
            ann.draw_pretty(img, thickness=3) # before

            # Translate label with name 'lemon'
            new_labels = []
            for label in ann.labels:
                if label.obj_class.name == 'lemon':
                    translate_label = label.translate(250, -350)
                    new_labels.append(translate_label)
                else:
                    new_labels.append(label)
            ann = ann.clone(labels=new_labels)
            ann.draw_pretty(new_img, thickness=3)  # after
        """
        return self.clone(geometry=self.geometry.translate(drow=drow, dcol=dcol))

    def fliplr(self, img_size: Tuple[int, int]) -> LabelBase:
        """
        Clones the current Label and flips it horizontally. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.fliplr>`.

        :param img_size: Input image size (height, width) of the Annotation to which Label belongs.
        :type img_size: Tuple[int, int]
        :return: New instance of Label with flipped geometry
        :rtype: :class:`Label<LabelBase>`
        """
        return self.clone(geometry=self.geometry.fliplr(img_size))

    def flipud(self, img_size: Tuple[int, int]) -> LabelBase:
        """
        Clones the current Label and flips it vertically. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.flipud>`.

        :param img_size: Input image size (height, width) of the Annotation to which Label belongs.
        :type img_size: Tuple[int, int]
        :return: New instance of Label with flipped geometry
        :rtype: :class:`Label<LabelBase>`
        """
        return self.clone(geometry=self.geometry.flipud(img_size))

    def _get_font(self, img_size):
        """
        The function get size of font for image with given size
        :return: font for drawing
        """
        return sly_font.get_font(font_size=sly_font.get_readable_font_size(img_size))

    def _draw_tags(self, bitmap, font):
        bbox = self.geometry.to_bbox()
        texts = [tag.get_compact_str() for tag in self.tags]
        sly_image.draw_text_sequence(
            bitmap=bitmap,
            texts=texts,
            anchor_point=(bbox.top, bbox.left),
            corner_snap=sly_image.CornerAnchorMode.BOTTOM_LEFT,
            font=font,
        )

    def draw(
        self,
        bitmap: np.ndarray,
        color: Optional[List[int, int, int]] = None,
        thickness: Optional[int] = 1,
        draw_tags: Optional[bool] = False,
        tags_font: Optional[FreeTypeFont] = None,
    ) -> None:
        """
        Draws current Label on image. Modifies Mask. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.draw>`.

        :param bitmap: image.
        :type bitmap: np.ndarray
        :param color: Drawing color in :class:`[R, G, B]`.
        :type color: List[int, int, int], optional
        :param thickness: Thickness of the drawing figure.
        :type thickness: int, optional
        :param draw_tags: Determines whether to draw tags on bitmap or not.
        :type draw_tags: bool, optional
        :param tags_font: Font of tags to be drawn, uses `FreeTypeFont <https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.FreeTypeFont>`_ from `PIL <https://pillow.readthedocs.io/en/stable/index.html>`_.
        :type tags_font: FreeTypeFont, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        """
        effective_color = take_with_default(color, self.obj_class.color)
        self.geometry.draw(
            bitmap, effective_color, thickness, config=self.obj_class.geometry_config
        )
        if draw_tags:
            if tags_font is None:
                tags_font = self._get_font(bitmap.shape[:2])
            self._draw_tags(bitmap, tags_font)

    def draw_contour(
        self,
        bitmap: np.ndarray,
        color: Optional[List[int, int, int]] = None,
        thickness: Optional[int] = 1,
        draw_tags: Optional[bool] = False,
        tags_font: Optional[FreeTypeFont] = None,
    ) -> None:
        """
        Draws Label geometry contour on the given image. Modifies mask. Mostly used for internal implementation. See usage example in :class:`Annotation<supervisely.annotation.annotation.Annotation.draw_contour>`.

        :param bitmap: image.
        :type bitmap: np.ndarray
        :param color: Drawing color in :class:`[R, G, B]`.
        :type color: List[int, int, int], optional
        :param thickness: Thickness of the drawn contour.
        :type thickness: int, optional
        :param draw_tags: Determines whether to draw tags on bitmap or not.
        :type draw_tags: bool, optional
        :param tags_font: Font of tags to be drawn, uses `FreeTypeFont <https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.FreeTypeFont>`_ from `PIL <https://pillow.readthedocs.io/en/stable/index.html>`_.
        :type tags_font: FreeTypeFont, optional
        :return: :class:`None<None>`
        :rtype: :class:`NoneType<NoneType>`
        """
        effective_color = take_with_default(color, self.obj_class.color)
        self.geometry.draw_contour(
            bitmap, effective_color, thickness, config=self.obj_class.geometry_config
        )
        if draw_tags:
            if tags_font is None:
                tags_font = self._get_font(bitmap.shape[:2])
            self._draw_tags(bitmap, tags_font)

    @property
    def area(self) -> float:
        """
        Label area.

        :return: Area of current geometry in Label.
        :rtype: :class:`float`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Create label
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(0, 0, 500, 600), class_dog)

            figure_area = label_dog.area
            print(figure_area)
            # Output: 301101.0
        """
        return self.geometry.area

    def convert(self, new_obj_class: ObjClass) -> List[LabelBase]:
        """
        Converts Label geometry to another geometry shape.

        :param new_obj_class: ObjClass with new geometry shape.
        :type new_obj_class: ObjClass
        :return: List of Labels with converted geometries
        :rtype: :class:`List[Label]<LabelBase>`
        :Usage example:

         .. code-block:: python

           import supervisely as sly

           # Create label
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            label_dog = sly.Label(sly.Rectangle(0, 0, 500, 600), class_dog)

            print(label_dog.geometry.to_json())
            # {'geometryType': 'rectangle'}

            label_cat = sly.ObjClass('cat', sly.Bitmap)

            convert_label = label_dog.convert(label_cat)
            for label_bitmap in convert_label:
                print(label_bitmap.geometry.to_json())
                # Output: {'geometryType': 'bitmap'}
        """
        labels = []
        geometries = self.geometry.convert(new_obj_class.geometry_type)
        for g in geometries:
            labels.append(self.clone(geometry=g, obj_class=new_obj_class))
        return labels

    @property
    def binding_key(self):
        return self._binding_key

    @binding_key.setter
    def binding_key(self, key: Union[str, None]):
        if key is not None and type(key) is not str:
            raise TypeError("Key has to be of type string or None")
        self._binding_key = key

    @property
    def labeler_login(self):
        return self.geometry.labeler_login

class Label(LabelBase):
    def _validate_geometry_type(self):
        """
        Checks geometry type for correctness
        """
        if self._obj_class.geometry_type != AnyGeometry:
            if type(self._geometry) is not self._obj_class.geometry_type:
                raise RuntimeError(
                    "Input geometry type {!r} != geometry type of ObjClass {}".format(
                        type(self._geometry), self._obj_class.geometry_type
                    )
                )


class PixelwiseScoresLabel(LabelBase):
    def _validate_geometry_type(self):
        if type(self._geometry) is not MultichannelBitmap:
            raise RuntimeError(
                "Input geometry type {!r} != geometry type of ObjClass {}".format(
                    type(self._geometry), MultichannelBitmap
                )
            )
