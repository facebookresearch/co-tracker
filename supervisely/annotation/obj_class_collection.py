# coding: utf-8
"""collection with :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` instances"""

# docs
from __future__ import annotations
from typing import List, Optional, Dict, Iterator
from supervisely import logger
from supervisely.annotation.renamer import Renamer

from collections import defaultdict
from supervisely.collection.key_indexed_collection import KeyIndexedCollection
from supervisely.io.json import JsonSerializable
from supervisely.annotation.obj_class import ObjClass
from supervisely.imaging.color import rgb2hex, hex2rgb


class ObjClassCollection(KeyIndexedCollection, JsonSerializable):
    """
    Collection with :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` instances. :class:`ObjClassCollection<ObjClassCollection>` object is immutable.

    :raises: :class:`DuplicateKeyError` if instance with given name already exist
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create ObjClass (see class ObjClass for more information)
        class_lemon = sly.ObjClass('lemon', sly.Rectangle)
        class_kiwi = sly.ObjClass('kiwi', sly.Bitmap)

        class_arr = [class_lemon, class_kiwi]

        # Create ObjClassCollection from ObjClasses
        classes = sly.ObjClassCollection(class_arr)

        # Add items to ObjClassCollection
        class_potato = sly.ObjClass('potato', sly.Bitmap)

        # Remember that ObjClassCollection is immutable, and we need to assign new instance of ObjClassCollection to a new variable
        classes = classes.add(class_potato)

        # You can also add multiple items to collection
        class_cabbage = sly.ObjClass('cabbage', sly.Rectangle)
        class_carrot = sly.ObjClass('carrot', sly.Bitmap)
        class_turnip = sly.ObjClass('turnip', sly.Polygon)

        classes = classes.add_items([class_cabbage, class_carrot, class_turnip])

        # Has key, checks if given key exist in collection
        classes.has_key("cabbage")
        # Output: True

        # Intersection, finds intersection of given list of instances with collection items
        class_dog = sly.ObjClass('dog', sly.Rectangle)
        class_cat = sly.ObjClass('cat', sly.Rectangle)
        class_turtle = sly.ObjClass('turtle', sly.Rectangle)

        classes_animals = sly.ObjClassCollection([class_dog, class_cat, class_turtle])

        classes_intersections = classes.intersection(classes_animals)
        print(classes_intersections.to_json())
        # Output: []

        # Let's add the potato ObjClass from another collection and compare them again
        classes_animals = classes_animals.add(class_potato)

        classes_intersections = classes.intersection(classes_animals)
        print(classes_intersections.to_json())
        # Output: [
        #     {
        #         "title":"potato",
        #         "shape":"bitmap",
        #         "color":"#8A570F",
        #         "geometry_config":{},
        #         "hotkey":""
        #     }
        # ]

        # Difference, finds difference between collection and given list of ObjClass or ObjClassCollection
        class_car = sly.ObjClass('car', sly.Rectangle)
        class_bicycle = sly.ObjClass('bicycle', sly.Rectangle)

        classes_vehicles = sly.ObjClassCollection([class_car, class_bicycle])

        class_pedestrian = sly.ObjClass('pedestrian', sly.Rectangle)
        class_road = sly.ObjClass('road', sly.Rectangle)

        difference = classes_vehicles.difference([class_pedestrian, class_road])
        print(difference.to_json())
        # Output: [
        #     {
        #         "title":"car",
        #         "shape":"rectangle",
        #         "color":"#8A0F3B",
        #         "geometry_config":{},
        #         "hotkey":""
        #     },
        #     {
        #         "title":"bicycle",
        #         "shape":"rectangle",
        #         "color":"#0F8A1F",
        #         "geometry_config":{},
        #         "hotkey":""
        #     }
        # ]

        # Merge, merges collection and given list of ObjClasses
        c_1 = sly.ObjClassCollection([class_car, class_bicycle])
        c_2 = sly.ObjClassCollection([class_pedestrian, class_road])

        с_3 = c_1.merge(c_2)
        print(с_3.to_json())
        # Output: [
        #     {
        #         "title":"pedestrian",
        #         "shape":"rectangle",
        #         "color":"#8A0F27",
        #         "geometry_config":{},
        #         "hotkey":""
        #     },
        #     {
        #         "title":"road",
        #         "shape":"rectangle",
        #         "color":"#8A620F",
        #         "geometry_config":{},
        #         "hotkey":""
        #     },
        #     {
        #         "title":"car",
        #         "shape":"rectangle",
        #         "color":"#8A0F3B",
        #         "geometry_config":{},
        #         "hotkey":""
        #     },
        #     {
        #         "title":"bicycle",
        #         "shape":"rectangle",
        #         "color":"#0F8A1F",
        #         "geometry_config":{},
        #         "hotkey":""
        #     }
        # ]

        # Merge will raise ValueError if item name from given list is in collection but items in both are different
        class_bicycle_1 = sly.ObjClass('bicycle', sly.Rectangle)
        class_bicycle_2 = sly.ObjClass('bicycle', sly.Bitmap)

        classes_1 = sly.ObjClassCollection([class_bicycle_1])
        classes_2 = sly.ObjClassCollection([class_bicycle_2])

        test_merge = classes_1.merge(classes_2)
        # Output: ValueError: Error during merge for key 'bicycle': values are different

        # Let's try to create now a collection where ObjClasses have identical names
        class_cow = sly.ObjClass('cow', sly.Rectangle)
        class_chicken = sly.ObjClass('cow', sly.Rectangle)

        test_classes = sly.ObjClassCollection([class_cow, class_chicken])
        # Output: DuplicateKeyError: "Key 'cow' already exists"
    """

    item_type = ObjClass

    def to_json(self) -> List[Dict]:
        """
        Convert the ObjClassCollection to a list of json dicts. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: List of dicts in json format
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            class_lemon = sly.ObjClass('lemon', sly.Rectangle)
            class_kiwi = sly.ObjClass('kiwi', sly.Bitmap)

            # Add ObjClasses to ObjClassCollection
            classes = sly.ObjClassCollection([class_lemon, class_kiwi])

            classes_json = classes.to_json()
            print(classes_json)
            # Output: [
            #      {
            #           "title": "lemon",
            #           "shape": "rectangle",
            #           "color": "#300F8A",
            #           "geometry_config": {},
            #           "hotkey": ""
            #      },
            #      {
            #           "title": "kiwi",
            #           "shape": "bitmap",
            #           "color": "#7C0F8A",
            #           "geometry_config": {},
            #           "hotkey": ""
            #      }
            # ]
        """
        return [obj_class.to_json() for obj_class in self]

    @classmethod
    def from_json(cls, data: List[Dict]) -> ObjClassCollection:
        """
        Convert a list with dicts in json format to ObjClassCollection. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: List with dicts in json format.
        :type data: List[dict]
        :return: ObjClassCollection object
        :rtype: :class:`ObjClassCollection<ObjClassCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            data = [
                 {
                      "title": "lemon",
                      "shape": "rectangle",
                      "color": "#300F8A",
                      "hotkey": ""
                 },
                 {
                      "title": "kiwi",
                      "shape": "bitmap",
                      "color": "#7C0F8A",
                      "hotkey": ""
                 }
            ]

            classes = sly.ObjClassCollection.from_json(data)
        """
        obj_classes = [ObjClass.from_json(obj_class_json) for obj_class_json in data]
        return cls(obj_classes)

    def validate_classes_colors(self, logger: Optional[logger] = None) -> str or None:
        """
        Checks for unique colors in the ObjClassCollection.

        :param logger: Input logger.
        :type logger: logger, optional
        :return: Notification if there are objects with the same colors, otherwise :class:`None`
        :rtype: :class:`str` or :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            # Let's create 2 ObjClasses with the same color
            class_lemon = sly.ObjClass('lemon', sly.Rectangle, [0, 0, 0])
            class_kiwi = sly.ObjClass('kiwi', sly.Bitmap, [0, 0, 0])

            # Add them to ObjClassCollection
            classes = sly.ObjClassCollection([class_lemon, class_kiwi])

            print(classes.validate_classes_colors())
            # Output: Classes ['lemon', 'kiwi'] have the same RGB color = [0, 0, 0]

            # Now let's change colors of our ObjClasses
            class_lemon = sly.ObjClass('lemon', sly.Rectangle, [255, 0, 0])
            class_kiwi = sly.ObjClass('kiwi', sly.Bitmap, [0, 0, 255])

            classes = sly.ObjClassCollection([class_lemon, class_kiwi])

            print(classes.validate_classes_colors())
            # Output: None
        """
        color_names = defaultdict(list)
        for obj_class in self:
            hex = rgb2hex(obj_class.color)
            color_names[hex].append(obj_class.name)

        class_colors_notify = None
        for hex_color, class_names in color_names.items():
            if len(class_names) > 1:
                warn_str = "Classes {!r} have the same RGB color = {!r}".format(
                    class_names, hex2rgb(hex_color)
                )
                if logger is not None:
                    logger.warn(warn_str)
                if class_colors_notify is None:
                    class_colors_notify = ""
                class_colors_notify += warn_str + "\n\n"
        return class_colors_notify

    def __iter__(self) -> Iterator[ObjClass]:
        return next(self)

    def refresh_ids_from(self, classes: ObjClassCollection):
        for new_class in classes:
            my_class = self.get(new_class.name)
            if my_class is None:
                continue
            my_class._set_id(new_class.sly_id)


def make_renamed_classes(
    src_obj_classes: ObjClassCollection,
    renamer: Renamer,
    skip_missing: Optional[bool] = False,
) -> ObjClassCollection:
    renamed_classes = []
    for src_cls in src_obj_classes:
        renamed_name = renamer.rename(src_cls.name)
        if renamed_name is not None:
            renamed_classes.append(src_cls.clone(name=renamed_name))
        elif not skip_missing:
            raise KeyError(
                "Object class name {} could not be mapped to a destination name.".format(
                    src_cls.name
                )
            )
    return ObjClassCollection(items=renamed_classes)
