# coding: utf-8

# docs
from __future__ import annotations
from supervisely.project.project_type import ProjectType
from supervisely.annotation.tag_meta import TagMeta
from typing import List, Dict, Optional, Tuple, Union


from supervisely.io.json import JsonSerializable
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely._utils import take_with_default

from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.rectangle import Rectangle


class ProjectMetaJsonFields:
    OBJ_CLASSES = "classes"
    IMG_TAGS = "tags_images"
    OBJ_TAGS = "tags_objects"
    TAGS = "tags"
    PROJECT_TYPE = "projectType"


def _merge_img_obj_tag_metas(
    img_tag_metas: ObjClassCollection, obj_tag_metas: ObjClassCollection
) -> ObjClassCollection:
    obj_tag_metas_to_add = []
    for obj_tag_meta in obj_tag_metas:
        img_tag_meta_same_key = img_tag_metas.get(obj_tag_meta.key(), None)
        if img_tag_meta_same_key is None:
            obj_tag_metas_to_add.append(obj_tag_meta)
        elif not img_tag_meta_same_key.is_compatible(obj_tag_meta):
            raise ValueError(
                "Unable to merge tag metas for images and objects. Found tags with the same name, but incompatible "
                "values. \n Image-level tag meta: {}\n Object-level tag meta: {}.\n Rename one of the tags to have a "
                "unique name to be able to load project meta.".format(
                    str(img_tag_meta_same_key), str(obj_tag_meta)
                )
            )
    return img_tag_metas.add_items(obj_tag_metas_to_add)


class ProjectMeta(JsonSerializable):
    """
    General information about ProjectMeta. :class:`ProjectMeta<ProjectMeta>` object is immutable.

    :param obj_classes: ObjClassCollection or just list that stores ObjClass instances with unique names.
    :type obj_classes: ObjClassCollection or List[ObjClass], optional
    :param tag_metas: TagMetaCollection or just list that stores TagMeta instances with unique names.
    :type tag_metas: TagMetaCollection or List[TagMeta], optional
    :param project_type: Type of items in project: images, videos, volumes, point_clouds.
    :type project_type: str, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        #Empty ProjectMeta example
        meta = sly.ProjectMeta()
        print(meta)
        # Output:
        # ProjectMeta:
        # Object Classes
        # +------+-------+-------+--------+
        # | Name | Shape | Color | Hotkey |
        # +------+-------+-------+--------+
        # +------+-------+-------+--------+
        # Tags
        # +------+------------+-----------------+--------+---------------+--------------------+
        # | Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
        # +------+------------+-----------------+--------+---------------+--------------------+
        # +------+------------+-----------------+--------+---------------+--------------------+

        #More complex ProjectMeta example
        lemon = sly.ObjClass('lemon', sly.Rectangle)
        kiwi = sly.ObjClass('kiwi', sly.Polygon)
        tag_fruit = sly.TagMeta('fruit', sly.TagValueType.ANY_STRING)
        objects = sly.ObjClassCollection([lemon, kiwi])
        # or objects = [lemon, kiwi]
        tags = sly.TagMetaCollection([tag_fruit])
        # or tags = [tag_fruit]
        meta = sly.ProjectMeta(obj_classes=objects, tag_metas=tags, project_type=sly.ProjectType.IMAGES)
        print(meta)
        # Output:
        # +-------+-----------+----------------+--------+
        # |  Name |   Shape   |     Color      | Hotkey |
        # +-------+-----------+----------------+--------+
        # | lemon | Rectangle | [108, 15, 138] |        |
        # |  kiwi |  Polygon  | [15, 98, 138]  |        |
        # +-------+-----------+----------------+--------+
        # Tags
        # +-------+------------+-----------------+--------+---------------+--------------------+
        # |  Name | Value type | Possible values | Hotkey | Applicable to | Applicable classes |
        # +-------+------------+-----------------+--------+---------------+--------------------+
        # | fruit | any_string |       None      |        |      all      |         []         |
        # +-------+------------+-----------------+--------+---------------+--------------------+

        # Example 2

        cat_class = sly.ObjClass("cat", sly.Rectangle, color=[0, 255, 0])
        scene_tag = sly.TagMeta("scene", sly.TagValueType.ANY_STRING)
        meta = sly.ProjectMeta(obj_classes=[cat_class], tag_metas=[scene_tag])
    """

    def __init__(
        self,
        obj_classes: Optional[Union[ObjClassCollection, List[ObjClass]]] = None,
        tag_metas: Optional[Union[TagMetaCollection, List[TagMeta]]] = None,
        project_type: Optional[ProjectType] = None,
    ):

        if obj_classes is None:
            self._obj_classes = ObjClassCollection()
        elif isinstance(obj_classes, list):
            self._obj_classes = ObjClassCollection(obj_classes)
        elif isinstance(obj_classes, ObjClassCollection):
            self._obj_classes = obj_classes
        else:
            raise TypeError(f"obj_classes argument has unknown type {type(obj_classes)}")

        if tag_metas is None:
            self._tag_metas = TagMetaCollection()
        elif isinstance(tag_metas, list):
            self._tag_metas = TagMetaCollection(tag_metas)
        elif isinstance(tag_metas, TagMetaCollection):
            self._tag_metas = tag_metas
        else:
            raise TypeError(f"tag_metas argument has unknown type {type(tag_metas)}")

        self._project_type = project_type

    @property
    def obj_classes(self) -> ObjClassCollection:
        """
        Collection of ObjClasses in ProjectMeta.

        :return: ObjClassCollection object
        :rtype: :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            lemon = sly.ObjClass('lemon', sly.Rectangle)
            kiwi = sly.ObjClass('kiwi', sly.Polygon)
            objects = sly.ObjClassCollection([lemon, kiwi])
            # or objects = [lemon, kiwi]

            meta = sly.ProjectMeta(obj_classes=objects, project_type=sly.ProjectType.IMAGES)

            meta_classes = meta.obj_classes
            print(meta_classes.to_json())
            # Output: [
            #     {
            #         "title":"lemon",
            #         "shape":"rectangle",
            #         "color":"#6C0F8A",
            #         "geometry_config":{
            #
            #         },
            #         "hotkey":""
            #     },
            #     {
            #         "title":"kiwi",
            #         "shape":"polygon",
            #         "color":"#0F628A",
            #         "geometry_config":{
            #
            #         },
            #         "hotkey":""
            #     }
            # ]
        """
        return self._obj_classes

    @property
    def tag_metas(self) -> TagMetaCollection:
        """
        Collection of TagMetas in ProjectMeta.

        :return: TagMetaCollection object
        :rtype: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_fruit = sly.TagMeta('fruit', sly.TagValueType.ANY_STRING)
            tags = sly.TagMetaCollection([tag_fruit])
            # or tags = [tag_fruit]

            meta = sly.ProjectMeta(tag_metas=tags)

            meta_tags = meta.tag_metas
            print(meta_tags.to_json())
            # Output: [
            #     {
            #         "name":"fruit",
            #         "value_type":"any_string",
            #         "color":"#818A0F",
            #         "hotkey":"",
            #         "applicable_type":"all",
            #         "classes":[]
            #     }
            # ]
        """
        return self._tag_metas

    @property
    def project_type(self):
        """
        Type of project. See possible value types in :class:`ProjectType<supervisely.project.project_type.ProjectType>`.

        :return: Project type
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta(project_type=sly.ProjectType.IMAGES)

            print(meta.project_type)
            # Output: <ProjectType.IMAGES: 'images'>
        """
        return self._project_type

    def to_json(self) -> Dict:
        """
        Convert the ProjectMeta to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :return: Json format as a dict
        :rtype: :class:`dict`

        :Usage example:

         .. code-block:: python

            meta_json = meta.to_json()
            print(meta_json)
            # Output: {
            #     "classes": [
            #         {
            #             "title": "lemon",
            #             "shape": "rectangle",
            #             "color": "#720F8A",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "kiwi",
            #             "shape": "polygon",
            #             "color": "#8A0F6F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": [
            #         {
            #             "name": "fruit",
            #             "value_type": "any_string",
            #             "color": "#788A0F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        res = {
            ProjectMetaJsonFields.OBJ_CLASSES: self._obj_classes.to_json(),
            ProjectMetaJsonFields.TAGS: self._tag_metas.to_json(),
        }
        if self._project_type is not None:
            res[ProjectMetaJsonFields.PROJECT_TYPE] = self._project_type
        return res

    @classmethod
    def from_json(cls, data: Dict) -> ProjectMeta:
        """
        Convert a json dict to ProjectMeta. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: ProjectMeta in json format as a dict.
        :type data: dict
        :return: ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_json = {
                "classes": [
                    {
                        "title": "lemon",
                        "shape": "rectangle",
                        "color": "#8A0F7B"
                    },
                    {
                        "title": "kiwi",
                        "shape": "polygon",
                        "color": "#8A0F50"
                    }
                ],
                "tags": [
                    {
                        "name": "fruit",
                        "value_type": "any_string",
                        "color": "#0F6F8A"
                    }
                ]
            }
            meta = sly.ProjectMeta.from_json(meta_json)
        """
        tag_metas_json = data.get(ProjectMetaJsonFields.TAGS, [])
        img_tag_metas_json = data.get(ProjectMetaJsonFields.IMG_TAGS, [])
        obj_tag_metas_json = data.get(ProjectMetaJsonFields.OBJ_TAGS, [])
        project_type = data.get(ProjectMetaJsonFields.PROJECT_TYPE, None)

        if len(tag_metas_json) > 0:
            # New format - all project tags in a single collection.
            if any(len(x) > 0 for x in [img_tag_metas_json, obj_tag_metas_json]):
                raise ValueError(
                    "Project meta JSON contains both the {!r} section (current format merged tags for all of "
                    "the project) and {!r} or {!r} sections (legacy format with separate collections for images "
                    "and labeled objects). Either new format only or legacy format only are supported, but not a "
                    "mix.".format(
                        ProjectMetaJsonFields.TAGS,
                        ProjectMetaJsonFields.IMG_TAGS,
                        ProjectMetaJsonFields.OBJ_TAGS,
                    )
                )
            tag_metas = TagMetaCollection.from_json(tag_metas_json)
        else:
            img_tag_metas = TagMetaCollection.from_json(img_tag_metas_json)
            obj_tag_metas = TagMetaCollection.from_json(obj_tag_metas_json)
            tag_metas = _merge_img_obj_tag_metas(img_tag_metas, obj_tag_metas)

        return cls(
            obj_classes=ObjClassCollection.from_json(data[ProjectMetaJsonFields.OBJ_CLASSES]),
            tag_metas=tag_metas,
            project_type=project_type,
        )

    def merge(self, other: ProjectMeta) -> ProjectMeta:
        """
        Merge all instances from given ProjectMeta into a single ProjectMeta object.

        :param other: ProjectMeta object.
        :type other: ProjectMeta
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :raises: :class:`ValueError` Upon attempt to merge metas which contain the same obj class or tag meta
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta_1 = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            meta_1 = meta_1.add_obj_class(class_cat)
            meta_1 = meta_1.add_tag_meta(tag_cat)

            meta_2 = sly.ProjectMeta()
            class_dog = sly.ObjClass('dog', sly.Rectangle)
            tag_dog = sly.TagMeta('dog_tag', sly.TagValueType.ANY_STRING)
            meta_2 = meta_2.add_obj_class(class_dog)
            meta_2 = meta_2.add_tag_meta(tag_dog)

            merge_meta = meta_1.merge(meta_2)
            merge_meta_json = merge_meta.to_json()
            print(json.dumps(merge_meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "dog",
            #             "shape": "rectangle",
            #             "color": "#0F8A62",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#340F8A",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": [
            #         {
            #             "name": "dog_tag",
            #             "value_type": "any_string",
            #             "color": "#380F8A",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         },
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#8A0F82",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        return self.clone(
            obj_classes=self._obj_classes.merge(other.obj_classes),
            tag_metas=self._tag_metas.merge(other._tag_metas),
        )

    def clone(
        self,
        obj_classes: Optional[Union[ObjClassCollection, List[ObjClass]]] = None,
        tag_metas: Optional[Union[TagMetaCollection, List[TagMeta]]] = None,
        project_type: Optional[str] = None,
    ) -> ProjectMeta:
        """
        Clone makes a copy of ProjectMeta with new fields, if fields are given, otherwise it will use original ProjectMeta fields.

        :param obj_classes: ObjClassCollection or just list that stores ObjClass instances with unique names.
        :type obj_classes: ObjClassCollection or List[ObjClass], optional
        :param tag_metas: TagMetaCollection that stores TagMeta instances with unique names.
        :type tag_metas: TagMetaCollection or List[TagMeta], optional
        :param project_type: Type of items in project: images, videos, volumes, point_clouds.
        :type project_type: str, optional
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            collection_cat = sly.ObjClassCollection([class_cat])
            # or collection_cat = [class_cat]
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            collection_tag_cat = sly.TagMetaCollection([tag_cat])
            # or collection_tag_cat = [tag_cat]
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            new_meta = meta.clone(obj_classes=collection_cat, tag_metas=collection_tag_cat)
            new_meta_json = new_meta.to_json()
            print(json.dumps(new_meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#190F8A",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#8A6D0F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        return ProjectMeta(
            obj_classes=take_with_default(obj_classes, self.obj_classes),
            tag_metas=take_with_default(tag_metas, self.tag_metas),
            project_type=take_with_default(project_type, self.project_type),
        )

    def add_obj_class(self, new_obj_class: ObjClass) -> ProjectMeta:
        """
        Adds given ObjClass to ProjectMeta.

        :param new_obj_class: ObjClass object.
        :type new_obj_class: ObjClass
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_obj_class(class_cat)
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#178A0F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }
        """
        return self.add_obj_classes([new_obj_class])

    def add_obj_classes(self, new_obj_classes: List[ObjClass]) -> ProjectMeta:
        """
        Adds given ObjClasses to ProjectMeta.

        :param new_obj_classes: List of ObjClass objects.
        :type new_obj_classes: List[ObjClass]
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            class_dog = sly.ObjClass('dog', sly.Bitmap)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_obj_classes([class_cat, class_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#8A0F3F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#8A0F56",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }
        """
        return self.clone(obj_classes=self.obj_classes.add_items(new_obj_classes))

    def add_tag_meta(self, new_tag_meta: TagMeta) -> ProjectMeta:
        """
        Adds given TagMeta to ProjectMeta.

        :param new_tag_meta: TagMeta object.
        :type new_tag_meta: TagMeta
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_tag_meta(tag_cat)
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#178A0F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        return self.add_tag_metas([new_tag_meta])

    def add_tag_metas(self, new_tag_metas: List[TagMeta]) -> ProjectMeta:
        """
        Adds given TagMetas to ProjectMeta.

        :param new_tag_metas: List of TagMeta objects.
        :type new_tag_metas: List[TagMeta]
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            tag_dog = sly.TagMeta('dog_tag', sly.TagValueType.ANY_STRING)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_tag_metas([tag_cat, tag_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#0F248A",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         },
            #         {
            #             "name": "dog_tag",
            #             "value_type": "any_string",
            #             "color": "#8A5C0F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        return self.clone(tag_metas=self.tag_metas.add_items(new_tag_metas))

    @staticmethod
    def _delete_items(collection, item_names):
        """
        :param collection: ObjClassCollection or TagMetaCollection instance
        :param item_names: list of item names to delete
        :return: list of items, which are in collection and not in given list of items to delete
        """
        names_to_delete = set(item_names)
        res_items = []
        for item in collection:
            if item.key() not in names_to_delete:
                res_items.append(item)
        return res_items

    def delete_obj_class(self, obj_class_name: str) -> ProjectMeta:
        """
        Removes given ObjClass by name from ProjectMeta.

        :param obj_class_name: ObjClass name.
        :type obj_class_name: str
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_obj_class(class_cat)
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#268A0F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }

            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.delete_obj_class('cat')
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": []
            # }
        """
        return self.delete_obj_classes([obj_class_name])

    def delete_obj_classes(self, obj_class_names: List[str]) -> ProjectMeta:
        """
        Removes given ObjClasses by names from ProjectMeta.

        :param obj_class_names: List of ObjClasses names.
        :type obj_class_names: List[str]
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            class_dog = sly.ObjClass('dog', sly.Bitmap)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_obj_classes([class_cat, class_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#8A0F18",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#0F8A7F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }

            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.delete_obj_classes(['cat', 'dog'])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": []
            # }
        """
        res_items = self._delete_items(self._obj_classes, obj_class_names)
        return self.clone(obj_classes=ObjClassCollection(res_items))

    def delete_tag_meta(self, tag_name: str) -> ProjectMeta:
        """
        Removes given TagMeta by name from ProjectMeta.

        :param tag_name: TagMeta name.
        :type tag_name: str
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_tag_meta(tag_cat)
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#8A540F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }

            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.delete_tag_meta('cat_tag')
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": []
            # }
        """
        return self.delete_tag_metas([tag_name])

    def delete_tag_metas(self, tag_names: List[str]) -> ProjectMeta:
        """
        Removes given TagMetas by names from ProjectMeta.

        :param tag_names: List of TagMetas names.
        :type tag_names: List[TagMeta]
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            tag_dog = sly.TagMeta('dog_tag', sly.TagValueType.ANY_STRING)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_tag_metas([tag_cat, tag_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#0F298A",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         },
            #         {
            #             "name": "dog_tag",
            #             "value_type": "any_string",
            #             "color": "#8A410F",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }

            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.delete_tag_metas(['cat_tag', 'dog_tag'])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": []
            # }
        """
        res_items = self._delete_items(self._tag_metas, tag_names)
        return self.clone(tag_metas=TagMetaCollection(res_items))

    def get_obj_class(self, obj_class_name: str) -> ObjClass:
        """
        Get given ObjClass by name from ProjectMeta.

        :param obj_class_name: ObjClass name.
        :type obj_class_name: str
        :return: ObjClass object
        :rtype: :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            class_dog = sly.ObjClass('dog', sly.Bitmap)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_obj_classes([class_cat, class_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#8A140F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#0F8A35",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }

            class_cat = meta.get_obj_class('cat')
            print(class_cat)
            # Output:
            # Name:  cat       Shape: Rectangle    Color: [138, 20, 15]  Geom. settings: {}              Hotkey

            class_elephant = meta.get_obj_class('elephant')
            print(class_elephant)
            # Output:
            # None
        """
        return self._obj_classes.get(obj_class_name)

    def get_tag_meta(self, tag_name: str) -> TagMeta:
        """
        Get given TagMeta by name from ProjectMeta.

        :param tag_name: TagMeta name.
        :type tag_name: str
        :return: TagMeta object.
        :rtype: :class:`TagMeta<supervisely.annotation.tag_meta.TagMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            tag_cat = sly.TagMeta('cat_tag', sly.TagValueType.ANY_STRING)
            tag_dog = sly.TagMeta('dog_tag', sly.TagValueType.ANY_STRING)
            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.add_tag_metas([tag_cat, tag_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [],
            #     "tags": [
            #         {
            #             "name": "cat_tag",
            #             "value_type": "any_string",
            #             "color": "#590F8A",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         },
            #         {
            #             "name": "dog_tag",
            #             "value_type": "any_string",
            #             "color": "#0F8A88",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }

            tag_cat = meta.get_tag_meta('cat_tag')
            print(tag_cat)
            # Output:
            # Name:  cat_tag                  Value type:any_string    Possible values:None       Hotkey                  Applicable toall        Applicable classes[]

            tag_elephant = meta.get_tag_meta('elephant_tag')
            print(tag_elephant)
            # Output:
            # None
        """
        return self._tag_metas.get(tag_name)

    @staticmethod
    def merge_list(metas: List[ProjectMeta]) -> ProjectMeta:
        """
        Merge ProjectMetas from given list of ProjectMetas into single ProjectMeta object.

        :param metas: List of ProjectMeta objects.
        :type metas: List[ProjectMeta]
        :return: New instance of ProjectMeta object
        :rtype: :class:`ProjectMeta<ProjectMeta>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()

            meta_1 = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Rectangle)
            meta_1 = meta_1.add_obj_class(class_cat)

            meta_2 = sly.ProjectMeta()
            tag_dog = sly.TagMeta('dog_tag', sly.TagValueType.ANY_STRING)
            meta_2 = meta_2.add_tag_meta(tag_dog)

            # Remember that ProjectMeta object is immutable, and we need to assign new instance of ProjectMeta to a new variable
            meta = meta.merge_list([meta_1, meta_2])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#0F8A45",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": [
            #         {
            #             "name": "dog_tag",
            #             "value_type": "any_string",
            #             "color": "#320F8A",
            #             "hotkey": "",
            #             "applicable_type": "all",
            #             "classes": []
            #         }
            #     ]
            # }
        """
        res_meta = ProjectMeta()
        for meta in metas:
            res_meta = res_meta.merge(meta)
        return res_meta

    def __str__(self):
        result = "ProjectMeta:\n"
        result += "Object Classes\n{}\n".format(str(self._obj_classes))
        result += "Tags\n{}\n".format(str(self._tag_metas))
        return result

    def __eq__(self, other: ProjectMeta):
        if self.obj_classes == other.obj_classes and self.tag_metas == other.tag_metas:
            return True
        return False

    def __ne__(self, other: ProjectMeta):
        return not self == other

    def to_segmentation_task(
        self, keep_geometries: Optional[List] = [Polygon, Bitmap], target_classes=None
    ) -> Tuple[ProjectMeta, Dict[ObjClass, ObjClass]]:
        """
        Convert project meta classes geometries with keep_geometries types to Bitmaps and create new ProjectMeta.

        :param keep_geometries: List of geometries that can be converted.
        :type keep_geometries: List, optional
        :return: New project meta and dict correspondences of old classes to new
        :rtype: :class:`Tuple[ProjectMeta, Dict[ObjClass, ObjClass]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Polygon)
            class_dog = sly.ObjClass('dog', sly.Bitmap)
            meta = meta.add_obj_classes([class_cat, class_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "polygon",
            #             "color": "#208A0F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#8A570F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }

            res_meta, mapping = meta.to_segmentation_task()
            res_meta_json = res_meta.to_json()
            print(json.dumps(res_meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "bitmap",
            #             "color": "#208A0F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#8A570F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }
        """
        mapping = {}
        res_classes = []
        for obj_class in self.obj_classes:
            obj_class: ObjClass

            if target_classes is None or obj_class.name in target_classes:
                if obj_class.geometry_type in keep_geometries:
                    if obj_class.geometry_type == Bitmap:
                        mapping[obj_class] = obj_class
                        res_classes.append(obj_class)
                    else:
                        new_class = obj_class.clone(geometry_type=Bitmap)
                        mapping[obj_class] = new_class
                        res_classes.append(new_class)
                else:
                    mapping[obj_class] = None
            else:
                mapping[obj_class] = None

        res_meta = self.clone(obj_classes=ObjClassCollection(res_classes))
        return res_meta, mapping

    def to_detection_task(
        self, convert_classes: Optional[bool] = False
    ) -> Tuple[ProjectMeta, Dict[ObjClass, ObjClass]]:
        """
        Convert project meta classes geometries to Rectangles or skip them and create new ProjectMeta.

        :param convert_classes: Convert classes with no Rectangle type to Rectangle or skip them.
        :type convert_classes: bool, optional
        :return: New project meta and dict correspondences of old classes to new
        :rtype: :class:`Tuple[ProjectMeta, Dict[ObjClass, ObjClass]]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            meta = sly.ProjectMeta()
            class_cat = sly.ObjClass('cat', sly.Polygon)
            class_dog = sly.ObjClass('dog', sly.Bitmap)
            meta = meta.add_obj_classes([class_cat, class_dog])
            meta_json = meta.to_json()
            print(json.dumps(meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "polygon",
            #             "color": "#208A0F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "bitmap",
            #             "color": "#8A570F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }

            res_meta, mapping = meta.to_detection_task(convert_classes=True)
            res_meta_json = res_meta.to_json()
            print(json.dumps(res_meta_json, indent=4))
            # Output: {
            #     "classes": [
            #         {
            #             "title": "cat",
            #             "shape": "rectangle",
            #             "color": "#3A0F8A",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         },
            #         {
            #             "title": "dog",
            #             "shape": "rectangle",
            #             "color": "#8A310F",
            #             "geometry_config": {},
            #             "hotkey": ""
            #         }
            #     ],
            #     "tags": []
            # }
        """
        mapping = {}
        res_classes = []
        for obj_class in self.obj_classes:
            obj_class: ObjClass
            if obj_class.geometry_type == Rectangle:
                mapping[obj_class] = obj_class
                res_classes.append(obj_class)
            else:
                if convert_classes is True:
                    new_class = obj_class.clone(geometry_type=Rectangle)
                    mapping[obj_class] = new_class
                    res_classes.append(new_class)
                else:
                    # ignore class
                    mapping[obj_class] = None
        res_meta = self.clone(obj_classes=ObjClassCollection(res_classes))
        return res_meta, mapping
