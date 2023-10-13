# coding: utf-8
"""base class for :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`  and :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` instances"""


# docs
from __future__ import annotations
from typing import List, Dict, Optional, Any


from prettytable import PrettyTable
from supervisely._utils import take_with_default
from typing import List, Iterable
from collections import defaultdict


class DuplicateKeyError(KeyError):
    r"""Raised when trying to add already existing key to
    :class:`KeyIndexedCollection <supervisely.collection.key_indexed_collection.KeyIndexedCollection>`"""
    pass


class KeyObject:
    """
    Base class fo objects that should implement ``key`` method. Child classes then can be stored in KeyIndexedCollection.
    """
    def key(self):
        raise NotImplementedError()


class KeyIndexedCollection:
    """
    Base class for :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`  and :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` instances.
    It is an analogue of python's standard Dict. It allows to store objects inherited from :class:`KeyObject <supervisely.collection.key_indexed_collection.KeyObject>`.

    :param items: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`  and :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
    :type items: list, optional
    :raises: :class:`DuplicateKeyError<supervisely.collection.key_indexed_collection.DuplicateKeyError>`, when trying to add object with already existing key

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
        item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
        collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
        print(collection.to_json())
        # Output: [
        #     {
        #         "name": "cat",
        #         "value_type": "none",
        #         "color": "#8A0F12",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     },
        #     {
        #         "name": "turtle",
        #         "value_type": "any_string",
        #         "color": "#8A860F",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     }
        # ]

        # Try to add item with a key that already exists in the collection
        dublicate_item = sly.ObjClass('cat', sly.Rectangle)
        new_collection = collection.add(dublicate_item)
        # Output:
        # supervisely.collection.key_indexed_collection.DuplicateKeyError: "Key 'cat' already exists"

        # Add item with a key that not exist in the collection
        item_dog = sly.ObjClass('dog', sly.Rectangle)
        new_collection = collection.add(item_dog)
        print(new_collection.to_json())
        # Output: [
        #     {
        #         "name": "cat",
        #         "value_type": "none",
        #         "color": "#668A0F",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     },
        #     {
        #         "name": "turtle",
        #         "value_type": "any_string",
        #         "color": "#4D0F8A",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     },
        #     {
        #         "title": "dog",
        #         "shape": "rectangle",
        #         "color": "#0F7F8A",
        #         "geometry_config": {},
        #         "hotkey": ""
        #     }
        # ]
    """

    item_type = KeyObject
    """
    The type of items that can be storred in collection. Defaul value is 
    :class:`KeyObject <supervisely.collection.key_indexed_collection.KeyObject>`. 
    Field has to be overridden in child class. Before adding object to collection its type is compared with 
    ``item_type`` and ``TypeError`` exception is raised if it differs. Collection is immutable.
    """
    def __init__(self, items: Optional[List[KeyObject]]=None):
        self._collection = {}
        self._add_items_impl(self._collection, take_with_default(items, []))

    def _add_impl(self, dst_collection, item):
        """
        Add given item to given collection. Raise error if type of item not KeyObject or item with an item with that name is already in given collection
        :param dst_collection: dictionary
        :param item: ObjClass, TagMeta or Tag class object
        :return: dictionary
        """
        if not isinstance(item, KeyIndexedCollection.item_type):
            raise TypeError(
                'Item type ({!r}) != {!r}'.format(type(item).__name__, KeyIndexedCollection.item_type.__name__))
        if item.key() in dst_collection:
            raise DuplicateKeyError('Key {!r} already exists'.format(item.key()))
        dst_collection[item.key()] = item

    def _add_items_impl(self, dst_collection, items):
        """
        Add items from input list to given collection. Raise error if type of item not KeyObject or item with an item with that name is already in given collection
        :param dst_collection: dictionary
        :param items: list of ObjClass, TagMeta or Tag class objects
        """
        for item in items:
            self._add_impl(dst_collection, item)

    def add(self, item: KeyObject) -> KeyIndexedCollection:
        """
        Add given item to collection.

        :param item: :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`  or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` object.
        :type item: KeyObject
        :return: New instance of KeyIndexedCollection
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            # Remember that KeyIndexedCollection object is immutable, and we need to assign new instance of KeyIndexedCollection to a new variable
            item_dog = sly.ObjClass('dog', sly.Rectangle)
            new_collection = collection.add(item_dog)
        """
        return self.clone(items=[*self.items(), item])

    def add_items(self, items: List[KeyObject]) -> KeyIndexedCollection:
        """
        Add items from given list to collection.

        :param items: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
        :type items:  List[KeyObject]
        :return: New instance of KeyIndexedCollection
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            # Remember that KeyIndexedCollection object is immutable, and we need to assign new instance of KeyIndexedCollection to a new variable
            item_dog = sly.ObjClass('dog', sly.Rectangle)
            item_mouse = sly.ObjClass('mouse', sly.Bitmap)
            new_collection = collection.add_items([item_dog, item_mouse])
        """
        return self.clone(items=[*self.items(), *items])

    def get(self, key: str, default: Optional[Any]=None) -> KeyObject:
        """
        Get item from collection with given key(name).

        :param items: Name of KeyObject in collection.
        :type items:  str
        :param default: The value that is returned if there is no key in the collection.
        :type default: optional
        :return: :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` object
        :rtype: :class:`KeyObject<KeyObject>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])

            item_cat = collection.get('cat')
            print(item_cat)
            # Output:
            # Name:  cat                      Value type:none          Possible values:None       Hotkey                  Applicable toall        Applicable classes[]

            item_not_exist = collection.get('no_item', {1: 2})
            print(item_not_exist)
            # Output:
            # {1: 2}
        """
        return self._collection.get(key, default)

    def __next__(self):
        for value in self._collection.values():
            yield value

    def __iter__(self):
        return next(self)

    def __contains__(self, item):
        return (isinstance(item, KeyIndexedCollection.item_type)
                and item == self._collection.get(item.key()))

    def __len__(self):
        return len(self._collection)

    def items(self) -> List[KeyObject]:
        """
        Get list of all items in collection.

        :return: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects
        :rtype: :class:`List[KeyObject]`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            items = collection.items()
            print(items)
            # Output:
            # [<supervisely.annotation.tag_meta.TagMeta object at 0x7fd08eae4340>,
            #  <supervisely.annotation.tag_meta.TagMeta object at 0x7fd08eae4370>]
        """
        return list(self._collection.values())

    def clone(self, items: Optional[List[KeyObject]]=None) -> KeyIndexedCollection:
        """
        Makes a copy of KeyIndexedCollection with new fields, if fields are given, otherwise it will use fields of the original KeyIndexedCollection.

        :param items: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
        :type items:  List[KeyObject], optional
        :return: New instance of KeyIndexedCollection
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            # Remember that KeyIndexedCollection object is immutable, and we need to assign new instance of KeyIndexedCollection to a new variable
            new_collection = collection.clone()
        """
        return type(self)(items=(items if items is not None else self.items()))

    def keys(self) -> List[str]:
        """
        Get list of all keys(item names) in collection.

        :return: List of collection keys
        :rtype: :class:`List[str]`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            keys = collection.keys() # ['cat', 'turtle']
        """
        return list(self._collection.keys())

    def has_key(self, key: str) -> bool:
        """
        Check if given key(item name exist in collection).

        :param key: The key to look for in the collection.
        :type key:  str
        :return: Is the key in the collection or not
        :rtype: :class:`bool`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])

            collection.has_key('cat') # True
            collection.has_key('hamster') # False
        """
        return key in self._collection

    def intersection(self, other: List[KeyObject]) -> KeyIndexedCollection:
        """
        Find intersection of given list of instances with collection items.

        :param key: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
        :type key:  List[KeyObject]
        :raises: :class:`ValueError` if find items with same keys(item names)
        :return: KeyIndexedCollection object
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])

            item_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            items = [item_dog, item_turtle]

            intersection = collection.intersection(items)
            print(intersection.to_json())
            # Output: [
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#760F8A",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        common_items = []
        for other_item in other:
            our_item = self.get(other_item.key())
            if our_item is not None:
                if our_item != other_item:
                    raise ValueError("Different values for the same key {!r}".format(other_item.key()))
                else:
                    common_items.append(our_item)
        return self.clone(common_items)

    def difference(self, other: List[KeyObject]) -> KeyIndexedCollection:
        """
        Find difference between collection and given list of instances.

        :param key: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
        :type key:  List[KeyObject]
        :return: KeyIndexedCollection object
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])

            item_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            items = [item_dog, item_turtle]

            diff = collection.difference(items)
            print(diff.to_json())
            # Output: [
            #     {
            #         "name": "cat",
            #         "value_type": "none",
            #         "color": "#8A150F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        items = [item for item in self.items() if item not in other]
        return self.clone(items)

    def merge(self, other: KeyIndexedCollection) -> KeyIndexedCollection:
        """
        Merge collection and other KeyIndexedCollection object.

        :param key: KeyIndexedCollection object.
        :type key:  KeyIndexedCollection
        :raises: :class:`ValueError` if item name from given list is in collection but items in both are different
        :return: KeyIndexedCollection object
        :rtype: :class:`KeyIndexedCollection<KeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])

            item_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_dog, item_turtle])

            merge = collection.merge(other_collection)
            print(merge.to_json())
            # Output: [
            #     {
            #         "name": "dog",
            #         "value_type": "none",
            #         "color": "#8A6C0F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "cat",
            #         "value_type": "none",
            #         "color": "#0F4A8A",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#4F0F8A",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        new_items = []
        for other_item in other.items():
            our_item = self.get(other_item.key())
            if our_item is None:
                new_items.append(other_item)
            elif our_item != other_item:
                raise ValueError('Error during merge for key {!r}: values are different'.format(other_item.key()))
        return self.clone(new_items + self.items())

    def __str__(self):
        res_table = PrettyTable()
        res_table.field_names = self.item_type.get_header_ptable()
        for item in self:
            res_table.add_row(item.get_row_ptable())
        return res_table.get_string()

    def to_json(self) -> List[Dict]:
        """
        Convert the KeyIndexedCollection to a json serializable list.

        :return: List of json serializable dicts
        :rtype: :class:`List[dict]`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            collection = sly.collection.key_indexed_collection.KeyIndexedCollection([item_cat, item_turtle])
            collection_json = collection.to_json()
            # Output: [
            #     {
            #         "name": "cat",
            #         "value_type": "none",
            #         "color": "#8A0F12",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#8A860F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        return [item.to_json() for item in self]

    def __eq__(self, other: KeyIndexedCollection):
        if len(self) != len(other):
            return False
        for cur_item in self:
            other_item = other.get(cur_item.key())
            if other_item is None or cur_item != other_item:
                return False
        return True

    def __ne__(self, other: KeyIndexedCollection):
        return not self == other


class MultiKeyIndexedCollection(KeyIndexedCollection):
    """
    Base class for :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` instances. MultiKeyIndexedCollection makes it possible to add an object with an already existing key.

    :param items: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>`  and :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
    :type items: list, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
        item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
        # Create item with same key 'cat'
        other_cat = sly.ObjClass('cat', sly.Rectangle)
        collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])
        print(collection.to_json())
        # Output: [
        #     {
        #         "name": "cat",
        #         "value_type": "none",
        #         "color": "#0F198A",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     },
        #     {
        #         "title": "cat",
        #         "shape": "rectangle",
        #         "color": "#0F8A6B",
        #         "geometry_config": {},
        #         "hotkey": ""
        #     },
        #     {
        #         "name": "turtle",
        #         "value_type": "any_string",
        #         "color": "#0F658A",
        #         "hotkey": "",
        #         "applicable_type": "all",
        #         "classes": []
        #     }
        # ]
    """
    def __init__(self, items: Optional[List]=None):
        self._collection = defaultdict(list)
        self._add_items_impl(self._collection, take_with_default(items, []))

    def _add_impl(self, dst_collection, item):
        if not isinstance(item, MultiKeyIndexedCollection.item_type):
            raise TypeError(
                'Item type ({!r}) != {!r}'.format(type(item).__name__, MultiKeyIndexedCollection.item_type.__name__))
        dst_collection[item.key()].append(item)

    def get(self, key: str, default: Optional[Any]=None) -> KeyObject:
        """
        Get item from collection with given key(name). If there are many values for the same key, the first value will be returned.

        :param items: Name of KeyObject in collection.
        :type items:  str
        :param default: The value that is returned if there is no key in the collection.
        :type default: optional
        :return: :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` object
        :rtype: :class:`KeyObject<KeyObject>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])
            item = collection.get('cat')
            print(item)
            # Output:
            # Name:  cat                      Value type:none          Possible values:None       Hotkey                  Applicable toall        Applicable classes[]
        """
        result = self._collection.get(key, default)
        if not result:
            return None
        return result[0]

    def get_all(self, key: str, default: Optional[List[Any]]=[]) -> List[KeyObject]:
        """
        Get item from collection with given key(name). If there are many values for the same key,all values will be returned by list.

        :param items: Name of KeyObject in collection.
        :type items:  str
        :param default: The value that is returned if there is no key in the collection.
        :type default: List, optional
        :return: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects or empty list
        :rtype: :class:`List[KeyObject]` or :class:`list`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])
            items = collection.get('cat')
            print(items)
            # Output:
            # [<supervisely.annotation.tag_meta.TagMeta object at 0x7f0278662340>, <supervisely.annotation.obj_class.ObjClass object at 0x7f02786623a0>]
        """
        return self._collection.get(key, default)

    def __next__(self):
        for tag_list in self._collection.values():
            for tag in tag_list:
                yield tag

    def __contains__(self, item):
        return (isinstance(item, MultiKeyIndexedCollection.item_type)
                and item in self.get_all(item.key()))

    def __len__(self):
        return sum([len(tag_list) for tag_list in self._collection.values()])

    def items(self) -> List[KeyObject]:
        """
        Get list of all items in collection.

        :return: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects
        :rtype: :class:`List[KeyObject]`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])
            print(collection.items())
            # Output:
            # [<supervisely.annotation.tag_meta.TagMeta object at 0x7fdbd28ce340>,
            #  <supervisely.annotation.obj_class.ObjClass object at 0x7fdbd28ce3a0>,
            #  <supervisely.annotation.tag_meta.TagMeta object at 0x7fdbd28ce370>]
        """
        res = []
        for tag_list in self._collection.values():
            res.extend(tag_list)
        return res

    def intersection(self, other: List[KeyObject]) -> MultiKeyIndexedCollection:
        """
        Find intersection of given list of instances with collection items.

        :param key: List of :class:`ObjClassCollection<supervisely.annotation.obj_class_collection.ObjClassCollection>`, :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` or :class:`TagCollection<supervisely.annotation.tag_collection.TagCollection>` objects.
        :type key:  List[KeyObject]
        :raises: :class:`ValueError` if find items with same keys(item names)
        :return: MultiKeyIndexedCollection object
        :rtype: :class:`MultiKeyIndexedCollection<MultiKeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])

            # Note, item_cat_2 have same key as item_cat, but another value
            item_cat_2 = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            items = [item_cat_2, item_turtle]

            intersect = collection.intersection(items)
            print(intersect.to_json())
            # Output: [
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#5B8A0F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        common_items = []
        for other_item in other:
            key_list = self.get_all(other_item.key())
            for our_item in key_list:
                if our_item == other_item:
                    common_items.append(our_item)
        return self.clone(common_items)

    def merge(self, other: List[KeyObject]) -> MultiKeyIndexedCollection:
        """
        Merge collection with other MultiKeyIndexedCollection object.

        :param key: MultiKeyIndexedCollection object.
        :type key:  MultiKeyIndexedCollection
        :return: MultiKeyIndexedCollection object
        :rtype: :class:`MultiKeyIndexedCollection<MultiKeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])

            item_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_dog, item_turtle])

            merge = collection.merge(other_collection)
            print(merge.to_json())
            # Output: [
            #     {
            #         "name": "cat",
            #         "value_type": "none",
            #         "color": "#198A0F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "title": "cat",
            #         "shape": "rectangle",
            #         "color": "#898A0F",
            #         "geometry_config": {},
            #         "hotkey": ""
            #     },
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#650F8A",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#0F8A83",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "dog",
            #         "value_type": "none",
            #         "color": "#1A8A0F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        new_items = [*self.items(), *other.items()]
        return self.clone(items=new_items)

    def merge_without_duplicates(self, other: MultiKeyIndexedCollection) -> MultiKeyIndexedCollection:
        """
        Merge collection with other MultiKeyIndexedCollection object. Duplicates will be ignored.

        :param key: MultiKeyIndexedCollection object.
        :type key:  MultiKeyIndexedCollection
        :raises: :class:`ValueError` if item name from given MultiKeyIndexedCollection is in collection but items in both are different
        :return: MultiKeyIndexedCollection object
        :rtype: :class:`MultiKeyIndexedCollection<MultiKeyIndexedCollection>`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            item_cat = sly.TagMeta('cat', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_cat = sly.ObjClass('cat', sly.Rectangle)
            collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_cat, item_turtle, other_cat])

            item_dog = sly.TagMeta('dog', sly.TagValueType.NONE)
            item_turtle = sly.TagMeta('turtle', sly.TagValueType.ANY_STRING)
            other_collection = sly.collection.key_indexed_collection.MultiKeyIndexedCollection([item_dog, item_turtle])

            merge = collection.merge_without_duplicates(other_collection)
            print(merge.to_json())
            # Output: [
            #     {
            #         "name": "dog",
            #         "value_type": "none",
            #         "color": "#8A0F37",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "name": "cat",
            #         "value_type": "none",
            #         "color": "#778A0F",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     },
            #     {
            #         "title": "cat",
            #         "shape": "rectangle",
            #         "color": "#8A0F76",
            #         "geometry_config": {},
            #         "hotkey": ""
            #     },
            #     {
            #         "name": "turtle",
            #         "value_type": "any_string",
            #         "color": "#850F8A",
            #         "hotkey": "",
            #         "applicable_type": "all",
            #         "classes": []
            #     }
            # ]
        """
        return super().merge(other)
