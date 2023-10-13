from supervisely.app.widgets import Widget
from supervisely.app import DataJson, StateJson
from supervisely import (
    TagMetaCollection,
    ObjClassCollection,
    TagMeta,
    ObjClass,
    TagValueType,
    color,
)
from typing import Union, List


class MatchTagMetasOrClasses(Widget):
    def __init__(
        self,
        left_collection: Union[TagMetaCollection, ObjClassCollection, None] = None,
        right_collection: Union[TagMetaCollection, ObjClassCollection, None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        selectable: bool = False,
        suffix: Union[str, None] = None,
        widget_id: str = None,
    ):
        if not type(left_collection) is type(right_collection):
            raise TypeError("Collections should be of same type")
        self._collections_type = type(left_collection)
        self._left_collection = left_collection
        self._right_collection = right_collection
        if left_name is None:
            self._left_name = (
                "Left Tag Metas" if self._collections_type is TagMetaCollection else "Left Classes"
            )
        else:
            self._left_name = left_name
        if right_name is None:
            self._right_name = (
                "Right Tag Metas"
                if self._collections_type is TagMetaCollection
                else "Right Classes"
            )
        else:
            self._right_name = right_name
        self._selectable = selectable
        self._suffix = suffix

        self._table = self._get_table()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "table": self._table,
            "left_name": self._left_name,
            "right_name": self._right_name,
            "selectable": self._selectable,
        }

    def get_json_state(self):
        return {"selected": []}

    def set(
        self,
        left_collection: Union[TagMetaCollection, ObjClassCollection, None] = None,
        right_collection: Union[TagMetaCollection, ObjClassCollection, None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        suffix: Union[str, None] = None,
    ):
        if not type(left_collection) is type(right_collection):
            raise TypeError("Collections should be of same type")
        self._collections_type = type(left_collection)
        self._left_collection = left_collection
        self._right_collection = right_collection
        self._left_name = left_name if left_name is not None else self._left_name
        self._right_name = right_name if right_name is not None else self._right_name
        self._suffix = suffix

        self._table = self._get_table()
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def get_stat(self):
        stat = {
            "match": [],
            "only_left": [],
            "only_right": [],
            "different_shape": [],
            "different_value_type": [],
            "different_one_of_options": [],
            "match_suffix": [],
            "different_shape_suffix": [],
            "different_value_type_suffix": [],
            "different_one_of_options_suffix": [],
        }
        for row in self._table:
            message_to_key = {
                "Match": "match",
                "Not found in right Project": "only_left",
                "Not found in left Project": "only_right",
                "Different shape": "different_shape",
                "Different value type": "different_value_type",
                "Type OneOf: conflict of possible values": "different_one_of_options",
                "Match with suffix": "match_suffix",
                "[Match with suffix] Different shape": "different_shape_suffix",
                "[Match with suffix] Different value type": "different_value_type_suffix",
                "[Match with suffix] Type OneOf: conflict of possible values": "different_one_of_options_suffix",
            }
            message = row.get("infoMessage")
            name1 = row.get("name1", None)
            name2 = row.get("name2", None)
            stat[message_to_key[message]].append((name1, name2))
        return stat

    def get_selected(self):
        return StateJson()[self.widget_id]["selected"]

    def _get_table(self):
        if self._left_collection is None:
            return []
        items1 = {item.name: 1 for item in self._left_collection}
        items2 = {item.name: 1 for item in self._right_collection}
        names = items1.keys() | items2.keys()
        mutual = items1.keys() & items2.keys()

        def get_mutual_with_suffix(names1, names2, suffix):
            left = {}
            right = {}
            l = len(suffix)
            for name1 in names1:
                name1: str
                if name1.endswith(suffix) and name1[:-l] in names2:
                    right[name1[:-l]] = name1
            for name2 in names2:
                name2: str
                if name2.endswith(suffix) and name2[:-l] in names1:
                    left[name2[:-l]] = name2
            return left, right

        mutual_with_suffix_left = {}
        mutual_with_suffix_right = {}
        if self._suffix is not None:
            mutual_with_suffix_left, mutual_with_suffix_right = get_mutual_with_suffix(
                items1.keys() - mutual, items2.keys() - mutual, self._suffix
            )

        diff1 = (
            items1.keys()
            - mutual
            - mutual_with_suffix_left.keys()
            - set(mutual_with_suffix_right.values())
        )
        diff2 = (
            items2.keys()
            - mutual
            - mutual_with_suffix_right.keys()
            - set(mutual_with_suffix_left.values())
        )

        match = []
        differ = []
        match_suffix = []
        differ_suffix = []
        missed = []

        def set_info(d, index, meta):
            d[f"name{index}"] = meta.name
            d[f"color{index}"] = color.rgb2hex(meta.color)
            if type(meta) is ObjClass:
                d[f"shape{index}"] = meta.geometry_type.geometry_name()
                d[f"shapeIcon{index}"] = "zmdi zmdi-shape"
            else:
                meta: TagMeta
                d[f"shape{index}"] = meta.value_type
                d[f"shapeIcon{index}"] = "zmdi zmdi-label"

        for name in names:
            compare = {}
            meta1 = self._left_collection.get(name)
            if meta1 is not None:
                set_info(compare, 1, meta1)
            meta2 = self._right_collection.get(name)
            if meta2 is not None:
                set_info(compare, 2, meta2)

            if name in mutual:
                flag = True
                if type(meta1) is ObjClass and meta1.geometry_type != meta2.geometry_type:
                    flag = False
                    diff_msg = "Different shape"
                if type(meta1) is TagMeta:
                    meta1: TagMeta
                    meta2: TagMeta
                    if meta1.value_type != meta2.value_type:
                        flag = False
                        diff_msg = "Different value type"
                    if meta1.value_type == TagValueType.ONEOF_STRING and set(
                        meta1.possible_values
                    ) != set(meta2.possible_values):
                        flag = False
                        diff_msg = "Type OneOf: conflict of possible values"

                if flag is False:
                    compare["infoMessage"] = diff_msg
                    compare["infoColor"] = "#FFBF00"
                    compare["infoIcon"] = (["zmdi zmdi-flag"],)
                    differ.append(compare)
                else:
                    compare["infoMessage"] = "Match"
                    compare["infoColor"] = "green"
                    compare["infoIcon"] = (["zmdi zmdi-check"],)
                    match.append(compare)
            elif name in mutual_with_suffix_left.keys() | mutual_with_suffix_right.keys():
                if name in mutual_with_suffix_left:
                    meta2 = self._right_collection.get(name + self._suffix)
                    set_info(compare, 2, meta2)
                else:
                    meta1 = self._left_collection.get(name + self._suffix)
                    set_info(compare, 1, meta1)
                flag = True
                if type(meta1) is ObjClass and meta1.geometry_type != meta2.geometry_type:
                    flag = False
                    diff_msg = "[Match with suffix] Different shape"
                if type(meta1) is TagMeta:
                    meta1: TagMeta
                    meta2: TagMeta
                    if meta1.value_type != meta2.value_type:
                        flag = False
                        diff_msg = "[Match with suffix] Different value type"
                    if meta1.value_type == TagValueType.ONEOF_STRING and set(
                        meta1.possible_values
                    ) != set(meta2.possible_values):
                        flag = False
                        diff_msg = "[Match with suffix] Type OneOf: conflict of possible values"

                if flag is False:
                    compare["infoMessage"] = diff_msg
                    compare["infoColor"] = "#FFBF00"
                    compare["infoIcon"] = (["zmdi zmdi-flag"],)
                    differ_suffix.append(compare)
                else:
                    compare["infoMessage"] = "Match with suffix"
                    compare["infoColor"] = "green"
                    compare["infoIcon"] = (["zmdi zmdi-check"],)
                    match_suffix.append(compare)
            elif name in diff1 | diff2:
                if name in diff1:
                    compare["infoMessage"] = "Not found in right Project"
                    compare["infoIcon"] = [
                        "zmdi zmdi-alert-circle-o",
                        "zmdi zmdi-long-arrow-right",
                    ]
                    compare["iconPosition"] = "right"
                else:
                    compare["infoMessage"] = "Not found in left Project"
                    compare["infoIcon"] = [
                        "zmdi zmdi-long-arrow-left",
                        "zmdi zmdi-alert-circle-o",
                    ]
                compare["infoColor"] = "red"
                missed.append(compare)

        table = []
        if match:
            match.sort(key=lambda x: x["name1"])
        table.extend(match)
        if differ:
            differ.sort(key=lambda x: x["name1"])
        table.extend(differ)
        if match_suffix:
            match_suffix.sort(key=lambda x: x["name1"])
        table.extend(match_suffix)
        if differ_suffix:
            differ_suffix.sort(key=lambda x: x["name1"])
        table.extend(differ_suffix)
        table.extend(missed)

        return table


class MatchTagMetas(MatchTagMetasOrClasses):
    def __init__(
        self,
        left_collection: Union[TagMetaCollection, List[TagMeta], None] = None,
        right_collection: Union[TagMetaCollection, List[TagMeta], None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        selectable: bool = False,
        suffix: Union[str, None] = None,
        widget_id: str = None,
    ):
        if type(left_collection) is list:
            left_collection = TagMetaCollection(left_collection)
        if type(right_collection) is list:
            right_collection = TagMetaCollection(right_collection)

        super().__init__(
            left_collection=left_collection,
            right_collection=right_collection,
            left_name=left_name,
            right_name=right_name,
            selectable=selectable,
            suffix=suffix,
            widget_id=widget_id,
        )

    def set(
        self,
        left_collection: Union[TagMetaCollection, List[TagMeta], None] = None,
        right_collection: Union[TagMetaCollection, List[TagMeta], None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        suffix: Union[str, None] = None,
    ):
        if type(left_collection) is list:
            left_collection = TagMetaCollection(left_collection)
        if type(right_collection) is list:
            right_collection = TagMetaCollection(right_collection)

        super().set(
            left_collection=left_collection,
            right_collection=right_collection,
            left_name=left_name,
            right_name=right_name,
            suffix=suffix,
        )

    def get_stat(self):
        stat = super().get_stat()
        fields = {
            "match",
            "only_left",
            "only_right",
            "different_value_type",
            "different_one_of_options",
            "match_suffix",
            "different_value_type_suffix",
            "different_one_of_options_suffix",
        }
        stat = {key: value for key, value in stat.items() if key in fields}
        return stat


class MatchObjClasses(MatchTagMetasOrClasses):
    def __init__(
        self,
        left_collection: Union[ObjClassCollection, List[ObjClass], None] = None,
        right_collection: Union[ObjClassCollection, List[ObjClass], None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        selectable: bool = False,
        suffix: Union[str, None] = None,
        widget_id: str = None,
    ):
        if type(left_collection) is list:
            left_collection = ObjClassCollection(left_collection)
        if type(right_collection) is list:
            right_collection = ObjClassCollection(right_collection)

        super().__init__(
            left_collection=left_collection,
            right_collection=right_collection,
            left_name=left_name,
            right_name=right_name,
            selectable=selectable,
            suffix=suffix,
            widget_id=widget_id,
        )

    def set(
        self,
        left_collection: Union[ObjClassCollection, List[ObjClass], None] = None,
        right_collection: Union[ObjClassCollection, List[ObjClass], None] = None,
        left_name: Union[str, None] = None,
        right_name: Union[str, None] = None,
        suffix: Union[str, None] = None,
    ):
        if type(left_collection) is list:
            left_collection = ObjClassCollection(left_collection)
        if type(right_collection) is list:
            right_collection = ObjClassCollection(right_collection)

        super().set(
            left_collection=left_collection,
            right_collection=right_collection,
            left_name=left_name,
            right_name=right_name,
            suffix=suffix,
        )

    def get_stat(self):
        stat = super().get_stat()
        fields = {
            "match",
            "only_left",
            "only_right",
            "different_shape",
            "match_suffix",
            "different_shape_suffix",
        }
        stat = {key: value for key, value in stat.items() if key in fields}
        return stat
