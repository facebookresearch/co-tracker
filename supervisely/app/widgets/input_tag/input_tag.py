from typing import Dict, Union
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.app.widgets import Widget
from supervisely.app.widgets import Switch, Empty, Input, InputNumber, RadioGroup, OneOf, Select
from supervisely.app import DataJson


VALUE_TYPE_NAME = {
    str(TagValueType.NONE): "NONE",
    str(TagValueType.ANY_STRING): "TEXT",
    str(TagValueType.ONEOF_STRING): "ONE OF",
    str(TagValueType.ANY_NUMBER): "NUMBER",
}

VALUE_TYPES = [
    str(TagValueType.NONE),
    str(TagValueType.ANY_NUMBER),
    str(TagValueType.ANY_STRING),
    str(TagValueType.ONEOF_STRING),
]


class InputTag(Widget):
    def __init__(
        self,
        tag_meta: TagMeta,
        max_width: int = 300,
        widget_id: int = None,
    ):
        self._input_widgets = {}
        self._init_input_components()
        self._conditional_widget = Select(
            items=[
                Select.Item(value_type, content=self._input_widgets[value_type])
                for value_type in VALUE_TYPES
            ]
        )
        self._value_changed_callbacks = {}

        self._tag_meta = tag_meta
        # if TagMeta ValueType is ONEOF_STRING, then we need to set items (possible values options) for RadioGroup
        if self._tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv) for pv in self._tag_meta.possible_values]
            self._input_widgets[str(TagValueType.ONEOF_STRING)].set_items(items)
        self._conditional_widget.set_value(str(self._tag_meta.value_type))

        self._value_type_name = VALUE_TYPE_NAME[self._tag_meta.value_type]
        self._name = f"<b>{self._tag_meta.name}</b>"
        self._max_width = self._get_max_width(max_width)
        self._activation_widget = Switch()
        self._input_widget = OneOf(self._conditional_widget)

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _init_input_components(self):
        self._input_widgets[str(TagValueType.NONE)] = Empty()
        self._input_widgets[str(TagValueType.ANY_NUMBER)] = InputNumber(debounce=500)
        self._input_widgets[str(TagValueType.ANY_STRING)] = Input()
        self._input_widgets[str(TagValueType.ONEOF_STRING)] = RadioGroup(items=[])

    def _get_max_width(self, value):
        if value < 150:
            value = 150
        return f"{value}px"

    def get_tag_meta(self):
        return self._tag_meta

    def activate(self):
        self._activation_widget.on()

    def deactivate(self):
        self._activation_widget.off()

    def is_active(self):
        return self._activation_widget.is_switched()

    @property
    def value(self):
        return self._get_value()

    @value.setter
    def value(self, value):
        self._set_value(value)

    def is_valid_value(self, value):
        return self._tag_meta.is_valid_value(value)

    def set(self, tag: Union[Tag, None]):
        if tag is None:
            self._set_default_value()
            self.deactivate()
        else:
            self._set_value(tag.value)
            self.activate()

    def get_tag(self):
        if not self.is_active():
            return None
        tag_value = self._get_value()
        return Tag(self._tag_meta, tag_value)

    def _get_value(self):
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, Empty):
            return None
        else:
            return input_widget.get_value()

    def _set_value(self, value):
        if not self.is_valid_value(value):
            raise ValueError(f'Tag value "{value}" is invalid')
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, InputNumber):
            input_widget.value = value
        if isinstance(input_widget, Input):
            input_widget.set_value(value)
        if isinstance(input_widget, RadioGroup):
            input_widget.set_value(value)

    def _set_default_value(self):
        input_widget = self._input_widgets[self._tag_meta.value_type]
        if isinstance(input_widget, InputNumber):
            input_widget.value = 0
        if isinstance(input_widget, Input):
            input_widget.set_value("")
        if isinstance(input_widget, RadioGroup):
            input_widget.set_value(None)

    def get_json_data(self):
        return {
            "name": self._name,
            "valueType": self._value_type_name,
            "maxWidth": self._max_width,
        }

    def get_json_state(self) -> Dict:
        return None

    def value_changed(self, func):
        for value_type, input_widget in self._input_widgets.items():
            if isinstance(input_widget, Empty):
                self._value_changed_callbacks[value_type] = func
            else:
                self._value_changed_callbacks[value_type] = input_widget.value_changed(func)

        def inner(*args, **kwargs):
            return self._value_changed_callbacks[self._tag_meta.value_type](*args, **kwargs)

        return inner

    def selection_changed(self, func):
        return self._activation_widget.value_changed(func)

    def set_tag_meta(self, tag_meta: TagMeta):
        self._tag_meta = tag_meta
        self._value_type_name = VALUE_TYPE_NAME[self._tag_meta.value_type]
        self._name = f"<b>{self._tag_meta.name}</b>"
        # if TagMeta ValueType is ONEOF_STRING, then we need to set items (possible values options) for RadioGroup
        if self._tag_meta.value_type == str(TagValueType.ONEOF_STRING):
            items = [RadioGroup.Item(pv, pv) for pv in self._tag_meta.possible_values]
            self._input_widgets[str(TagValueType.ONEOF_STRING)].set_items(items)

        self._conditional_widget.set_value(str(self._tag_meta.value_type))
        self._set_default_value()
        self.deactivate()
        self.update_data()
        DataJson().send_changes()
