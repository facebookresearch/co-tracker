from io import StringIO
from pathlib import Path
from typing import List, Optional, Dict

from supervisely.app import StateJson, DataJson
from supervisely.app.widgets import Widget, Editor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

def initialize():
    try:
        # imports here because ordinar import cause error https://github.com/supervisely/issues/issues/1872
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap
    except ModuleNotFoundError:
        raise ModuleNotFoundError('This dependency not provided by Supervisely SDK.\nPlease, install it manually if nedeed.\npip install ruamel.yaml')
    
    class MyYAML(YAML):
        def dump(self, data, stream=None, **kw):
            inefficient = False
            if stream is None:
                inefficient = True
                stream = StringIO()
            YAML.dump(self, data, stream, **kw)
            if inefficient:
                return stream.getvalue()
    return MyYAML(), CommentedMap

class TabsDynamic(Widget):
    class TabPane:
        def __init__(self, label: str, content: Widget):
            self.label = label
            self.name = label  # identifier corresponding to the active tab
            self.content = content

    def __init__(
        self,
        filepath_or_raw_yaml: str,
        type: Optional[Literal["card", "border-card"]] = "border-card",
        disabled: Optional[bool] = False,
        widget_id=None,
    ):  
        self._disabled = disabled
        if Path(filepath_or_raw_yaml[-50:]).is_file():
            data_source = open(filepath_or_raw_yaml, "r")
        else:
            data_source = filepath_or_raw_yaml

        yaml, CommentedMap = initialize()
        self._data = yaml.load(data_source)
        self._common_data = self._data.copy()
        
        self._items_dict = {}
        self._items = []
        for label, yaml_fragment in self._data.items():
            if isinstance(yaml_fragment, CommentedMap):
                yaml_str = yaml.dump(yaml_fragment)
                editor = Editor(yaml_str, language_mode='yaml', height_px=250)
                if self._disabled:
                    editor.readonly = True
                self._items_dict[label] = editor
                self._items.append(TabsDynamic.TabPane(label=label, content=editor))
                del self._common_data[label]


        if len(self._common_data) > 0:
            yaml_str = yaml.dump(self._common_data)
            editor = Editor(yaml_str, language_mode='yaml', height_px=250)
            if self._disabled:
                editor.readonly = True
            self._items_dict['hyparameters'] = editor
            self._items.append(TabsDynamic.TabPane(label='hyparameters', content=editor))

        assert len(set(self._items_dict.keys())) == len(self._items_dict.keys()), ValueError("All of tab labels should be unique.")
        assert len(self._items_dict.keys()) == len(self._items), ValueError("labels length must be equal to contents length in Tabs widget.")

        self._value = list(self._items_dict.keys())[0]
        self._type = type
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "type": self._type,
            "disabled": self._disabled
        }

    def get_json_state(self) -> Dict:
        return {"value": self._value}

    def set_active_tab(self, value: str):
        self._value = value
        StateJson()[self.widget_id]["value"] = self._value
        StateJson().send_changes()

    def get_active_tab(self) -> str:
        return StateJson()[self.widget_id]["value"]
    
    def get_merged_yaml(self, as_dict: bool = False):
        yaml, CommentedMap = initialize()
        yaml_data = yaml.load('hyparameters:')
        for label, editor in self._items_dict.items():
            label_yaml_data = yaml.load(editor.get_text())
            if label == 'hyparameters':
                for key, value in label_yaml_data.items():
                    yaml_data[key] = value
            else:
                yaml_data[label] = label_yaml_data
        del yaml_data['hyparameters']
        if as_dict:
            return yaml_data
        return yaml.dump(yaml_data)
    
    def disable(self):
        self._disabled = True
        for key, editor in self._items_dict.items():
            editor.readonly = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        for key, editor in self._items_dict.items():
            editor.readonly = False
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()