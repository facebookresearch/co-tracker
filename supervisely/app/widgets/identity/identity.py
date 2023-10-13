from supervisely.app.widgets import Widget


class Identity(Widget):
    def __init__(self, content: Widget, widget_id: str = None):
        self._content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}
