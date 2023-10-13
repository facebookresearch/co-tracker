from supervisely.app.widgets import Widget


class Empty(Widget):
    def __init__(self, widget_id: str = None):
        super().__init__(widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
