from supervisely.api.api import Api
from supervisely.project.project_meta import ProjectMeta


class EmptyGridGallery:
    def __init__(self):
        self._options = {}
        self._init_options = False

    def to_json(self):
        return {
            "content": {
                "projectMeta": ProjectMeta().to_json(),
                "layout": [],
                "annotations": {}
            },
            "options": {}
        }