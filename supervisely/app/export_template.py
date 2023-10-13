from typing import Optional

import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.output import set_download
from supervisely.sly_logger import logger


class Export:
    class Context:
        def __init__(
            self, team_id: int, workspace_id: int, project_id: int, dataset_id: Optional[int] = None
        ):
            self._team_id = team_id
            self._workspace_id = workspace_id
            self._project_id = project_id
            self._dataset_id = dataset_id

        def __str__(self):
            return (
                f"Team ID: {self._team_id}\n"
                f"Workspace ID: {self._workspace_id}\n"
                f"Project ID: {self._project_id}\n"
                f"Dataset ID: {self._dataset_id}\n"
            )

        @property
        def team_id(self) -> int:
            return self._team_id

        @property
        def workspace_id(self) -> int:
            return self._workspace_id

        @property
        def project_id(self) -> int:
            return self._project_id

        @property
        def dataset_id(self) -> int:
            return self._dataset_id

    def process(self, context: Context) -> str:
        raise NotImplementedError("implement your own method when inherit")

    def run(self):
        api = Api.from_env()
        team_id = env.team_id()
        workspace_id = env.workspace_id()
        project_id = env.project_id(raise_not_found=False)
        dataset_id = env.dataset_id(raise_not_found=False)

        project = api.project.get_info_by_id(id=project_id)
        if project is None:
            raise ValueError(
                f"Project with ID: '{project_id}' either doesn't exist, archived or you don't have access to it"
            )
        logger.info(f"Exporting Project: id={project.id}, name={project.name}, type={project.type}")

        if dataset_id is not None:
            dataset = api.dataset.get_info_by_id(id=dataset_id)
            if dataset is None:
                raise ValueError(
                    f"Dataset with ID: '{dataset_id}' either doesn't exist, archived or you don't have access to it"
                )
            logger.info(f"Exporting Dataset: id={dataset.id}, name={dataset.name}")

        context = self.Context(
            team_id=team_id, workspace_id=workspace_id, project_id=project_id, dataset_id=dataset_id
        )

        local_path = self.process(context=context)

        if type(local_path) is not str:
            raise ValueError("Path must be a 'string'")

        set_download(local_path)
