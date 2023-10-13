# coding: utf-8
"""create or manipulate already existing labeling jobs"""

# docs
from __future__ import annotations

import time
from typing import List, NamedTuple, Dict, Optional, Callable, Union, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

from supervisely.collection.str_enum import StrEnum
from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    RemoveableBulkModuleApi,
    ModuleWithStatus,
    WaitingTimeExceeded,
)


class LabelingJobInfo(NamedTuple):
    id: int
    name: str
    readme: str
    description: str
    team_id: int
    workspace_id: int
    workspace_name: str
    project_id: int
    project_name: str
    dataset_id: int
    dataset_name: str
    created_by_id: int
    created_by_login: str
    assigned_to_id: int
    assigned_to_login: str
    reviewer_id: int
    reviewer_login: str
    created_at: str
    started_at: str
    finished_at: str
    status: str
    disabled: bool
    images_count: int
    finished_images_count: int
    rejected_images_count: int
    accepted_images_count: int
    progress_images_count: int
    classes_to_label: list
    tags_to_label: list
    images_range: tuple
    objects_limit_per_image: int
    tags_limit_per_image: int
    filter_images_by_tags: list
    include_images_with_tags: list
    exclude_images_with_tags: list
    entities: list


class LabelingJobApi(RemoveableBulkModuleApi, ModuleWithStatus):
    """
    API for working with Labeling Jobs. :class:`LabelingJobApi<LabelingJobApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

        jobs = api.labeling_job.get_list(9) # api usage example
    """

    class Status(StrEnum):
        """Labeling Job status."""

        PENDING = "pending"
        """"""
        IN_PROGRESS = "in_progress"
        """"""
        ON_REVIEW = "on_review"
        """"""
        COMPLETED = "completed"
        """"""
        STOPPED = "stopped"
        """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple LabelingJobInfo information about Labeling Job.

        :Example:

         .. code-block:: python

             LabelingJobInfo(id=2,
                             name='Annotation Job (#1) (#1) (dataset_01)',
                             readme='',
                             description='',
                             team_id=4,
                             workspace_id=8,
                             workspace_name='First Workspace',
                             project_id=58,
                             project_name='tutorial_project',
                             dataset_id=54,
                             dataset_name='dataset_01',
                             created_by_id=4,
                             created_by_login='anna',
                             assigned_to_id=4,
                             assigned_to_login='anna',
                             reviewer_id=4,
                             reviewer_login='anna',
                             created_at='2020-04-08T15:10:12.618Z',
                             started_at='2020-04-08T15:10:19.833Z',
                             finished_at='2020-04-08T15:13:39.788Z',
                             status='completed',
                             disabled=False,
                             images_count=3,
                             finished_images_count=0,
                             rejected_images_count=1,
                             accepted_images_count=2,
                             progress_images_count=2,
                             classes_to_label=[],
                             tags_to_label=[],
                             images_range=(1, 5),
                             objects_limit_per_image=None,
                             tags_limit_per_image=None,
                             filter_images_by_tags=[],
                             include_images_with_tags=[],
                             exclude_images_with_tags=[],
                             entities=None)
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.README,
            ApiField.DESCRIPTION,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.WORKSPACE_NAME,
            ApiField.PROJECT_ID,
            ApiField.PROJECT_NAME,
            ApiField.DATASET_ID,
            ApiField.DATASET_NAME,
            ApiField.CREATED_BY_ID,
            ApiField.CREATED_BY_LOGIN,
            ApiField.ASSIGNED_TO_ID,
            ApiField.ASSIGNED_TO_LOGIN,
            ApiField.REVIEWER_ID,
            ApiField.REVIEWER_LOGIN,
            ApiField.CREATED_AT,
            ApiField.STARTED_AT,
            ApiField.FINISHED_AT,
            ApiField.STATUS,
            ApiField.DISABLED,
            ApiField.IMAGES_COUNT,
            ApiField.FINISHED_IMAGES_COUNT,
            ApiField.REJECTED_IMAGES_COUNT,
            ApiField.ACCEPTED_IMAGES_COUNT,
            ApiField.PROGRESS_IMAGES_COUNT,
            ApiField.CLASSES_TO_LABEL,
            ApiField.TAGS_TO_LABEL,
            ApiField.IMAGES_RANGE,
            ApiField.OBJECTS_LIMIT_PER_IMAGE,
            ApiField.TAGS_LIMIT_PER_IMAGE,
            ApiField.FILTER_IMAGES_BY_TAGS,
            ApiField.INCLUDE_IMAGES_WITH_TAGS,
            ApiField.EXCLUDE_IMAGES_WITH_TAGS,
            ApiField.ENTITIES,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **LabelingJobInfo**.
        """
        return "LabelingJobInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        """ """
        if info is None:
            return None
        else:
            field_values = []
            for field_name in self.info_sequence():
                if field_name in [
                    ApiField.INCLUDE_IMAGES_WITH_TAGS,
                    ApiField.EXCLUDE_IMAGES_WITH_TAGS,
                ]:
                    continue
                value = None
                if type(field_name) is str:
                    if skip_missing is True:
                        value = info.get(field_name, None)
                    else:
                        value = info[field_name]
                elif type(field_name) is tuple:
                    for sub_name in field_name[0]:
                        if value is None:
                            if skip_missing is True:
                                value = info.get(sub_name, None)
                            else:
                                value = info[sub_name]
                        else:
                            value = value[sub_name]
                else:
                    raise RuntimeError("Can not parse field {!r}".format(field_name))

                if field_name == ApiField.FILTER_IMAGES_BY_TAGS:
                    field_values.append(value)
                    include_images_with_tags = []
                    exclude_images_with_tags = []
                    for fv in value:
                        key = ApiField.NAME
                        if key not in fv:
                            key = "title"
                        if fv["positive"] is True:
                            include_images_with_tags.append(fv[key])
                        else:
                            exclude_images_with_tags.append(fv[key])
                    field_values.append(include_images_with_tags)
                    field_values.append(exclude_images_with_tags)
                    continue
                elif (
                    field_name == ApiField.CLASSES_TO_LABEL or field_name == ApiField.TAGS_TO_LABEL
                ):
                    value = []
                    for fv in value:
                        key = ApiField.NAME
                        if ApiField.NAME not in fv:
                            key = "title"
                        value.append(fv[key])
                elif field_name == ApiField.IMAGES_RANGE:
                    value = (value["start"], value["end"])

                field_values.append(value)

            res = self.InfoType(*field_values)
            return LabelingJobInfo(**res._asdict())

    def _remove_batch_api_method_name(self):
        """Api remove method name."""

        return "jobs.bulk.remove"

    def _remove_batch_field_name(self):
        """Returns onstant string that represents API field name."""

        return ApiField.IDS

    def create(
        self,
        name: str,
        dataset_id: int,
        user_ids: List[int],
        readme: Optional[str] = None,
        description: Optional[str] = None,
        classes_to_label: Optional[List[str]] = None,
        objects_limit_per_image: Optional[int] = None,
        tags_to_label: Optional[List[str]] = None,
        tags_limit_per_image: Optional[int] = None,
        include_images_with_tags: Optional[List[str]] = None,
        exclude_images_with_tags: Optional[List[str]] = None,
        images_range: Optional[List[int, int]] = None,
        reviewer_id: Optional[int] = None,
        images_ids: Optional[List[int]] = [],
    ) -> List[LabelingJobInfo]:
        """
        Creates Labeling Job and assigns given Users to it.

        :param name: Labeling Job name in Supervisely.
        :type name: str
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param user_ids: User IDs in Supervisely to assign Users as labelers to Labeling Job.
        :type user_ids: List[int]
        :param readme: Additional information about Labeling Job.
        :type readme: str, optional
        :param description: Description of Labeling Job.
        :type description: str, optional
        :param classes_to_label: List of classes to label in Dataset.
        :type classes_to_label: List[str], optional
        :param objects_limit_per_image: Limit the number of objects that the labeler can create on each image.
        :type objects_limit_per_image: int, optional
        :param tags_to_label: List of tags to label in Dataset.
        :type tags_to_label: List[str], optional
        :param tags_limit_per_image: Limit the number of tags that the labeler can create on each image.
        :type tags_limit_per_image: int, optional
        :param include_images_with_tags: Include images with given tags for processing by labeler.
        :type include_images_with_tags: List[str], optional
        :param exclude_images_with_tags: Exclude images with given tags for processing by labeler.
        :type exclude_images_with_tags: List[str], optional
        :param images_range: Limit number of images to be labeled for each labeler.
        :type images_range: List[int, int], optional
        :param reviewer_id: User ID in Supervisely to assign User as Reviewer to Labeling Job.
        :type reviewer_id: int, optional
        :param images_ids: List of images ids to label in dataset
        :type images_ids: List[int], optional
        :return: List of information about new Labeling Job. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingJobInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_name = 'alex'
            dataset_id = 602
            new_label_jobs = api.labeling_job.create(user_name, dataset_id, user_ids=[111, 222], readme='Readmy text',
                                                     description='Work for labelers', objects_limit_per_image=5, tags_limit_per_image=3)
            print(new_label_jobs)
            # Output: [
            #     [
            #         92,
            #         "alex (#1) (#3)",
            #         "Readmy text",
            #         "Work for labelers",
            #         13,
            #         29,
            #         "Labelling Workspace",
            #         494,
            #         "Test Dataset",
            #         602,
            #         "ds1",
            #         8,
            #         "alex",
            #         111,
            #         "quantigo273",
            #         8,
            #         "alex",
            #         "2021-03-25T11:04:34.031Z",
            #         null,
            #         null,
            #         "pending",
            #         false,
            #         3,
            #         0,
            #         0,
            #         0,
            #         0,
            #         [],
            #         [],
            #         [
            #             null,
            #             null
            #         ],
            #         5,
            #         3,
            #         [],
            #         [],
            #         [],
            #         [
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287244,
            #                 "name": "IMG_0813"
            #             },
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287246,
            #                 "name": "IMG_0432"
            #             },
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287245,
            #                 "name": "IMG_0315"
            #             }
            #         ]
            #     ],
            #     [
            #         93,
            #         "alex (#2) (#3)",
            #         "Readmy text",
            #         "Work for labelers",
            #         13,
            #         29,
            #         "Labelling Workspace",
            #         494,
            #         "Test Dataset",
            #         602,
            #         "ds1",
            #         8,
            #         "alex",
            #         222,
            #         "quantigo19",
            #         8,
            #         "alex",
            #         "2021-03-25T11:04:34.031Z",
            #         null,
            #         null,
            #         "pending",
            #         false,
            #         3,
            #         0,
            #         0,
            #         0,
            #         0,
            #         [],
            #         [],
            #         [
            #             null,
            #             null
            #         ],
            #         5,
            #         3,
            #         [],
            #         [],
            #         [],
            #         [
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287248,
            #                 "name": "IMG_8454"
            #             },
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287249,
            #                 "name": "IMG_6896"
            #             },
            #             {
            #                 "reviewStatus": "none",
            #                 "id": 287247,
            #                 "name": "IMG_1942"
            #             }
            #         ]
            #     ]
            # ]
        """
        if classes_to_label is None:
            classes_to_label = []
        if tags_to_label is None:
            tags_to_label = []

        filter_images_by_tags = []
        if include_images_with_tags is not None:
            for tag_name in include_images_with_tags:
                filter_images_by_tags.append({"name": tag_name, "positive": True})

        if exclude_images_with_tags is not None:
            for tag_name in exclude_images_with_tags:
                filter_images_by_tags.append({"name": tag_name, "positive": False})

        if objects_limit_per_image is None:
            objects_limit_per_image = 0

        if tags_limit_per_image is None:
            tags_limit_per_image = 0

        data = {
            ApiField.NAME: name,
            ApiField.DATASET_ID: dataset_id,
            ApiField.USER_IDS: user_ids,
            # ApiField.DESCRIPTION: description,
            ApiField.META: {
                "classes": classes_to_label,
                "projectTags": tags_to_label,
                "imageTags": filter_images_by_tags,
                "imageFiguresLimit": objects_limit_per_image,
                "imageTagsLimit": tags_limit_per_image,
                "entityIds": images_ids,
            },
        }

        if readme is not None:
            data[ApiField.README] = str(readme)

        if description is not None:
            data[ApiField.DESCRIPTION] = str(description)

        if images_range is not None:
            if len(images_range) != 2:
                raise RuntimeError("images_range has to contain 2 elements (start, end)")
            images_range = {"start": images_range[0], "end": images_range[1]}
            data[ApiField.META]["range"] = images_range

        if reviewer_id is not None:
            data[ApiField.REVIEWER_ID] = reviewer_id

        response = self._api.post("jobs.add", data)
        # created_jobs_json = response.json()

        created_jobs = []
        for job in response.json():
            created_jobs.append(self.get_info_by_id(job[ApiField.ID]))
        return created_jobs

    def get_list(
        self,
        team_id: int,
        created_by_id: Optional[int] = None,
        assigned_to_id: Optional[int] = None,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        show_disabled: Optional[bool] = False,
    ) -> List[LabelingJobInfo]:
        """
        Get list of information about Labeling Job in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param created_by_id: ID of the User who created the LabelingJob.
        :type created_by_id: int, optional
        :param assigned_to_id: ID of the assigned User.
        :type assigned_to_id: int, optional
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int, optional
        :param show_disabled: Show disabled Labeling Jobs.
        :type show_disabled: bool, optional
        :return: List of information about Labeling Jobs. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingJobInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_jobs = api.labeling_job.get_list(4)
            print(label_jobs)
            # Output: [
            #     [
            #         2,
            #         "Annotation Job (#1) (#1) (dataset_01)",
            #         "",
            #         "",
            #         4,
            #         8,
            #         "First Workspace",
            #         58,
            #         "tutorial_project",
            #         54,
            #         "dataset_01",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         "2020-04-08T15:10:12.618Z",
            #         "2020-04-08T15:10:19.833Z",
            #         "2020-04-08T15:13:39.788Z",
            #         "completed",
            #         false,
            #         3,
            #         0,
            #         1,
            #         2,
            #         2,
            #         [],
            #         [],
            #         [
            #             1,
            #             5
            #         ],
            #         null,
            #         null,
            #         [],
            #         [],
            #         [],
            #         null
            #     ],
            #     [
            #         3,
            #         "Annotation Job (#1) (#2) (dataset_02)",
            #         "",
            #         "",
            #         4,
            #         8,
            #         "First Workspace",
            #         58,
            #         "tutorial_project",
            #         55,
            #         "dataset_02",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         "2020-04-08T15:10:12.618Z",
            #         "2020-04-08T15:15:46.749Z",
            #         "2020-04-08T15:17:33.572Z",
            #         "completed",
            #         false,
            #         2,
            #         0,
            #         0,
            #         2,
            #         2,
            #         [],
            #         [],
            #         [
            #             1,
            #             5
            #         ],
            #         null,
            #         null,
            #         [],
            #         [],
            #         [],
            #         null
            #     ]
            # ]
        """
        filters = []
        if created_by_id is not None:
            filters.append(
                {"field": ApiField.CREATED_BY_ID[0][0], "operator": "=", "value": created_by_id}
            )
        if assigned_to_id is not None:
            filters.append(
                {"field": ApiField.ASSIGNED_TO_ID[0][0], "operator": "=", "value": assigned_to_id}
            )
        if project_id is not None:
            filters.append({"field": ApiField.PROJECT_ID, "operator": "=", "value": project_id})
        if dataset_id is not None:
            filters.append({"field": ApiField.DATASET_ID, "operator": "=", "value": dataset_id})
        return self.get_list_all_pages(
            "jobs.list",
            {ApiField.TEAM_ID: team_id, "showDisabled": show_disabled, ApiField.FILTER: filters},
        )

    def stop(self, id: int) -> None:
        """
        Makes Labeling Job unavailable for labeler with given User ID.

        :param id: User ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.stop(9)
        """
        self._api.post("jobs.stop", {ApiField.ID: id})

    def get_info_by_id(self, id: int) -> LabelingJobInfo:
        """
        Get Labeling Job information by ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: Information about Labeling Job. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`LabelingJobInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_job_info = api.labeling_job.get_info_by_id(2)
            print(label_job_info)
            # Output: [
            #     2,
            #     "Annotation Job (#1) (#1) (dataset_01)",
            #     "",
            #     "",
            #     4,
            #     8,
            #     "First Workspace",
            #     58,
            #     "tutorial_project",
            #     54,
            #     "dataset_01",
            #     4,
            #     "anna",
            #     4,
            #     "anna",
            #     4,
            #     "anna",
            #     "2020-04-08T15:10:12.618Z",
            #     "2020-04-08T15:10:19.833Z",
            #     "2020-04-08T15:13:39.788Z",
            #     "completed",
            #     false,
            #     3,
            #     0,
            #     1,
            #     2,
            #     2,
            #     [],
            #     [],
            #     [
            #         1,
            #         5
            #     ],
            #     null,
            #     null,
            #     [],
            #     [],
            #     [],
            #     [
            #         {
            #             "reviewStatus": "rejected",
            #             "id": 283,
            #             "name": "image_03"
            #         },
            #         {
            #             "reviewStatus": "accepted",
            #             "id": 282,
            #             "name": "image_02"
            #         },
            #         {
            #             "reviewStatus": "accepted",
            #             "id": 281,
            #             "name": "image_01"
            #         }
            #     ]
            # ]
        """
        return self._get_info_by_id(id, "jobs.info")

    def archive(self, id: int) -> None:
        """
        Archives Labeling Job with given ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.archive(23)
        """
        self._api.post("jobs.archive", {ApiField.ID: id})

    def get_status(self, id: int) -> LabelingJobApi.Status:
        """
        Get status of Labeling Job with given ID.

        :param id: Labeling job ID in Supervisely.
        :type id: int
        :return: Labeling Job Status
        :rtype: :class:`Status<supervisely.api.labeling_job_api.LabelingJobApi.Status>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            job_status = api.labeling_job.get_status(4)
            print(job_status) # pending
        """
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        """ """
        # there is no ERROR status for labeling job
        pass

    def wait(
        self,
        id: int,
        target_status: str,
        wait_attempts: Optional[int] = None,
        wait_attempt_timeout_sec: Optional[int] = None,
    ) -> None:
        """
        Wait for a Labeling Job to change to the expected target status.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param target_status: Expected result status of Labeling Job.
        :type target_status: str
        :param wait_attempts: Number of attempts to retry, when :class:`WaitingTimeExceeded` raises.
        :type wait_attempts: int, optional
        :param wait_attempt_timeout_sec: Time between attempts.
        :type wait_attempt_timeout_sec: int, optional
        :raises: :class:`WaitingTimeExceeded`, if waiting time exceeded
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.wait(4, 'completed', wait_attempts=2, wait_attempt_timeout_sec=1)
            # supervisely.api.module_api.WaitingTimeExceeded: Waiting time exceeded
        """
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status is target_status:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded("Waiting time exceeded")

    def get_stats(self, id: int) -> Dict:
        """
        Get stats of given Labeling Job ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: Dict with information about given Labeling Job
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            status = api.labeling_job.get_stats(3)
            print(status)
            # Output: {
            #     "job": {
            #         "editingDuration": 0,
            #         "annotationDuration": 720,
            #         "id": 3,
            #         "name": "Annotation Job (#1) (#2) (dataset_02)",
            #         "startedAt": "2020-04-08T15:15:46.749Z",
            #         "finishedAt": "2020-04-08T15:17:33.572Z",
            #         "imagesCount": 2,
            #         "finishedImagesCount": 2,
            #         "tagsStats": [
            #             {
            #                 "id": 24,
            #                 "color": "#ED68A1",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "car_color"
            #             },
            #             {
            #                 "id": 19,
            #                 "color": "#A0A08C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "cars_number"
            #             },
            #             {
            #                 "id": 20,
            #                 "color": "#D98F7E",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "like"
            #             },
            #             {
            #                 "id": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "person_gender"
            #             },
            #             {
            #                 "parentId": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "person_gender (male)"
            #             },
            #             {
            #                 "parentId": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "person_gender (female)"
            #             },
            #             {
            #                 "id": 21,
            #                 "color": "#855D79",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "situated"
            #             },
            #             {
            #                 "parentId": 21,
            #                 "color": "#855D79",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "situated (inside)"
            #             },
            #             {
            #                 "parentId": 21,
            #                 "color": "#855D79",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "situated (outside)"
            #             },
            #             {
            #                 "id": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "vehicle_age"
            #             },
            #             {
            #                 "parentId": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "vehicle_age (modern)"
            #             },
            #             {
            #                 "parentId": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "vehicle_age (vintage)"
            #             }
            #         ]
            #     },
            #     "classes": [
            #         {
            #             "id": 43,
            #             "color": "#F6FF00",
            #             "shape": "rectangle",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "bike",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 42,
            #             "color": "#BE55CE",
            #             "shape": "polygon",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "car",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 41,
            #             "color": "#FD0000",
            #             "shape": "polygon",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "dog",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 40,
            #             "color": "#00FF12",
            #             "shape": "bitmap",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "person",
            #             "labelsCount": 0
            #         }
            #     ],
            #     "images": {
            #         "total": 2,
            #         "images": [
            #             {
            #                 "id": 285,
            #                 "reviewStatus": "accepted",
            #                 "annotationDuration": 0,
            #                 "totalDuration": 0,
            #                 "name": "image_01",
            #                 "labelsCount": 0
            #             },
            #             {
            #                 "id": 284,
            #                 "reviewStatus": "accepted",
            #                 "annotationDuration": 0,
            #                 "totalDuration": 0,
            #                 "name": "image_02",
            #                 "labelsCount": 0
            #             }
            #         ]
            #     }
            # }
        """
        response = self._api.post("jobs.stats", {ApiField.ID: id})
        return response.json()

    def get_activity(
        self, team_id: int, job_id: int, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> DataFrame:
        import pandas as pd

        """
        Get all activity for given Labeling Job by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param job_id: Labeling Job ID in Supervisely.
        :type job_id: int
        :param progress_cb: Function for tracking progress
        :type progress_cb: tqdm, optional
        :return: Activity data as `pd.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        :rtype: :class:`pd.DataFrame`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            activity = api.labeling_job.get_activity(3)
            print(activity)
            # Output:
            #   userId         action  ... tagId                 meta
            # 0       4  update_figure  ...   NaN                   {}
            # 1       4  create_figure  ...   NaN                   {}
            # 2       4     attach_tag  ...  20.0                   {}
            # 3       4     attach_tag  ...  21.0  {'value': 'inside'}
            # 4       4     attach_tag  ...  24.0      {'value': '12'}
            # 5       4  update_figure  ...   NaN                   {}
            # 6       4  update_figure  ...   NaN                   {}
            # 7       4  update_figure  ...   NaN                   {}
            # 8       4  create_figure  ...   NaN                   {}
            # 9       4  update_figure  ...   NaN                   {}
            # [10 rows x 18 columns]
        """
        activity = self._api.team.get_activity(
            team_id, filter_job_id=job_id, progress_cb=progress_cb
        )
        df = pd.DataFrame(activity)
        return df

    def set_status(self, id: int, status: str) -> None:
        """
        Sets Labeling Job status.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param status: New Labeling Job status
        :type status: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.labeling_job_api.LabelingJobApi.Status import COMPLETED

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.set_status(id=9, status="completed")
        """
        self._api.post("jobs.set-status", {ApiField.ID: id, ApiField.STATUS: status})
