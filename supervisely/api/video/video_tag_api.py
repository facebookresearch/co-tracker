# coding: utf-8

# docs
from typing import List, Optional, Union
from supervisely.annotation.tag_meta import TagMeta
from supervisely.video_annotation.video_tag import VideoTag

from supervisely.api.module_api import ApiField
from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.video_annotation.video_tag_collection import VideoTagCollection


class VideoTagApi(TagApi):
    """
    :class:`VideoTag<supervisely.video_annotation.video_tag.VideoTag>` for a single video. :class:`VideoTagApi<VideoTagApi>` object is immutable.
    """

    _entity_id_field = ApiField.VIDEO_ID
    _method_bulk_add = "videos.tags.bulk.add"

    def remove_from_video(self, tag_id: int) -> None:
        """
        Remove tag from video.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.video.tag.remove_from_video(video_tag_id)

        """

        self._api.post("videos.tags.remove", {ApiField.ID: tag_id})

    def update_frame_range(self, tag_id: int, frame_range: List[int]) -> None:
        """
        Update VideoTag frame range in video.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :param frame_range: New VideoTag frame range.
        :type frame_range: List[int]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_frame_range = [5, 10]
            api.video.tag.update_frame_range(video_tag_id, new_frame_range)
        """

        self._api.post(
            "videos.tags.update", {ApiField.ID: tag_id, ApiField.FRAME_RANGE: frame_range}
        )

    def update_value(self, tag_id: int, tag_value: Union[str, int]) -> None:
        """
        Update VideoTag value.

        :param tag_id: VideoTag ID in Supervisely.
        :type tag_id: int
        :param tag_value: New VideoTag value.
        :type tag_value: str or int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.video.tag.update_value(video_tag_id, 'new_tag_value')
        """

        self._api.post("videos.tags.update-value", {ApiField.ID: tag_id, ApiField.VALUE: tag_value})

    def add_tag(
        self,
        project_meta_tag_id: int,
        video_id: int,
        value: Optional[Union[str, int]] = None,
        frame_range: Optional[List[int]] = None,
    ) -> int:
        """
        Add VideoTag to video.

        :param project_meta_tag_id: TagMeta ID in Supervisely.
        :type project_meta_tag_id: int
        :param video_id: Video ID in Supervidely.
        :type video_id: int
        :param value: New VideoTag value.
        :type value: str or int
        :param frame_range: New VideoTag frame range.
        :type frame_range: List[int]
        :return: None
        :rtype: dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_frame_range = [5, 10]
            api.video.tag.add_tag(project_meta_tag_id, video_id, 'tag_value', frame_range)
        """

        request_data = {ApiField.TAG_ID: project_meta_tag_id, ApiField.VIDEO_ID: video_id}
        if value:
            request_data[ApiField.VALUE] = value
        if frame_range:
            request_data[ApiField.FRAME_RANGE] = frame_range

        resp = self._api.post("videos.tags.add", request_data)
        # {'imageId': 3267369, 'tagId': 368985, 'id': 2296676, 'createdAt': '2022-09-20T11:52:33.829Z', 'updatedAt': '2022-09-20T11:52:33.829Z', 'labelerLogin': 'max'}
        return resp.json()

    def add(self, video_id: int, tag: VideoTag, update_id_inplace=True) -> int:
        """
        Add VideoTag to video.

        :param video_id: Video ID in Supervidely.
        :type video_id: int
        :param tag: VideoTag j,ject.
        :type tag: VideoTag
        :param update_id_inplace: Specify if
        :return: VideoTag ID in Supervisely
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 19402023

            tag_meta = sly.TagMeta('Animal', sly.TagValueType.NONE)
            # Tag has to exists in project.
            tag = VideoTag(tag_meta)

            api.video.tag.add(video_id=video_id, tag=tag)
        """

        from supervisely.project.project_meta import ProjectMeta

        if tag.meta.sly_id is None:
            if update_id_inplace is True:
                video_info = self._api.video.get_info_by_id(video_id)
                meta_json = self._api.project.get_meta(video_info.project_id)
                meta = ProjectMeta.from_json(meta_json)
                server_tag_meta = meta.get_tag_meta(tag.meta.name)
                if server_tag_meta is None:
                    raise KeyError(
                        f"Tag with name {tag.meta.name} not found in project with id {video_info.project_id}"
                    )
                tag.meta._set_id(server_tag_meta.sly_id)
            else:
                raise ValueError("tag_meta.sly_id is None, get updated project meta from server")

        resp_json = self.add_tag(tag.meta.sly_id, video_id, tag.value, tag.frame_range)
        tag_id = resp_json.get("id")
        created_at = resp_json.get("createdAt")
        updated_at = resp_json.get("updatedAt")
        user = resp_json.get("labelerLogin")
        if update_id_inplace is True and tag_id is not None:
            tag._set_id(tag_id)
        if update_id_inplace is True and created_at is not None:
            tag._set_created_at(created_at)
        if update_id_inplace is True and updated_at is not None:
            tag._set_updated_at(updated_at)
        if update_id_inplace is True and user is not None:
            tag._set_labeler_login(user)
        return tag_id

    def download_list(self, id: int, project_meta: ProjectMeta) -> VideoTagCollection:
        """
        Download VideoTagCollection with all tags of the video.

        :param id: Video ID in Supervidely.
        :type id: int
        :param project_meta_tag_id: TagMeta ID in Supervisely.
        :type project_meta_tag_id: int
        :return: All tags of the video in VideoTagCollection format
        :rtype: VideoTagCollection
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            video_id = 19402023
            project_id = 17209

            meta_json = api.project.get_meta(project_id)
            project_meta = sly.ProjectMeta.from_json(meta_json)

            tags_collection = api.video.tag.download_list(video_id, project_meta)
        """

        data = self._api.video.get_json_info_by_id(id, True)
        tags_json = data["tags"]
        # for tag_json in tags_json:
        #     tag_meta_id = tag_json["tagId"]
        #     tag_meta = project_meta.tag_metas.get_by_id(tag_meta_id)
        #     if tag_meta is None:
        #         raise KeyError(
        #             f"Tag meta with id={tag_meta_id} not found in project meta. Please, update project meta from server"
        #         )
        #     tag_json["name"] = tag_meta.name
        # tags = VideoTagCollection.from_json(tags_json, project_meta.tag_metas)
        # return tags
        tags = VideoTagCollection.from_api_response(tags_json, project_meta.tag_metas)
        return tags

    def remove(self, tag: VideoTag):
        """
        Remove tag from video.

        :param VideoTag in Supervisely.
        :type tag: VideoTag
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.video.tag.remove(video_tag)

        """

        if tag.sly_id is None:
            raise ValueError(
                "Only tags with ID (tag.sly_id field) can be removed. Such tags have to be downloaded from server or have ID"
            )
        self.remove_from_video(tag.sly_id)


class VideoObjectTagApi(TagApi):
    _entity_id_field = ApiField.OBJECT_ID
    _method_bulk_add = "annotation-objects.tags.bulk.add"

    def add(
        self,
        tag_meta_id: int,
        object_id: int,
        value: Optional[Union[str, int]] = None,
        frame_range: Optional[List[int]] = None,
    ) -> int:
        """Add a tag to an annotation object.

        :param tag_meta_id: TagMeta ID in project `tag_metas`
        :type tag_meta_id: int
        :param object_id: Object ID in project annotation objects
        :type object_id: int
        :param value: possible_values from TagMeta, defaults to None
        :type value: Optional[Union[str, int]], optional
        :param frame_range: array of strictly 2 frame numbers, defaults to None
        :type frame_range: Optional[List[int]], optional
        :return: ID of the tag assigned to the object
        :rtype: int
        """
        request_body = {
            ApiField.TAG_ID: tag_meta_id,
            ApiField.OBJECT_ID: object_id,
        }
        if value is not None:
            request_body[ApiField.VALUE] = value
        if frame_range is not None:
            request_body[ApiField.FRAME_RANGE] = frame_range

        response = self._api.post("annotation-objects.tags.add", request_body)
        id = response.json()[ApiField.ID]
        return id

    def remove(self, tag_id: int) -> None:
        """Remove tag from video annotation object.

        :param tag_id: tag ID of certain object
        :type tag_id: int
        """
        request_body = {ApiField.ID: tag_id}

        self._api.post("annotation-objects.tags.remove", request_body)

    def update_value(self, tag_id: int, value: Union[str, int]) -> None:
        """Update tag value for video annotation object.
        You could use only those values, which are correspond to TagMeta `value_type` and `possible_values`

        :param tag_id: tag ID of certain object
        :type tag_id: int
        :param value: possible_values from TagMeta
        :type value: Union[str, int]
        """
        request_body = {
            ApiField.ID: tag_id,
            ApiField.VALUE: value,
        }
        self._api.post("annotation-objects.tags.update-value", request_body)

    def update_frame_range(self, tag_id: int, frame_range: List[int]) -> None:
        """Update tag frames for video annotation object.

        :param tag_id: tag ID of certain object
        :type tag_id: int
        :param frame_range: range of possible frames, it must always have strictly 2 values
        :type frame_range: List[int]
        """
        request_body = {
            ApiField.ID: tag_id,
            ApiField.FRAME_RANGE: frame_range,
        }
        self._api.post("annotation-objects.tags.update", request_body)
