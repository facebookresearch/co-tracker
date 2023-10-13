# coding: utf-8

# docs
from __future__ import annotations
from typing import List, NamedTuple, Optional, Dict

from supervisely.api.module_api import ApiField, ModuleApi, RemoveableBulkModuleApi
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely._utils import batched


class FigureApi(RemoveableBulkModuleApi):
    """
    Figure object for :class:`VideoAnnotation<supervisely.video_annotation.video_annotation.VideoAnnotation>`.
    """
    @staticmethod
    def info_sequence():
        """
        NamedTuple FigureInfo information about Figure.

        :Example:

         .. code-block:: python

            FigureInfo(id=588801373,
                       updated_at='2020-12-22T06:37:13.183Z',
                       created_at='2020-12-22T06:37:13.183Z',
                       entity_id=186648101,
                       object_id=112482,
                       project_id=110366,
                       dataset_id=419886,
                       frame_index=0,
                       geometry_type='bitmap',
                       geometry={'bitmap': {'data': 'eJwdlns8...Cgj4=', 'origin': [335, 205]}})
        """
        return [ApiField.ID,
                ApiField.UPDATED_AT,
                ApiField.CREATED_AT,
                ApiField.ENTITY_ID,
                ApiField.OBJECT_ID,
                ApiField.PROJECT_ID,
                ApiField.DATASET_ID,
                ApiField.FRAME_INDEX,
                ApiField.GEOMETRY_TYPE,
                ApiField.GEOMETRY
                ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of NamedTuple for class.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.video.figure.info_tuple_name()
            print(tuple_name) # FigureInfo
        """

        return "FigureInfo"

    def get_info_by_id(self, id: int) -> NamedTuple:
        """
        Get Figure information by ID.

        :param id: Figure ID in Supervisely.
        :type id: int
        :return: Information about Figure. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_id = 588801373

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            figure_info = api.video.figure.get_info_by_id(figure_id)
            print(figure_info)
            # Output: [
            #     588801373,
            #     "2020-12-22T06:37:13.183Z",
            #     "2020-12-22T06:37:13.183Z",
            #     186648101,
            #     112482,
            #     110366,
            #     419886,
            #     0,
            #     "bitmap",
            #     {
            #         "bitmap": {
            #             "data": "eJw...Cgj4=",
            #             "origin": [
            #                 335,
            #                 205
            #             ]
            #         }
            #     }
            # ]
        """
        return self._get_info_by_id(id, 'figures.info')

    def create(self, entity_id: int, object_id: int, meta: Dict, geometry_json: Dict, geometry_type, track_id: int=None):
        """"""
        input_figure = {
            ApiField.META: meta,
            ApiField.OBJECT_ID: object_id,
            ApiField.GEOMETRY_TYPE: geometry_type,
            ApiField.GEOMETRY: geometry_json,
        }

        if track_id is not None:
            input_figure[ApiField.TRACK_ID] = track_id

        body = {ApiField.ENTITY_ID: entity_id, ApiField.FIGURES: [input_figure]}

        response = self._api.post("figures.bulk.add", body)
        return response.json()[0][ApiField.ID]

    def get_by_ids(self, dataset_id: int, ids: List[int]) -> List[NamedTuple]:
        """
        Get Figures information by IDs from given dataset ID.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param ids: List of Figures IDs.
        :type ids: List[int]
        :return: List of information about Figures. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 466642
            figures_ids = [642155547, 642155548, 642155549]
            figures_infos = api.video.figure.get_by_ids(dataset_id, figures_ids)
            print(figures_infos)
            # Output: [
            #     [
            #         642155547,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         0,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2240,
            #                         1041
            #                     ],
            #                     [
            #                         2463,
            #                         1187
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ],
            #     [
            #         642155548,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         1,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2248,
            #                         1048
            #                     ],
            #                     [
            #                         2455,
            #                         1176
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ],
            #     [
            #         642155549,
            #         "2021-03-23T13:25:34.705Z",
            #         "2021-03-23T13:25:34.705Z",
            #         198703211,
            #         152118,
            #         124976,
            #         466642,
            #         2,
            #         "rectangle",
            #         {
            #             "points": {
            #                 "exterior": [
            #                     [
            #                         2237,
            #                         1046
            #                     ],
            #                     [
            #                         2464,
            #                         1179
            #                     ]
            #                 ],
            #                 "interior": []
            #             }
            #         }
            #     ]
            # ]
        """
        filters = [{"field": "id", "operator": "in", "value": ids}]
        figures_infos = self.get_list_all_pages(
            "figures.list", {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters}
        )

        if len(ids) != len(figures_infos):
            ids_downloaded = [info.id for info in figures_infos]
            raise RuntimeError(
                "Ids don't exist on server: {}".format(set(ids_downloaded) - set(ids))
            )

        id_to_item = {info.id: info for info in figures_infos}

        figures = []
        for input_id in ids:
            figures.append(id_to_item[input_id])

        return figures

    def _append_bulk(
        self,
        entity_id,
        figures_json,
        figures_keys,
        key_id_map: KeyIdMap,
        field_name=ApiField.ENTITY_ID,
    ):
        """"""
        if len(figures_json) == 0:
            return
        for (batch_keys, batch_jsons) in zip(
            batched(figures_keys, batch_size=100), batched(figures_json, batch_size=100)
        ):
            resp = self._api.post(
                "figures.bulk.add",
                {field_name: entity_id, ApiField.FIGURES: batch_jsons},
            )
            for key, resp_obj in zip(batch_keys, resp.json()):
                figure_id = resp_obj[ApiField.ID]
                key_id_map.add_figure(key, figure_id)
