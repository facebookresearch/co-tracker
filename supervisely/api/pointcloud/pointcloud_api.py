# coding: utf-8

# docs
from requests import Response
from typing import List, NamedTuple, Dict, Optional, Callable, Union
from supervisely.task.progress import Progress
from tqdm import tqdm

from collections import defaultdict
from supervisely.api.module_api import ApiField, RemoveableBulkModuleApi
from supervisely.io.fs import ensure_base_path, get_file_hash
from supervisely._utils import batched

from supervisely.api.pointcloud.pointcloud_annotation_api import PointcloudAnnotationAPI
from supervisely.api.pointcloud.pointcloud_object_api import PointcloudObjectApi
from supervisely.api.pointcloud.pointcloud_figure_api import PointcloudFigureApi
from supervisely.api.pointcloud.pointcloud_tag_api import PointcloudTagApi
from requests_toolbelt import MultipartDecoder, MultipartEncoder


class PointcloudInfo(NamedTuple):
    """
    Object with :class:`Pointcloud<supervisely.pointcloud.pointcloud>` parameters from Supervisely.

    :Example:

    .. code-block:: python

        PointcloudInfo(
            id=19373403,
            frame=None,
            description='',
            name='000063.pcd',
            team_id=435,
            workspace_id=687,
            project_id=17231,
            dataset_id=55875,
            link=None,
            hash='7EcJCyhq15V4NnZ8oiPrKQckmXXypO4saqFN7kgH08Y=',
            path_original='/h5unms4-public/point_clouds/Z/h/bc/roHZP5nP2.pcd',
            cloud_mime='image/pcd',
            figures_count=4,
            objects_count=4,
            tags=[],
            meta={},
            created_at='2023-02-07T19:36:44.897Z',
            updated_at='2023-02-07T19:36:44.897Z'
        )
    """

    #: :class:`int`: Point cloud ID in Supervisely.
    id: int

    #: :class:`int`: Number of frame in the point cloud
    frame: int

    #: :class:`str`: Point cloud description.
    description: str

    #: :class:`str`: Point cloud filename.
    name: str

    #: :class:`int`: :class:`TeamApi<supervisely.api.team_api.TeamApi>` ID in Supervisely.
    team_id: int

    #: :class:`int`: :class:`WorkspaceApi<supervisely.api.workspace_api.WorkspaceApi>` ID in Supervisely.
    workspace_id: int

    #: :class:`int`: :class:`Project<supervisely.project.project.Project>` ID in Supervisely.
    project_id: int

    #: :class:`int`: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
    dataset_id: int

    #: :class:`str`: Link to point cloud.
    link: str

    #: :class:`str`: Point cloud hash obtained by base64(sha256(file_content)).
    #: Use hash for files that are expected to be stored at Supervisely or your deployed agent.
    hash: str

    #: :class:`str`: Relative storage URL to point cloud. e.g.
    #: "/h5un6l2bnaz1vms4-public/pointclouds/Z/d/HD/lfgipl...NXrg5vz.mp4".
    path_original: str

    #: :class:`str`: MIME type of the point cloud.
    cloud_mime: str

    #: :class:`int`: Number of PointcloudFigure objects in the point cloud
    figures_count: int

    #: :class:`int`: Number of PointcloudObject objects in the point cloud
    objects_count: int

    #: :class:`list`: Pointcloud :class:`PointcloudTag<supervisely.pointcloud_annotation.pointcloud_tag.PointcloudTag>` list.
    tags: list

    #: :class:`dict`: A dictionary containing point cloud metadata.
    meta: dict

    #: :class:`str`: Point cloud creation time. e.g. "2019-02-22T14:59:53.381Z".
    created_at: str

    #: :class:`str`: Time of last point cloud update. e.g. "2019-02-22T14:59:53.381Z".
    updated_at: str


class PointcloudApi(RemoveableBulkModuleApi):
    """
    API for working with :class:`Pointcloud<supervisely.pointcloud.pointcloud>`.
    :class:`PointcloudApi<PointcloudApi>` object is immutable.

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

        pcd_id = 19618654
        pcd_info = api.pointcloud.get_info_by_id(pcd_id) # api usage example
    """

    def __init__(self, api):
        """
        :param api: Api class object
        """
        super().__init__(api)
        self.annotation = PointcloudAnnotationAPI(api)
        self.object = PointcloudObjectApi(api)
        self.figure = PointcloudFigureApi(api)
        self.tag = PointcloudTagApi(api)

    @staticmethod
    def info_sequence():
        """
        Get list of all :class:`PointcloudInfo<PointcloudInfo>` field names.

        :return: List of :class:`PointcloudInfo<PointcloudInfo>` field names.`
        :rtype: :class:`list`
        """

        return [
            ApiField.ID,
            ApiField.FRAME,
            ApiField.DESCRIPTION,
            ApiField.NAME,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.LINK,
            ApiField.HASH,
            ApiField.PATH_ORIGINAL,
            # ApiField.PREVIEW,
            ApiField.CLOUD_MIME,
            ApiField.FIGURES_COUNT,
            ApiField.ANN_OBJECTS_COUNT,
            ApiField.TAGS,
            ApiField.META,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
        ]

    @staticmethod
    def info_tuple_name():
        """
        Get string name of :class:`PointcloudInfo<PointcloudInfo>` NamedTuple.

        :return: NamedTuple name.
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            tuple_name = api.pointcloud.info_tuple_name()
            print(tuple_name) # PointCloudInfo
        """

        return "PointCloudInfo"

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        res = super(PointcloudApi, self)._convert_json_info(info, skip_missing=skip_missing)
        return PointcloudInfo(**res._asdict())

    def get_list(
        self,
        dataset_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
    ) -> List[PointcloudInfo]:
        """
        Get list of information about all point cloud for a given dataset ID.

        :param dataset_id: :class:`Dataset<supervisely.project.project.Dataset>` ID in Supervisely.
        :type dataset_id: int
        :param filters: List of parameters to sort output Pointclouds. See: https://dev.supervise.ly/api-docs/#tag/Point-Clouds/paths/~1point-clouds.list/get
        :type filters: List[Dict[str, str]], optional
        :return: List of the point clouds objects from the dataset with given id.
        :rtype: :class:`List[PointcloudInfo]`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dataset_id = 62664
            pcd_infos = api.pointcloud_episode.get_list(dataset_id)
            print(pcd_infos)
            # Output: [PointcloudInfo(...), PointcloudInfo(...)]

            id_list = [19618654, 19618657, 19618660]
            filtered_pointcloud_infos = api.pointcloud.get_list(dataset_id, filters=[{'field': 'id', 'operator': 'in', 'value': id_list}])
            print(filtered_pointcloud_infos)
            # Output:
            # [PointcloudInfo(id=19618654, ...), PointcloudInfo(id=19618657, ...), PointcloudInfo(id=19618660, ...)]
        """

        return self.get_list_all_pages(
            "point-clouds.list",
            {
                ApiField.DATASET_ID: dataset_id,
                ApiField.FILTER: filters or [],
            },
        )

    def get_info_by_id(self, id: int) -> PointcloudInfo:
        """
        Get point cloud information by ID in PointcloudInfo<PointcloudInfo> format.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param raise_error: Return an error if the point cloud info was not received.
        :type raise_error: bool
        :return: Information about point cloud. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`PointcloudInfo`

        :Usage example:

         .. code-block:: python

            import supervisely as sly


            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_id = 19373403
            pcd_info = api.pointcloud.get_info_by_id(pcd_id)
            print(pcd_info)

            # Output:
            # PointcloudInfo(
            #     id=19373403,
            #     frame=None,
            #     description='',
            #     name='000063.pcd',
            #     team_id=435,
            #     workspace_id=687,
            #     project_id=17231,
            #     dataset_id=55875,
            #     link=None,
            #     hash='7EcJCyhq15V4NnZ8oiPrKQckmXXypO4saqFN7kgH08Y=',
            #     path_original='/h5unms4-public/point_clouds/Z/h/bc/roHZP5nP2.pcd',
            #     cloud_mime='image/pcd',
            #     figures_count=4,
            #     objects_count=4,
            #     tags=[],
            #     meta={},
            #     created_at='2023-02-07T19:36:44.897Z',
            #     updated_at='2023-02-07T19:36:44.897Z'
            # )
        """
        return self._get_info_by_id(id, "point-clouds.info")

    def _download(self, id: int, is_stream: Optional[bool] = False):
        """
        :param id: int
        :param is_stream: bool
        :return: Response object containing pointcloud object with given id
        """
        response = self._api.post(
            "point-clouds.download",
            {ApiField.ID: id},
            stream=is_stream,
        )
        return response

    def download_path(self, id: int, path: str) -> None:
        """
        Download point cloud with given id on the given path.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :param path: Local save path for point cloud.
        :type path: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            storage_dir = sly.app.get_data_dir()
            pcd_id = 19373403
            pcd_info = api.pointcloud.get_info_by_id(pcd_id)
            save_path = os.path.join(storage_dir, pcd_info.name)

            api.pointcloud.download_path(pcd_id, save_path)
            print(os.listdir(storage_dir))

            # Output: ['000063.pcd']
        """

        response = self._download(id, is_stream=True)
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)

    def get_list_related_images(self, id: int) -> List:
        """
        Get information about related context images.

        :param id: Point cloud ID in Supervisely.
        :type id: int
        :return: List of dictionaries with informations about related images
        :rtype: List
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_id = 19373403
            img_infos = api.pointcloud.get_list_related_images(pcd_id)
            img_info = img_infos[0]
            print(img_info)

            # Output:
            # {
            #     'pathOriginal': '/h5un6qgms4-public/images/original/S/j/hJ/PwMg.png',
            #     'id': 473302,
            #     'entityId': 19373403,
            #     'createdAt': '2023-01-09T08:50:33.225Z',
            #     'updatedAt': '2023-01-09T08:50:33.225Z',
            #     'meta': {
            #         'deviceId': 'cam_2'},
            #         'fileMeta': {'mime': 'image/png',
            #         'size': 893783,
            #         'width': 1224,
            #         'height': 370
            #     },
            #     'hash': 'vxA+emfDNUkFP9P6oitMB5Q0rMlnskmV2jvcf47OjGU=',
            #     'link': None,
            #     'preview': '/previews/q/ext:jpeg/resize:fill:50:0:0/q:50/plain/h5ad-public/images/original/S/j/hJ/PwMg.png',
            #     'fullStorageUrl': 'https://dev.supervise.ly/hs4-public/images/original/S/j/hJ/PwMg.png',
            #     'name': 'img00'
            # }
        """

        dataset_id = self.get_info_by_id(id).dataset_id
        filters = [{"field": ApiField.ENTITY_ID, "operator": "=", "value": id}]
        return self.get_list_all_pages(
            "point-clouds.images.list",
            {ApiField.DATASET_ID: dataset_id, ApiField.FILTER: filters},
            convert_json_info_cb=lambda x: x,
        )

    def download_related_image(self, id: int, path: str) -> Response:
        """
        Download a related context image from Supervisely to local directory by image id.

        :param id: Related context imgage ID in Supervisely.
        :type id: int
        :param path: Local save path for point cloud.
        :type path: str
        :return: List of dictionaries with informations about related images
        :rtype: List
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            save_path = "src/output/img_0.png"
            img_info = api.pointcloud.get_list_related_images(pcd_info.id)[0]
            api.pointcloud.download_related_image(img_info["id"], save_path)
            print(f"Context image has been successfully downloaded to '{save_path}'")

        # Output:
        # Context image has been successfully downloaded to 'src/output/img_0.png'
        """

        response = self._api.post(
            "point-clouds.images.download",
            {ApiField.ID: id},
            stream=True,
        )
        ensure_base_path(path)
        with open(path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)
        return response

    # @TODO: copypaste from video_api
    def upload_hash(
        self,
        dataset_id: int,
        name: str,
        hash: str,
        meta: Optional[Dict] = None,
    ) -> PointcloudInfo:
        """
        Upload Pointcloud from given hash to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Point cloud name.
        :type name: str
        :param hash: Point cloud hash.
        :type hash: str
        :param meta: Point cloud metadata.
        :type meta: dict, optional
        :return: Information about point cloud. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`PointcloudInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_dataset_id = 62693

            src_pointcloud_id = 19618685
            pcd_info = api.pointcloud.get_info_by_id(src_pointcloud_id)
            hash, name, meta = pcd_info.hash, pcd_info.name, pcd_info.meta

            new_pcd_info = api.pointcloud.upload_hash(dst_dataset_id.id, name, hash, meta)
            print(new_pcd_info)

            # Output:
            # PointcloudInfo(
            #     id=19619507,
            #     frame=None,
            #     description='',
            #     name='0000000031.pcd',
            #     team_id=None,
            #     workspace_id=None,
            #     project_id=None,
            #     dataset_id=62694,
            #     link=None,
            #     hash='5w69Vv1i6JrqhU0Lw1UJAJFGPVWUzDG7O3f4QSwRfmE=',
            #     path_original='/j8a9qgms4-public/point_clouds/I/3/6U/L7YBY.pcd',
            #     cloud_mime='image/pcd',
            #     figures_count=None,
            #     objects_count=None,
            #     tags=None,
            #     meta={'frame': 31},
            #     created_at='2023-04-05T10:59:44.656Z',
            #     updated_at='2023-04-05T10:59:44.656Z'
            # )
        """

        meta = {} if meta is None else meta
        return self.upload_hashes(dataset_id, [name], [hash], [meta])[0]

    # @TODO: copypaste from video_api
    def upload_hashes(
        self,
        dataset_id: int,
        names: List[str],
        hashes: List[str],
        metas: Optional[List[Dict]] = None,
        progress_cb: Optional[Callable] = None,
    ) -> List[PointcloudInfo]:
        """
        Upload point clouds from given hashes to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Point cloud name.
        :type names: List[str]
        :param hashes: Point cloud hash.
        :type hashes: List[str]
        :param metas: Point cloud metadata.
        :type metas: Optional[List[Dict]], optional
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Progress, optional
        :return: List of informations about Pointclouds. See :class:`info_sequence<info_sequence>`
        :rtype: List[:class:`PointcloudInfo`]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            src_dataset_id = 62664
            dst_dataset_id = 62690

            src_pcd_infos = api.pointcloud.get_list(src_dataset_id)

            names = [pcd.name for pcd in src_pcd_infos[:4]]
            hashes = [pcd.hash for pcd in src_pcd_infos[:4]]
            metas = [pcd.meta for pcd in src_pcd_infos[:4]]

            dst_pcd_infos = api.pointcloud.get_list(dst_dataset_id)
            print(f"{len(dst_pcd_infos)} pointcloud before upload.")
            # Output:
            # 0 pointcloud before upload.

            new_pcd_infos = api.pointcloud.upload_hashes(dst_dataset_id, names, hashes, metas)
            print(f"{len(new_pcd_infos)} pointcloud after upload.")
            # Output:
            # 4 pointcloud after upload.
        """

        return self._upload_bulk_add(
            lambda item: (ApiField.HASH, item), dataset_id, names, hashes, metas, progress_cb
        )

    # @TODO: copypaste from video_api
    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        metas=None,
        progress_cb=None,
    ):
        if metas is None:
            metas = [{}] * len(items)

        results = []
        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise RuntimeError('Can not match "names" and "items" lists, len(names) != len(items)')

        for batch in batched(list(zip(names, items, metas))):
            images = []
            for name, item, meta in batch:
                item_tuple = func_item_to_kv(item)
                images.append(
                    {
                        ApiField.NAME: name,
                        item_tuple[0]: item_tuple[1],
                        ApiField.META: meta if meta is not None else {},
                    }
                )
            response = self._api.post(
                "point-clouds.bulk.add",
                {ApiField.DATASET_ID: dataset_id, ApiField.POINTCLOUDS: images},
            )
            if progress_cb is not None:
                progress_cb(len(images))

            results.extend([self._convert_json_info(item) for item in response.json()])
        name_to_res = {img_info.name: img_info for img_info in results}
        ordered_results = [name_to_res[name] for name in names]

        return ordered_results

    def upload_related_image(self, path: str) -> str:
        """
        Upload an image to the Supervisely. It generates us a hash for image.

        :param path: Image path.
        :type path: str
        :return: Hash for image. See :class:`info_sequence<info_sequence>`
        :rtype: str
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_file = src/input/img/000000.png"
            img_hash = api.pointcloud.upload_related_image(img_file)
            print(img_hash)

            # Output:
            # +R6dFy8nMEq6k82vHLxuakpqVBmyTTPj5hXdPfjAv/c=
        """

        return self.upload_related_images([path])[0]

    def upload_related_images(
        self,
        paths: List[str],
        progress_cb: Optional[Callable] = None,
    ) -> List[str]:
        """
        Upload a batch of related images to the Supervisely. It generates us a hashes for images.

        :param paths: Images pathes.
        :type paths: List[str]
        :return: List of hashes for images. See :class:`info_sequence<info_sequence>`
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_paths = ["src/input/img/000001.png", "src/input/img/000002.png"]
            img_hashes = api.pointcloud.upload_related_images(img_paths)

            # Output:
            # [+R6dFy8nMEq6k82vHLxuakpqVBmyTTPjdfGdPfjAv/c=, +hfjbufnbkLhJb32vHLxuakpqVBmyTTPj5hXdPfhhj1c]
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        return self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)

    def add_related_images(
        self,
        images_json: List[Dict],
        camera_names: List[str] = None,
    ) -> Dict:
        """
        Attach images to point cloud.

        :param images_json: List of dictionaries with dataset id, image name, hash and meta.
        :type images_json: List[Dict]
        :param camera_names: List of camera informations.
        :type camera_names: List[Dict]
        :return: Response object
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            img_paths = ["src/input/img/000001.png", "src/input/img/000002.png"]
            cam_paths = ["src/input/cam_info/000001.json", "src/input/cam_info/000002.json"]

            img_hashes = api.pointcloud.upload_related_images(img_paths)
            img_infos = []
            for i, cam_info_file in enumerate(cam_paths):
                # reading cam_info
                with open(cam_info_file, "r") as f:
                    cam_info = json.load(f)
                img_info = {
                    "entityId": pcd_infos[i].id,
                    "name": f"img_{i}",
                    "hash": img_hashes[i],
                    "meta": cam_info,
                }
                img_infos.append(img_info)
            api.pointcloud.add_related_images(img_infos)
        """

        if camera_names is not None:
            if len(camera_names) != len(images_json):
                ValueError("camera_names length must be equal to images_json length.")
            for img_ind, camera_name in enumerate(camera_names):
                images_json[img_ind][ApiField.META]["deviceId"] = camera_name
        response = self._api.post("point-clouds.images.add", {ApiField.IMAGES: images_json})
        return response.json()

    def upload_path(
        self,
        dataset_id: int,
        name: str,
        path: str,
        meta: Optional[Dict] = None,
    ) -> PointcloudInfo:
        """
        Upload point cloud with given path to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param name: Point cloud name.
        :type name: str
        :param path: Path to point cloud.
        :type path: str
        :param meta: Dictionary with metadata for point cloud.
        :type meta: Optional[Dict]
        :return: Information about point cloud
        :rtype: PointcloudInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pcd_file = "src/input/pcd/000000.pcd"
            pcd_info = api.pointcloud.upload_path(dataset.id, name="pcd_0", path=pcd_file)
            print(f'Point cloud "{pcd_info.name}" uploaded to Supervisely with ID:{pcd_info.id}')

            # Output:
            # Point cloud "pcd_0.pcd" uploaded to Supervisely with ID:19618685
        """

        metas = None if meta is None else [meta]
        return self.upload_paths(dataset_id, [name], [path], metas=metas)[0]

    def upload_paths(
        self,
        dataset_id: int,
        names: List[str],
        paths: List[str],
        progress_cb: Optional[Callable] = None,
        metas: Optional[List[Dict]] = None,
    ) -> List[PointcloudInfo]:
        """
        Upload point clouds with given paths to Dataset.

        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param names: Point clouds names.
        :type names: List[str]
        :param paths: Paths to point clouds.
        :type paths: List[str]
        :param progress_cb: Function for tracking upload progress.
        :type progress_cb: Progress, optional
        :param metas: List of dictionary with metadata for point cloud.
        :type metas: Optional[List[Dict]]
        :return: List of informations about point clouds
        :rtype: List[PointcloudInfo]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            paths = ["src/input/pcd/000001.pcd", "src/input/pcd/000002.pcd"]
            pcd_infos = api.pointcloud.upload_paths(dataset.id, names=["pcd_1", "pcd_2"], paths=paths)
            print(f'Point clouds uploaded to Supervisely with IDs: {[pcd_info.id for pcd_info in pcd_infos]}')

            # Output:
            # Point clouds uploaded to Supervisely with IDs: [19618685, 19618686]
        """

        def path_to_bytes_stream(path):
            return open(path, "rb")

        hashes = self._upload_data_bulk(path_to_bytes_stream, get_file_hash, paths, progress_cb)
        return self.upload_hashes(dataset_id, names, hashes, metas=metas)

    def check_existing_hashes(self, hashes: List[str]) -> List[str]:
        """
        Check if point clouds with given hashes are exist.

        :param paths: Point clouds hashes.
        :type paths: List[str]
        :return: List of point clouds hashes.
        :rtype: List[str]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            pointcloud_id = 19618685
            pcd_info = api.pointcloud.get_info_by_id(pointcloud_id)
            hash = api.pointcloud.check_existing_hashes([pcd_info.hash])
            print(hash)

            # Output:
            # ['5w69Vv1i6JrqhU0Lw1UJAJFGPhgkIhs7O3f4QSwRfmE=']
        """

        results = []
        if len(hashes) == 0:
            return results
        for hashes_batch in batched(hashes, batch_size=900):
            response = self._api.post("images.internal.hashes.list", hashes_batch)
            results.extend(response.json())
        return results

    def _upload_data_bulk(self, func_item_to_byte_stream, func_item_hash, items, progress_cb):
        hashes = []
        if len(items) == 0:
            return hashes

        hash_to_items = defaultdict(list)

        for idx, item in enumerate(items):
            item_hash = func_item_hash(item)
            hashes.append(item_hash)
            hash_to_items[item_hash].append(item)

        unique_hashes = set(hashes)
        remote_hashes = self.check_existing_hashes(list(unique_hashes))
        new_hashes = unique_hashes - set(remote_hashes)

        if progress_cb is not None:
            progress_cb(len(remote_hashes))

        # upload only new images to supervisely server
        items_to_upload = []
        for hash in new_hashes:
            items_to_upload.extend(hash_to_items[hash])

        for batch in batched(items_to_upload):
            content_dict = {}
            for idx, item in enumerate(batch):
                content_dict["{}-file".format(idx)] = (
                    str(idx),
                    func_item_to_byte_stream(item),
                    "pcd/*",
                )
            encoder = MultipartEncoder(fields=content_dict)
            self._api.post("point-clouds.bulk.upload", encoder)
            if progress_cb is not None:
                progress_cb(len(batch))

        return hashes
