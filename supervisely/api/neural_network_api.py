# coding: utf-8
"""download/upload/manipulate neural networks"""

import os
import tarfile
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import numpy as np
import json

from supervisely.api.module_api import ApiField, CloneableModuleApi, RemoveableModuleApi
from supervisely._utils import rand_str
from supervisely.io.fs import ensure_base_path, silent_remove
from supervisely.imaging import image as sly_image
from supervisely.project.project_meta import ProjectMeta


class NeuralNetworkApi(CloneableModuleApi, RemoveableModuleApi):
    """
    """
    @staticmethod
    def info_sequence():
        """
        """
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.CONFIG,
                ApiField.HASH,
                ApiField.ONLY_TRAIN,
                ApiField.PLUGIN_ID,
                ApiField.PLUGIN_VERSION,
                ApiField.SIZE,
                ApiField.WEIGHTS_LOCATION,
                ApiField.README,
                ApiField.TASK_ID,
                ApiField.USER_ID,
                ApiField.TEAM_ID,
                ApiField.WORKSPACE_ID,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT]

    @staticmethod
    def info_tuple_name():
        """
        """
        return 'ModelInfo'

    def get_list(self, workspace_id, filters=None):
        """
        """
        return self.get_list_all_pages('models.list',  {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        """
        """
        return self._get_info_by_id(id, 'models.info')

    def download(self, id):
        """
        """
        response = self._api.post('models.download', {ApiField.ID: id}, stream=True)
        return response

    def download_to_tar(self, workspace_id, name, tar_path, progress_cb=None):
        """
        """
        model = self.get_info_by_name(workspace_id, name)
        response = self.download(model.id)
        ensure_base_path(tar_path)
        with open(tar_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                if progress_cb is not None:
                    read_mb = len(chunk) / 1024.0 / 1024.0
                    progress_cb(read_mb)

    def download_to_dir(self, workspace_id, name, directory, progress_cb=None):
        """
        """
        model_tar = os.path.join(directory, rand_str(10) + '.tar')
        self.download_to_tar(workspace_id, name, model_tar, progress_cb)
        model_dir = os.path.join(directory, name)
        with tarfile.open(model_tar) as archive:
            archive.extractall(model_dir)
        silent_remove(model_tar)
        return model_dir

    def generate_hash(self, task_id):
        """"""
        response = self._api.post('models.hash.create', {ApiField.TASK_ID: task_id})
        return response.json()

    def upload(self, hash, archive_path, progress_cb=None):
        """
        """
        encoder = MultipartEncoder({'hash': hash,
                                    'weights': (os.path.basename(archive_path), open(archive_path, 'rb'), 'application/x-tar')})

        def callback(monitor_instance):
            read_mb = monitor_instance.bytes_read / 1024.0 / 1024.0
            if progress_cb is not None:
                progress_cb(read_mb)
        monitor = MultipartEncoderMonitor(encoder, callback)
        self._api.post('models.upload', monitor)

    def inference_remote_image(self, id, image_hash, ann=None, meta=None, mode=None):
        """
        """
        data = {
            "request_type": "inference",
            "meta": meta or ProjectMeta().to_json(),
            "annotation": ann or None,
            "mode": mode or {},
            "image_hash": image_hash
        }
        fake_img_data = sly_image.write_bytes(np.zeros([5, 5, 3]), '.jpg')
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data),
                                    'image': ("img", fake_img_data, "")})
        response = self._api.post('models.infer', MultipartEncoderMonitor(encoder))
        return response.json()

    def inference(self, id, img, ann=None, meta=None, mode=None, ext=None):
        """
        """
        data = {
            "request_type": "inference",
            "meta": meta or ProjectMeta().to_json(),
            "annotation": ann or None,
            "mode": mode or {},
        }
        img_data = sly_image.write_bytes(img, ext or '.jpg')
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data),
                                    'image': ("img", img_data, "")})

        response = self._api.post('models.infer', MultipartEncoderMonitor(encoder))
        return response.json()

    def get_output_meta(self, id, input_meta=None, inference_mode=None):
        """
        """
        data = {
            "request_type": "get_out_meta",
            "meta": input_meta or ProjectMeta().to_json(),
            "mode": inference_mode or {}
        }
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data)})
        response = self._api.post('models.infer', MultipartEncoderMonitor(encoder))
        response_json = response.json()
        if 'out_meta' in response_json:
            return response_json['out_meta']
        if 'output_meta' in response_json:
            return response_json['output_meta']
        return response.json()

    def get_deploy_tasks(self, model_id):
        """
        """
        response = self._api.post('models.info.deployed', {'id': model_id})
        return [task[ApiField.ID] for task in response.json()]

    def get_training_metrics(self, model_id):
        """
        """
        response = self._get_response_by_id(id=model_id, method='tasks.train-metrics', id_field=ApiField.MODEL_ID)
        return response.json() if (response is not None) else None

    def _clone_api_method_name(self):
        """
        """
        return 'models.clone'

    def _remove_api_method_name(self):
        """
        """
        return 'models.remove'

    def create_from_checkpoint(self, task_id, checkpoint_id, model_name, change_name_if_conflict=True):
        """
        """
        # FYI: checkpoint has these fields
        # 'modelTitle': 'my_model_name_006',
        # 'status': 'uploaded'

        self._api.task._validate_checkpoints_support(task_id)
        task_info = self._api.task.get_info_by_id(task_id)
        workspace_id = task_info[ApiField.WORKSPACE_ID]
        new_model_name = self.get_free_name(workspace_id, model_name)
        if new_model_name != model_name and change_name_if_conflict is False:
            raise KeyError("Model name={!r} already exists in workspace id={!r}".format(model_name, workspace_id))
        resp = self._api.post("models.create-from-checkpoint", {ApiField.ID: checkpoint_id,
                                                                ApiField.TASK_ID: task_id,
                                                                ApiField.NAME: new_model_name})
        process_task_id = resp.json()[ApiField.TASK_ID]
        if process_task_id is not None:
            self._api.task.wait(process_task_id, self._api.task.Status.FINISHED)
        else:
            # upload process skipped because checkpoint is already uploaded to server, just new model will be created
            pass
        return new_model_name