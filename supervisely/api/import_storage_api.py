# coding: utf-8
import os
from supervisely.api.module_api import ModuleApiBase, ApiField
from supervisely.io.fs import ensure_base_path, get_file_name_with_ext
from requests_toolbelt import MultipartEncoder
import mimetypes


class ImportStorageApi(ModuleApiBase):
    def get_meta_by_hashes(self, hashes):
        """
        """
        response = self._api.post('import-storage.internal.meta.list', {ApiField.HASHES: hashes})
        return response.json()
