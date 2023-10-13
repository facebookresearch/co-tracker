# coding: utf-8

import os
import os.path as osp
import hashlib
import shutil

from supervisely.io import fs as sly_fs


# stateless object
class FSCache:
    def __init__(self, name, storage_root):
        self._name = name
        self._storage_root = storage_root

    @classmethod
    def _copy_file_concurr(cls, src_path, dst_path):
        try:
            sly_fs.copy_file(src_path, dst_path)
        except OSError as e:
            # may be written by parallel process, skip in that case
            if not osp.isfile(dst_path):
                raise e

    @classmethod
    def _copy_dir_recursively(cls, src_path, dst_path):
        if sly_fs.dir_exists(dst_path):
            sly_fs.remove_dir(dst_path)
        shutil.copytree(src_path, dst_path)

    @classmethod
    def _get_obj_suffix(cls, st_path):
        _, suffix = osp.splitext(st_path.rstrip("/"))
        return suffix

    def _storage_obj_exists(self, st_path, suffix):
        raise NotImplementedError()

    def _get_suffix(self, path):
        raise NotImplementedError()

    def _write_obj_impl(self, src_path, st_path):
        raise NotImplementedError()

    def _read_obj_impl(self, st_path, dst_path):
        raise NotImplementedError()

    def _rm_obj_impl(self, st_path):
        raise NotImplementedError()

    @property
    def storage_root_path(self):
        return self._storage_root

    # slow
    def list_objects(self):
        def scan_deeper(paths):
            res = []
            for left_part in paths:
                for right_part in os.listdir(left_part):
                    new_p = osp.join(left_part, right_part)
                    res.append(new_p)
            return res

        obj_paths = [self._storage_root]
        for lvl in range(3):  # fixed, determined by get_storage_path impl
            obj_paths = scan_deeper(obj_paths)

        obj_pathes_suffixes = [(p, self._get_suffix(p)) for p in obj_paths]
        return obj_pathes_suffixes

    def get_storage_path(self, data_hash, suffix=""):
        if suffix:
            suffix = ".{}".format(suffix).replace("..", ".")
        st_hash = hashlib.sha256(data_hash.encode("utf-8")).hexdigest()
        st_path = osp.join(self._storage_root, st_hash[0:2], st_hash[2:5], st_hash + suffix)
        return st_path

    def check_storage_object(self, data_hash, suffix=""):
        st_path = self.get_storage_path(data_hash, suffix)
        if self._storage_obj_exists(st_path, suffix):
            return st_path
        return None

    def write_object(self, src_path, data_hash):
        suffix = self._get_suffix(src_path)
        st_path = self.get_storage_path(data_hash, suffix)
        if not self._storage_obj_exists(st_path, suffix):
            self._write_obj_impl(src_path, st_path)

    def write_objects(self, src_paths, datas_hashes, progress_cb=None):
        for src_path, data_hash in zip(src_paths, datas_hashes):
            self.write_object(src_path, data_hash)
            if progress_cb is not None:
                progress_cb()

    def read_object(self, data_hash, dst_path):
        suffix = self._get_suffix(dst_path)
        st_path = self.check_storage_object(data_hash, suffix)
        if not st_path:
            return None  # doesn't exist
        self._read_obj_impl(st_path, dst_path)
        return dst_path

    # @TODO: to remove??? used only in data_manager->download project
    def read_objects(self, dst_paths_hashes, progress_ctr):
        written_paths = []
        for dst_path, data_hash in dst_paths_hashes:
            res_path = self.read_object(data_hash, dst_path)
            if res_path:
                written_paths.append(res_path)
                progress_ctr.iter_done_report()
        return written_paths

    def remove_object(self, st_path, suffix=""):
        removable = self._storage_obj_exists(st_path, suffix)
        if removable:
            self._rm_obj_impl(st_path)
        return removable


# class ImageStorage(FSStorage):
class FileCache(FSCache):
    def _storage_obj_exists(self, st_path, suffix):
        if not suffix:
            raise ValueError("Storage {}. File ext is empty.".format(self._name))
        return osp.isfile(st_path)

    def _get_suffix(self, path):
        return sly_fs.get_file_ext(path)

    def _write_obj_impl(self, src_path, st_path):
        sly_fs.ensure_base_path(st_path)
        self._copy_file_concurr(src_path, st_path)

    def _read_obj_impl(self, st_path, dst_path):
        sly_fs.ensure_base_path(dst_path)
        sly_fs.copy_file(st_path, dst_path)

    def _rm_obj_impl(self, st_path):
        os.remove(st_path)


class NNCache(FSCache):
    def _storage_obj_exists(self, st_path, suffix):
        if suffix:
            raise ValueError("Storage {}. Unexpected suffix for NN dir.".format(self._name))
        return osp.isdir(st_path)

    def _get_suffix(self, path):
        return ""

    def _write_obj_impl(self, src_path, st_path):
        self._copy_dir_recursively(src_path, st_path)

    def _read_obj_impl(self, st_path, dst_path):
        self._copy_dir_recursively(st_path, dst_path)

    def _rm_obj_impl(self, st_path):
        sly_fs.remove_dir(st_path)


class EmptyCache(FSCache):
    def _storage_obj_exists(self, st_path, suffix):
        return False

    def _get_suffix(self, path):
        return ""

    def _write_obj_impl(self, src_path, st_path):
        raise NotImplementedError()

    def _read_obj_impl(self, st_path, dst_path):
        raise NotImplementedError()

    def _rm_obj_impl(self, st_path):
        raise NotImplementedError()

    def write_objects(self, src_paths, datas_hashes, progress_cb=None):
        raise NotImplementedError()

    def read_objects(self, dst_paths_hashes, progress_ctr):
        return []  # overridden to speed up
