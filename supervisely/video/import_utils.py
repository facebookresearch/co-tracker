# coding: utf-8

import os
import pathlib


def get_dataset_name(file_path, default="ds0"):
    """
    Find names of top-most directories in a hierarchy of import directory. Files from root import directory will be placed to dataset with name "ds0".
    :param file_path: str (path to import directory)
    :param default: str
    :return: str
    """
    dir_path = os.path.split(file_path)[0]
    ds_name = default
    path_parts = pathlib.Path(dir_path).parts
    if len(path_parts) != 1:
        ds_name = path_parts[1]
    return ds_name
