# coding: utf-8

import json
from typing import Dict, Optional


class JsonSerializable:
    def to_json(self):
        """ Serialize to JSON-compatible dict.
        :return: dict
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data):
        """
        Deserialize from a JSON-compatible dict
        :param data: JSON-compatible dict
        :return: Parsed object
        """
        raise NotImplementedError()


def load_json_file(filename: str) -> Dict:
    """
    Decoding data from json file with given filename.

    :param filename: Target file path.
    :type filename: str
    :returns: Json format as a dict
    :rtype: :class:`dict`
    :Usage example:

     .. code-block:: python

        from supervisely.io.json import load_json_file
        json_example = load_json_file('/home/admin/work/projects/examples/ann.json')
        print(json_example)
        # Output: {
        #     "description": "",
        #     "tags": [],
        #     "size": {
        #         "height": 800,
        #         "width": 1067
        #     },
        #     "objects": [
        #         {
        #             "id": 619053179,
        #             "classId": 2791451,
        #             "description": "",
        #             "geometryType": "bitmap",
        #             "labelerLogin": "alexxx",
        #             "createdAt": "2021-02-10T08:36:33.898Z",
        #             "updatedAt": "2021-02-10T08:39:24.828Z",
        #             "tags": [],
        #             "classTitle": "lemon",
        #             "bitmap": {
        #                 "data": "eJwBZgOZ/IlQTkcNChoKAAAADUlIR...AiRp9+EwAAAABJRU5ErkJgggn2cM0=",
        #                 "origin": [
        #                     531,
        #                     120
        #                 ]
        #             }
        #         },
        #         {
        #             "id": 619053347,
        #             "classId": 2791453,
        #             "description": "",
        #             "geometryType": "rectangle",
        #             "labelerLogin": "alexxx",
        #             "createdAt": "2021-02-10T08:39:08.369Z",
        #             "updatedAt": "2021-02-10T08:39:24.828Z",
        #             "tags": [],
        #             "classTitle": "kiwi",
        #             "points": {
        #                 "exterior": [
        #                     [
        #                         764,
        #                         387
        #                     ],
        #                     [
        #                         967,
        #                         608
        #                     ]
        #                 ],
        #                 "interior": []
        #             }
        #         },
        #         {
        #             "id": 619053355,
        #             "classId": 2791453,
        #             "description": "",
        #             "geometryType": "rectangle",
        #             "labelerLogin": "alexxx",
        #             "createdAt": "2021-02-10T08:39:16.938Z",
        #             "updatedAt": "2021-02-10T08:39:24.828Z",
        #             "tags": [],
        #             "classTitle": "kiwi",
        #             "points": {
        #                 "exterior": [
        #                     [
        #                         477,
        #                         543
        #                     ],
        #                     [
        #                         647,
        #                         713
        #                     ]
        #                 ],
        #                 "interior": []
        #             }
        #         }
        #     ]
        # }
    """
    with open(filename, encoding='utf-8') as fin:
        return json.load(fin)


def dump_json_file(data: Dict, filename: str, indent: Optional[int]=4) -> None:
    """
    Write given data in json format in file with given name.

    :param data: Data in json format as a dict.
    :type data: dict
    :param filename: Target file path to write data.
    :type filename: str
    :param indent: Json array elements and object members will be pretty-printed with that indent level.
    :type indent: int, optional
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.json import dump_json_file
        data = {1: 'example'}
        dump_json_file(data, '/home/admin/work/projects/examples/1.json')
    """
    with open(filename, 'w') as fout:
        json.dump(data, fout, indent=indent)


def flatten_json(data: Dict, sep: Optional[str]=".") -> Dict:
    """
    Normalize semi-structured JSON data into a flat table.

    :param data: Data in json format as a dict.
    :type data: dict
    :param sep: Nested records will generate names separated by sep.
    :type sep: str, optional
    :returns: Dict
    :rtype: :class:`dict`
    """
    import pandas as pd
    df = pd.json_normalize(data, sep=sep)
    return df.to_dict(orient='records')[0]


def modify_keys(data: Dict, prefix: Optional[str]=None, suffix: Optional[str]=None) -> Dict[str, str]:
    """
    Add prefix and suffix to keys of given dict.

    :param data: Data in json format as a dict.
    :type data: dict
    :param prefix: Prefix which will be added to dict.
    :type prefix: str, optional
    :param suffix: Suffix which will be added to dict.
    :type suffix: str, optional
    :returns: New dict with prefix and suffix in keys
    :rtype: :class:`dict`
    :Usage example:

     .. code-block:: python

        from supervisely.io.json import modify_keys
        data = {'1': 'example', '3': 4}
        new_data = modify_keys(data, prefix='pr_', suffix='_su')
        print(new_data)
        # Output: {'pr_1_su': 'example', 'pr_3_su': 4}
    """
    def _modify(k):
        res = k
        if prefix is not None:
            res = prefix + res
        if suffix is not None:
            res += suffix
        return res
    return {_modify(k): v for k, v in data.items()}
