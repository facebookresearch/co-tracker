# coding: utf-8

import json


NONE_STR = 'None'


def str_to_type_or_none(data_str, target_type):
    if data_str == NONE_STR:
        return None
    res = json.loads(data_str)
    if type(res) != target_type:
        raise TypeError("variable has to be of type {!r} or None".format(target_type))
    return res
