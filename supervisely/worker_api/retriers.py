# coding: utf-8

import sys
import time
import json
import requests


# should be stateless
class RetrierAbstract:
    def __init__(self, retry_cnt, wait_sec_first, wait_sec_max, timeout, swallow_exc=False):
        self.retry_cnt = int(retry_cnt)
        self.wait_sec = (wait_sec_first, wait_sec_max)
        if isinstance(timeout, list):  # requests lib timeout format
            self.timeout = tuple(timeout)
        else:
            self.timeout = timeout
        self.swallow_exc = swallow_exc

    def _determine_time_to_sleep(self, att_done):
        if att_done > 100:
            res = self.wait_sec[1]
        else:
            res = self.wait_sec[0] * (2 ** (att_done - 1))
            res = min(res, self.wait_sec[1])
        return res

    def _need_raise(self, att_done):
        if att_done < self.retry_cnt:
            time.sleep(self._determine_time_to_sleep(att_done))
        else:
            return not self.swallow_exc

    def request(self, cback, *args, **kwargs):
        raise NotImplementedError()


class RetrierAlways(RetrierAbstract):
    def request(self, cback, *args, **kwargs):
        for att in range(self.retry_cnt):
            try:
                return cback(*args, timeout=self.timeout, **kwargs)
            except Exception:
                if self._need_raise(att + 1):
                    raise
        return None


class RetrierAlwaysYield(RetrierAbstract):
    def request(self, cback, *args, **kwargs):
        for att in range(self.retry_cnt):
            try:
                yield from cback(*args, timeout=self.timeout, **kwargs)
                return
            except Exception:
                if self._need_raise(att + 1):
                    raise
        return None


class RetrierConnTO(RetrierAbstract):
    def request(self, cback, *args, **kwargs):
        for att in range(self.retry_cnt):
            try:
                return cback(*args, timeout=self.timeout, **kwargs)
            except (requests.ConnectionError, requests.ConnectTimeout):
                if self._need_raise(att + 1):
                    raise
        return None


class RetrierConnTOYield(RetrierAbstract):
    def request(self, cback, *args, **kwargs):
        for att in range(self.retry_cnt):
            try:
                yield from cback(*args, timeout=self.timeout, **kwargs)
                return
            except (requests.ConnectionError, requests.ConnectTimeout):
                if self._need_raise(att + 1):
                    raise
        return None


_default_retriers_config = {
    "__endless_stream_in": {
        "class": "RetrierAlwaysYield",
        "params": {
            "retry_cnt": 1000,
            "wait_sec_first": 2,
            "wait_sec_max": 4,
            "timeout": [
                4,
                30
            ]
        }
    },
    "__data_stream_in": {
        "class": "RetrierConnTOYield",
        "params": {
            "retry_cnt": 100,
            "wait_sec_first": 2,
            "wait_sec_max": 4,
            "timeout": [
                4,
                60
            ]
        }
    },
    "__data_stream_out": {
        "class": "RetrierConnTO",
        "params": {
            "retry_cnt": 100,
            "wait_sec_first": 2,
            "wait_sec_max": 4,
            "timeout": [
                4,
                60
            ]
        }
    },
    "__simple_request": {
        "class": "RetrierConnTO",
        "params": {
            "retry_cnt": 1000,
            "wait_sec_first": 1,
            "wait_sec_max": 4,
            "timeout": [
                4,
                60
            ]
        }
    },
    "Log": {
        "class": "RetrierAlways",
        "params": {
            "retry_cnt": 1000,
            "wait_sec_first": 1,
            "wait_sec_max": 1,
            "timeout": [
                4,
                30
            ],
            "swallow_exc": True
        }
    },
    "AgentConnected": {
        "class": "RetrierAlways",
        "params": {
            "retry_cnt": 1000000000,
            "wait_sec_first": 2,
            "wait_sec_max": 20,
            "timeout": [
                4,
                10
            ]
        }
    },
    "AgentPing": {
        "class": "RetrierAlways",
        "params": {
            "retry_cnt": 1000,
            "wait_sec_first": 1,
            "wait_sec_max": 4,
            "timeout": [
                4,
                10
            ]
        }
    }

}


def _retrier_from_cfg(cfg):
    cls_name = cfg['class']
    cls = getattr(sys.modules[__name__], cls_name)
    obj = cls(**cfg['params'])
    return obj


def retriers_from_cfg(cfg_path):
    if not cfg_path:
        cfg = _default_retriers_config
    else:
        cfg = json.load(open(cfg_path, 'r'))
    retriers = {selector: _retrier_from_cfg(item) for selector, item in cfg.items()}  # no validation for now
    return retriers
