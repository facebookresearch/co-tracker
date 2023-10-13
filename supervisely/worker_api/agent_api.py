# coding: utf-8

import os
import struct
import requests
import traceback
import time

from supervisely.worker_api.retriers import retriers_from_cfg
from supervisely.io.network_exceptions import process_requests_exception, process_unhandled_request


class AgentAPI:
    def __init__(self, token, server_address, ext_logger, cfg_path=None):
        self.logger = ext_logger
        self.server_address = server_address
        if ('http://' not in self.server_address) and ('https://' not in self.server_address):
            self.server_address = os.path.join('http://', self.server_address)
        self.server_address = os.path.join(self.server_address, 'agent')
        self.headers = {
            'Content-type': 'application/octet-stream',
            'Accept-Encoding': 'deflate',  # to override default 'Accept-Encoding': 'gzip, deflate'
        }
        if token is not None:
            self.headers['x-token'] = token
        self._retriers = retriers_from_cfg(cfg_path)

    def _get_retrier(self, api_method_name, common_selector):
        retrier = self._retriers.get(api_method_name)
        if retrier is None:
            retrier = self._retriers[common_selector]
        return retrier

    def add_to_metadata(self, key, value):
        self.headers[key] = value

    def rm_from_metadata(self, key):
        self.headers.pop(key, None)

    def _send_request(self, api_method_name, request_data, timeout, in_stream, addit_headers):
        url = os.path.join(self.server_address, api_method_name)
        if not addit_headers:
            addit_headers = {}
        cur_header = {**self.headers, **addit_headers}
        not_log_request = api_method_name != 'Log'
        server_reply = None
        try:
            server_reply = requests.post(url, headers=cur_header, data=request_data, stream=in_stream, timeout=timeout)
            server_reply.raise_for_status()
        except requests.RequestException as exc:
            process_requests_exception(self.logger, exc, api_method_name, url,
                                       verbose=not_log_request, swallow_exc=False, response=server_reply)
        except Exception as exc:
            process_unhandled_request(self.logger, exc)

        return server_reply

    # magic value 4 means four bytes for message length
    # http://www.sureshjoshi.com/development/streaming-protocol-buffers/
    # https://www.datadoghq.com/blog/engineering/protobuf-parsing-in-python/
    def _get_input_stream(self, api_method_name, res_proto_fn, request_data, timeout, addit_headers):
        def cut_len(msg_buff_):
            cur_m_len = struct.unpack('>I', msg_buff_[0:4])[0]
            return cur_m_len, msg_buff_[4:]

        def append_to_msg_buffer(msg_len_, msg_buf_, rest_buf):
            if msg_len_ > len(msg_buf_) + len(rest_buf):
                msg_buf_ = msg_buf_ + rest_buf
                rest_buf = b""
            else:
                tmp_cut_len = msg_len_ - len(msg_buf_)
                msg_buf_ = msg_buf_ + rest_buf[0:tmp_cut_len]
                rest_buf = rest_buf[tmp_cut_len:]
            return msg_buf_, rest_buf

        # request package crashes on empty chunks
        try:
            with self._send_request(api_method_name, request_data, timeout,
                                    in_stream=True, addit_headers=addit_headers) as reply:
                msg_len = None
                msg_buf = b""
                for buffer in reply.iter_content(chunk_size=None):
                    while len(buffer) > 0:
                        if msg_len is None:
                            if len(msg_buf) < 4:
                                msg_buf, buffer = append_to_msg_buffer(4, msg_buf, buffer)
                            if len(msg_buf) < 4:
                                continue
                            msg_len, msg_buf = cut_len(msg_buf)
                        msg_buf, buffer = append_to_msg_buffer(msg_len, msg_buf, buffer)
                        if msg_len == len(msg_buf):
                            if msg_len == 0:
                                pass
                            else:
                                proto_msg = res_proto_fn()
                                proto_msg.ParseFromString(msg_buf)
                                yield proto_msg
                            msg_len = None
                            msg_buf = b""
                if msg_len is not None:
                    raise RuntimeError('MISSED_STREAM_CHUNKS')
        except requests.exceptions.ChunkedEncodingError:
            raise RuntimeError('Unknown error during stream. Please contact support.')
        except requests.RequestException:
            raise
        except Exception as e:
            self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})
            raise e

    def _put_out_stream(self, api_method_name, res_proto_fn, chunk_generator, timeout, addit_headers):
        def bindata_generator():
            for chunk in chunk_generator:
                size = chunk.ByteSize()
                res_bytes_with_len = struct.pack('>I', size) + chunk.SerializeToString()
                yield res_bytes_with_len

        resp = self._send_request(api_method_name, bindata_generator(), timeout,
                                  in_stream=False, addit_headers=addit_headers)
        res_proto = res_proto_fn()
        res_proto.ParseFromString(resp.content)
        return res_proto

    # will not log it now
    def simple_request(self, api_method_name, res_proto_fn, proto_request, addit_headers=None):
        data_to_send = proto_request.SerializeToString()
        retrier = self._get_retrier(api_method_name, '__simple_request')
        resp = retrier.request(self._send_request, api_method_name, data_to_send, in_stream=False, addit_headers=addit_headers)
        if resp is None:
            return None  # swallowed exception
        res_proto = res_proto_fn()
        res_proto.ParseFromString(resp.content)
        return res_proto

    def get_stream_with_data(self, api_method_name, res_proto_fn, proto_request, addit_headers=None):
        data_to_send = proto_request.SerializeToString()
        retrier = self._get_retrier(api_method_name, '__data_stream_in')
        yield from retrier.request(self._get_input_stream,
                                   api_method_name, res_proto_fn, data_to_send, addit_headers=addit_headers)

    def get_endless_stream(self, api_method_name, res_proto_fn, proto_request, addit_headers=None,
                           server_fail_limit=10, wait_server_sec=10):
        data_to_send = proto_request.SerializeToString()
        for attempt in range(server_fail_limit):
            retrier = self._get_retrier(api_method_name, '__endless_stream_in')
            yield from retrier.request(self._get_input_stream,
                                       api_method_name, res_proto_fn, data_to_send, addit_headers=addit_headers)
            self.logger.warn('Endless input stream end', extra={'method': api_method_name})
            if attempt != server_fail_limit - 1:
                time.sleep(wait_server_sec)

    def put_stream_with_data(self, api_method_name, res_proto_fn, chunk_generator, addit_headers=None):
        retrier = self._get_retrier(api_method_name, '__data_stream_out')
        res = retrier.request(self._put_out_stream,
                              api_method_name, res_proto_fn, chunk_generator, addit_headers=addit_headers)
        return res

    @classmethod
    def catch_http_err(cls, code_list, fn, *args, **kwargs):
        try:
            res = fn(*args, **kwargs)
            return res, None
        except requests.exceptions.HTTPError as e:
            res_code = e.response.status_code
            if res_code in code_list:
                return None, res_code
            raise
