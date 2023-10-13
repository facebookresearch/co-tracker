# coding: utf-8

import os
import concurrent.futures
import json
import traceback
from copy import deepcopy
from queue import Queue
import threading

from supervisely.annotation.annotation import Annotation
from supervisely.function_wrapper import function_wrapper, function_wrapper_nofail
from supervisely.imaging.image import drop_image_alpha_channel
from supervisely.nn.hosted.inference_modes import InferenceModeFactory, InfModeFullImage, \
    MODE, NAME, get_effective_inference_mode_config
from supervisely.project.project_meta import ProjectMeta
from supervisely.worker_api.agent_api import AgentAPI
from supervisely.worker_api.agent_rpc import decode_image, download_image_from_remote, download_data_from_remote, \
    send_from_memory_generator
from supervisely.worker_api.interfaces import SingleImageInferenceInterface
from supervisely.worker_proto import worker_api_pb2 as api_proto
from supervisely.task.progress import report_agent_rpc_ready
from supervisely.api.api import Api


REQUEST_TYPE = 'request_type'
GET_OUT_META = 'get_out_meta'
INFERENCE = 'inference'
SUPPORTED_REQUEST_TYPES = [GET_OUT_META, INFERENCE]

MODEL_RESULT_SUFFIX = '_model'

DATA = 'data'
REQUEST_ID = 'request_id'
IMAGE_HASH = 'image_hash'

VIDEO = 'video'
VIDEO_ID = 'video_id'
FRAME_INDEX = 'frame_index'


class ConnectionClosedByServerException(Exception):
    pass


class AgentRPCServicerBase:
    NETW_CHUNK_SIZE = 1048576
    QUEUE_MAX_SIZE = 2000  # Maximum number of in-flight requests to avoid exhausting server memory.

    def __init__(self, logger, model_applier: SingleImageInferenceInterface, conn_config, cache):
        self.logger = logger
        self.server_address = conn_config['server_address']
        self.api = AgentAPI(token=conn_config['token'],
                            server_address=self.server_address,
                            ext_logger=self.logger)
        self.api.add_to_metadata('x-task-id', conn_config['task_id'])

        self.model_applier = model_applier
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.download_queue = Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.final_processing_queue = Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.image_cache = cache
        self._default_inference_mode_config = InfModeFullImage.make_default_config(
            model_result_suffix=MODEL_RESULT_SUFFIX)
        self.logger.info('Created AgentRPCServicer', extra=conn_config)

    def _load_image_from_sly(self, req_id, image_hash, src_node_token):
        self.logger.trace('Will look for image.', extra={
            REQUEST_ID: req_id, IMAGE_HASH: image_hash, 'src_node_token': src_node_token
        })
        img_data = self.image_cache.get(image_hash)
        if img_data is None:
            img_data_packed = download_image_from_remote(self.api, image_hash, src_node_token, self.logger)
            img_data = decode_image(img_data_packed)
            self.image_cache.add(image_hash, img_data)

        return img_data

    def _load_arbitrary_image(self, req_id):
        self.logger.trace('Will load arbitrary image.', extra={REQUEST_ID: req_id})
        img_data_packed = download_data_from_remote(self.api, req_id, self.logger)
        img_data = decode_image(img_data_packed)
        return img_data

    def _load_data_if_required(self, event_obj):
        try:
            req_id = event_obj[REQUEST_ID]
            event_data = event_obj[DATA]
            request_type = event_data.get(REQUEST_TYPE, INFERENCE)
            if request_type == INFERENCE:
                frame_info = event_data.get(VIDEO, None)
                if frame_info is None:
                    # For inference we need to download an image and add it to the event data.
                    image_hash = event_data.get(IMAGE_HASH)
                    if image_hash is None:
                        img_data = self._load_arbitrary_image(req_id)
                    else:
                        src_node_token = event_obj[DATA].get('src_node_token', '')
                        img_data = self._load_image_from_sly(req_id, image_hash, src_node_token)
                    event_data['image_arr'] = img_data
                    self.logger.trace('Input image is obtained.', extra={REQUEST_ID: req_id})
                else:
                    # download frame
                    video_id = frame_info[VIDEO_ID]
                    frame_index = frame_info[FRAME_INDEX]

                    image_uniq_key = "video_{}_frame_{}.png".format(video_id, frame_index)
                    img_data = self.image_cache.get(image_uniq_key)
                    if img_data is None:
                        api_token = event_data['api_token']
                        public_api = Api(self.server_address, api_token, retry_count=20, external_logger=self.logger)
                        img_data = public_api.video.frame.download_np(video_id, frame_index)
                        self.image_cache.add(image_uniq_key, img_data)

                    event_data['image_arr'] = img_data
                    self.logger.trace('Frame is obtained.', extra={REQUEST_ID: req_id})

            self.final_processing_queue.put(item=(event_data, req_id))
        except Exception as e:
            res_msg = {}
            self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})
            res_msg.update({'success': False, 'error': json.dumps(traceback.format_exc())})
            self.thread_pool.submit(function_wrapper_nofail, self._send_data, res_msg, req_id)  # skip errors

    def _send_data(self, out_msg, req_id):
        self.logger.trace('Will send output data.', extra={REQUEST_ID: req_id})
        out_bytes = json.dumps(out_msg).encode('utf-8')

        self.api.put_stream_with_data('SendGeneralEventData',
                                      api_proto.Empty,
                                      send_from_memory_generator(out_bytes, self.NETW_CHUNK_SIZE),
                                      addit_headers={'x-request-id': req_id})
        self.logger.trace('Output data is sent.', extra={REQUEST_ID: req_id})

    def _final_processing(self, in_msg):
        request_type = in_msg.get(REQUEST_TYPE, INFERENCE)

        if request_type == INFERENCE:
            img = in_msg['image_arr']
            if len(img.shape) != 3 or img.shape[2] not in [3, 4]:
                raise RuntimeError('Expect 3- or 4-channel image RGB(RGBA) [0..255].')
            elif img.shape[2] == 4:
                img = drop_image_alpha_channel(img)
            return self._do_single_img_inference(img, in_msg)
        elif request_type == GET_OUT_META:
            return {'out_meta': self._get_out_meta(in_msg).to_json()}
        else:
            raise RuntimeError('Unknown request type: {}. Only the following request types are supported: {}'.format(
                request_type, SUPPORTED_REQUEST_TYPES))

    def _do_single_img_inference(self, img, in_msg):
        raise NotImplementedError()

    def _get_out_meta(self, in_msg):
        raise NotImplementedError()

    def _sequential_final_processing(self):
        while True:
            in_msg, req_id = self.final_processing_queue.get(block=True, timeout=None)
            res_msg = {}
            try:
                res_msg.update(self._final_processing(in_msg))
                res_msg.update({'success': True})
            except Exception as e:
                self.logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})
                res_msg.update({'success': False, 'error': json.dumps(traceback.format_exc())})

            self.thread_pool.submit(function_wrapper_nofail, self._send_data, res_msg, req_id)  # skip errors

    def _load_data_loop(self):
        while True:
            event_obj = self.download_queue.get(block=True, timeout=None)
            self._load_data_if_required(event_obj)

    def run_inf_loop(self):
        def seq_inf_wrapped():
            function_wrapper(self._sequential_final_processing)  # exit if raised

        load_data_thread = threading.Thread(target=self._load_data_loop, daemon=True)
        load_data_thread.start()
        inference_thread = threading.Thread(target=seq_inf_wrapped, daemon=True)
        inference_thread.start()
        report_agent_rpc_ready()

        for gen_event in self.api.get_endless_stream('GetGeneralEventsStream',
                                                     api_proto.GeneralEvent, api_proto.Empty()):
            try:
                request_id = gen_event.request_id

                data = {}
                if gen_event.data is not None and gen_event.data != b'':
                    data = json.loads(gen_event.data.decode('utf-8'))

                event_obj = {REQUEST_ID: request_id, DATA: data}
                self.logger.debug('GET_INFERENCE_CALL', extra=event_obj)
                self.download_queue.put(event_obj, block=True)
            except Exception as error:
                self.logger.warning('Inference exception: ', extra={"error_message": str(error)})
                res_msg = {'success': False, 'error': json.dumps(str(error))}
                self.thread_pool.submit(function_wrapper_nofail, self._send_data, res_msg, request_id)

        raise ConnectionClosedByServerException('Requests stream to a deployed model closed by the server.')


class AgentRPCServicer(AgentRPCServicerBase):
    @staticmethod
    def _in_project_meta_from_msg(in_msg):
        pr_meta_json = in_msg.get('meta')
        return ProjectMeta.from_json(pr_meta_json) if pr_meta_json is not None else None

    def _make_inference_mode(self, msg_inference_mode_config, in_project_meta):
        inference_mode_config = get_effective_inference_mode_config(
            msg_inference_mode_config, deepcopy(self._default_inference_mode_config))
        return InferenceModeFactory.create(inference_mode_config, in_project_meta, self.model_applier)

    def _do_single_img_inference(self, img, in_msg):
        in_project_meta = self._in_project_meta_from_msg(in_msg)
        ann_json = in_msg.get('annotation')
        if ann_json is not None:
            if in_project_meta is None:
                raise ValueError('In order to perform inference with annotation you must specify the appropriate'
                                 ' project meta.')
            ann = Annotation.from_json(ann_json, in_project_meta)
        else:
            in_project_meta = in_project_meta or ProjectMeta()
            ann = Annotation(img.shape[:2])

        inference_mode = self._make_inference_mode(in_msg.get(MODE, {}), in_project_meta)
        inference_result = inference_mode.infer_annotate(img, ann)
        return inference_result.to_json()

    def _get_out_meta(self, in_msg):
        in_meta = self._in_project_meta_from_msg(in_msg) or ProjectMeta()
        inference_mode = self._make_inference_mode(in_msg.get(MODE, {}), in_meta)
        return inference_mode.out_meta


class SmarttoolRPCServicer(AgentRPCServicerBase):
    def _do_single_img_inference(self, img, in_msg):
        inference_result = self.model_applier.inference(img, in_msg)
        return inference_result.to_json()

    def _get_out_meta(self, in_msg):
        return self.model_applier.get_out_meta()


class InactiveRPCServicer(AgentRPCServicer):
    def __init__(self, logger, model_applier: SingleImageInferenceInterface, conn_config, cache):
        self.logger = logger
        self.model_applier = model_applier
        self._default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix=MODEL_RESULT_SUFFIX)
        self.logger.info('Created InactiveRPCServicer for internal usage', extra=conn_config)

    def run_inf_loop(self):
        raise RuntimeError("Method is not accessible")