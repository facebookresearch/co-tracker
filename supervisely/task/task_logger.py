# coding: utf-8

import queue
import logging
import threading
import concurrent.futures
import time
import json

from supervisely.sly_logger import get_task_logger, add_logger_handler, change_formatters_default_values, \
                                       ServiceType, EventType, add_default_logging_into_file, logger
from supervisely.api.module_api import ApiField

BATCH_SIZE_LOG = 50


class LogQueue:
    def __init__(self):
        self.q = queue.Queue()  # no limit

    def put_nowait(self, log_line):
        self.q.put_nowait(log_line)

    def _get_batch_nowait(self, batch_limit):
        log_lines = []
        for _ in range(batch_limit):
            try:
                log_line = self.q.get_nowait()
            except queue.Empty:
                break
            log_lines.append(log_line)
        return log_lines

    def get_log_batch_nowait(self):
        res = self._get_batch_nowait(BATCH_SIZE_LOG)
        return res

    def get_log_batch_blocking(self):
        first_log_line = self.q.get(block=True)
        rest_lines = self._get_batch_nowait(BATCH_SIZE_LOG)
        log_lines = [first_log_line] + rest_lines
        return log_lines


class SlyApiHandler(logging.Handler):
    def __init__(self, api):
        super().__init__()
        self._api = api
        self._stop_log_event = threading.Event()

        self.log_queue = LogQueue()
        self.executor_log = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future_log = self.executor_log.submit(self.submit_log)

    def emit(self, record):
        log_entry = self.format(record)
        log_entry = json.loads(log_entry)
        self.log_queue.put_nowait(log_entry)

    def submit_log(self):
        break_flag = False
        while True:
            log_lines = self.log_queue.get_log_batch_nowait()
            if len(log_lines) > 0:
                self._api.task.submit_logs(log_lines)
                break_flag = False
            else:
                if break_flag:
                    return True
                if self._stop_log_event.isSet():
                    print("stop here")
                    break_flag = True  # exit after next loop without data
                time.sleep(0.5)

    def stop_log_thread(self):
        self._stop_log_event.set()
        self.executor_log.shutdown(wait=True)
        return self.future_log.result()  # crash if log thread crashed


def init_global_task_logger(task_id, api=None, file_path=None):
    change_formatters_default_values(logger, 'service_type', ServiceType.TASK)
    change_formatters_default_values(logger, 'event_type', EventType.LOGJ)
    change_formatters_default_values(logger, 'task_id', task_id)

    if api is not None:
        handler = SlyApiHandler(api)
        add_logger_handler(logger, handler)
    if file_path is not None:
        #add_default_logging_into_file(logger, self.dir_logs)
        pass


def _stop_and_wait_logger(logger):
    for handler in logger.handlers:
        if type(handler) is SlyApiHandler:
            handler.stop_log_thread()


def log_task_finished(logger):
    if logger is None:
        return
    logger.info('TASK_END', extra={'event_type': EventType.TASK_FINISHED})
    _stop_and_wait_logger(logger)


def log_task_crashed(logger, e=None):
    if logger is None:
        return
    if e is None:
        e = Exception("Crashed without exception info")
    logger.critical('TASK_END', exc_info=True, extra={'event_type': EventType.TASK_CRASHED, 'exc_str': str(e)})
    _stop_and_wait_logger(logger)



