# coding: utf-8

import logging
import types
import datetime
import os
from collections import namedtuple
from enum import Enum
#import simplejson
from pythonjsonlogger import jsonlogger


###############################################################################

class ServiceType(Enum):
    AGENT = 1
    TASK = 2
    EXPORT = 3


class EventType(Enum):
    LOGJ = 1
    LOGS = 2
    TASK_STARTED = 3
    TASK_FINISHED = 4
    TASK_STOPPED = 5
    TASK_CRASHED = 6
    PROGRESS = 7
    CHECKPOINT = 8
    METRICS = 9
    TASK_REMOVED = 10
    MODEL_APPLIED = 11
    DTL_APPLIED = 12
    IMPORT_APPLIED = 13
    PROJECT_CREATED = 14
    TASK_VERIFIED = 15
    STEP_COMPLETE = 16
    TASK_DEPLOYED = 17
    AGENT_READY_FOR_TASKS = 18
    MISSED_TASK_FOUND = 19
    REPORT_CREATED = 20
    APP_FINISHED = 21


###############################################################################
# predefined levels


# level name: level, default exc_info, description
LogLevelSpec = namedtuple('LogLevelSpec', [
    'int',
    'add_exc_info',
    'descr',
])


LOGGING_LEVELS = {
    'FATAL': LogLevelSpec(50, True, 'Critical error'),
    'ERROR': LogLevelSpec(40, True, 'Error'),   # may be shown to end user
    'WARN': LogLevelSpec(30, False, 'Warning'),   # may be shown to end user
    'INFO': LogLevelSpec(20, False, 'Info'),   # may be shown to end user
    'DEBUG': LogLevelSpec(10, False, 'Debug'),
    'TRACE': LogLevelSpec(5, False, 'Trace'),
}


def _set_logging_levels(levels, the_logger):
    for lvl_name, (lvl, def_exc_info, _) in levels.items():
        logging.addLevelName(lvl, lvl_name.upper())  # two mappings

        def construct_logger_member(lvl_val, default_exc_info):
            return lambda self, msg, *args, exc_info=default_exc_info, **kwargs: \
                self.log(lvl_val,
                         msg,
                         *args,
                         exc_info=exc_info,
                         **kwargs)

        func = construct_logger_member(lvl, def_exc_info)
        bound_method = types.MethodType(func, the_logger)
        setattr(the_logger, lvl_name.lower(), bound_method)


###############################################################################


def _get_default_logging_fields():
    supported_keys = [
        'asctime',
        # 'created',
        # 'filename',
        # 'funcName',
        'levelname',
        # 'levelno',
        # 'lineno',
        # 'module',
        # 'msecs',
        'message',
        # 'name',
        # 'pathname',
        # 'process',
        # 'processName',
        # 'relativeCreated',
        # 'thread',
        # 'threadName'
    ]
    return ' '.join(['%({0:s})'.format(k) for k in supported_keys])


#def dumps_ignore_nan(obj, *args, **kwargs):
#    return  simplejson.dumps(obj, ignore_nan=True, *args, **kwargs)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    additional_fields = {}

    def __init__(self, format_string):
        super().__init__(format_string)#, json_serializer=dumps_ignore_nan)

    def process_log_record(self, log_record):
        log_record['timestamp'] = log_record.pop('asctime', None)

        levelname = log_record.pop('levelname', None)
        if levelname is not None:
            log_record['level'] = levelname.lower()

        e_info = log_record.pop('exc_info', None)
        if e_info is not None:
            if e_info == 'NoneType: None':  # python logger is not ok here
                pass
            else:
                log_record['stack'] = e_info.split('\n')

        return jsonlogger.JsonFormatter.process_log_record(self, log_record)

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        for field, val in CustomJsonFormatter.additional_fields.items():
            if (val is not None) and (field not in log_record):
                log_record[field] = val

    def formatTime(self, record, datefmt=None):
        ct = datetime.datetime.fromtimestamp(record.created)
        t = ct.strftime('%Y-%m-%dT%H:%M:%S')
        s = '%s.%03dZ' % (t, record.msecs)
        return s


def _construct_logger(the_logger, loglevel_text):
    for handler in the_logger.handlers:
        the_logger.removeHandler(handler)

    _set_logging_levels(LOGGING_LEVELS, the_logger)

    the_logger.setLevel(loglevel_text.upper())

    log_handler = logging.StreamHandler()
    add_logger_handler(the_logger, log_handler)

    the_logger.propagate = False


###############################################################################


def add_logger_handler(the_logger, log_handler):  # default format
    logger_fmt_string = _get_default_logging_fields()
    formatter = CustomJsonFormatter(logger_fmt_string)
    log_handler.setFormatter(formatter)
    the_logger.addHandler(log_handler)


def add_default_logging_into_file(the_logger, log_dir):
    fname = 'log_{}.txt'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    ofpath = os.path.join(log_dir, fname)

    log_handler_file = logging.FileHandler(filename=ofpath)
    add_logger_handler(the_logger, log_handler_file)


# runs on all formatters
def change_formatters_default_values(the_logger, field_name, value):
    for handler in the_logger.handlers:
        hfaf = handler.formatter.additional_fields
        if value is not None:
            hfaf[field_name] = value
        else:
            hfaf.pop(field_name, None)


def _get_loglevel_env():
    loglevel = os.getenv('LOG_LEVEL', None)
    if loglevel is None:
        loglevel = os.getenv('LOGLEVEL', 'INFO')
    return loglevel.upper()


def set_global_logger():
    loglevel = _get_loglevel_env()  # use the env to set loglevel
    the_logger = logging.getLogger('logger')  # optional logger name
    _construct_logger(the_logger, loglevel)
    return the_logger


def get_task_logger(task_id, loglevel=None):
    if loglevel is None:
        loglevel = _get_loglevel_env() # use the env to set loglevel
    logger_name = 'task_{}'.format(task_id)
    the_logger = logging.getLogger(logger_name)  # optional logger name
    _construct_logger(the_logger, loglevel)
    return the_logger


logger = set_global_logger()

