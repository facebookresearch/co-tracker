import functools
import time
from supervisely.sly_logger import logger


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        logger.debug(f"TIME {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def update_fields(func):
    """Update state field after executing function"""
    @functools.wraps(func)
    def wrapper_updater(*args, **kwargs):
        kwargs['fields_to_update'] = {}
        exception = None

        try:
            value = func(*args, **kwargs)
        except Exception as ex:
            value = None
            exception = ex

        user_api = kwargs.get('api', None)
        app_task_id = kwargs.get('task_id', None)

        if user_api and app_task_id and len(kwargs['fields_to_update']) > 0:
            user_api.task.set_fields_from_dict(app_task_id, kwargs['fields_to_update'])

        if exception:
            raise exception

        return value

    return wrapper_updater
