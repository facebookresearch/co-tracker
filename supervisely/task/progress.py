# coding: utf-8
from __future__ import annotations

import inspect
import math
from functools import partial
from typing import Optional

from tqdm import tqdm

from supervisely._utils import is_development, is_production, sizeof_fmt
from supervisely.sly_logger import EventType, logger


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    return epoch + train_it / float(train_its)


class Progress:
    """
    Modules operations monitoring and displaying statistics of data processing. :class:`Progress<Progress>` object is immutable.

    :param message: Progress message e.g. "Images uploaded:", "Processing:".
    :type message: str
    :param total_cnt: Total count.
    :type total_cnt: int
    :param ext_logger: Logger object.
    :type ext_logger: logger, optional
    :param is_size: Shows Label size.
    :type is_size: bool, optional
    :param need_info_log: Shows info log.
    :type need_info_log: bool, optional
    :param min_report_percent: Minimum report percent of total items in progress to log.
    :type min_report_percent: int, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.sly_logger import logger

        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        progress = sly.Progress("Images downloaded: ", len(img_infos), ext_logger=logger, is_size=True, need_info_log=True)
        api.image.download_paths(ds_id, image_ids, save_paths, progress_cb=progress.iters_done_report)

        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 0,
        #  "total": 6, "current_label": "0.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:45.659Z", "level": "info"}
        # {"message": "Images downloaded:  [0.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:45.660Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 1,
        #  "total": 6, "current_label": "1.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.134Z", "level": "info"}
        # {"message": "Images downloaded:  [1.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.134Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 2,
        #  "total": 6, "current_label": "2.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [2.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 3,
        #  "total": 6, "current_label": "3.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [3.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 4,
        #  "total": 6, "current_label": "4.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "Images downloaded:  [4.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.135Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 5,
        #  "total": 6, "current_label": "5.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "Images downloaded:  [5.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Images downloaded: ", "current": 6,
        #  "total": 6, "current_label": "6.0 B", "total_label": "6.0 B", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
        # {"message": "Images downloaded:  [6.0 B / 6.0 B]", "timestamp": "2021-03-17T13:57:46.136Z", "level": "info"}
    """

    def __init__(
        self,
        message: str,
        total_cnt: int,
        ext_logger: Optional[logger] = None,
        is_size: Optional[bool] = False,
        need_info_log: Optional[bool] = False,
        min_report_percent: Optional[int] = 1,
    ):
        self.is_size = is_size
        self.message = message
        self.total = total_cnt
        self.current = 0
        self.is_total_unknown = total_cnt == 0

        self.total_label = ""
        self.current_label = ""
        self._refresh_labels()

        self.reported_cnt = 0
        self.logger = logger if ext_logger is None else ext_logger
        self.report_every = max(1, math.ceil(total_cnt / 100 * min_report_percent))
        self.need_info_log = need_info_log

        mb5 = 5 * 1024 * 1024
        if self.is_size and self.is_total_unknown:
            self.report_every = mb5  # 5mb

        mb1 = 1 * 1024 * 1024
        if self.is_size and self.is_total_unknown is False and self.report_every < mb1:
            self.report_every = mb1  # 1mb

        if (
            self.is_size
            and self.is_total_unknown is False
            and self.total > 40 * 1024 * 1024
            and self.report_every < mb5
        ):
            self.report_every = mb5

        self.report_progress()

    def _refresh_labels(self):
        if self.is_size:
            self.total_label = (
                sizeof_fmt(self.total) if self.total > 0 else sizeof_fmt(self.current)
            )
            self.current_label = sizeof_fmt(self.current)
        else:
            self.total_label = str(self.total if self.total > 0 else self.current)
            self.current_label = str(self.current)

    def iter_done(self) -> None:
        """
        Increments the current iteration counter by 1
        """
        self.current += 1
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def iters_done(self, count: int) -> None:
        """
        Increments the current iteration counter by given count

        :param count: Amount of iters
        :type count: int
        """
        self.current += count
        if self.is_total_unknown:
            self.total = self.current
        self._refresh_labels()

    def report_progress(self) -> None:
        """
        Logs a message with level INFO in logger. Message contain type of progress, subtask message, current and total number of iterations

        :return: None
        :rtype: :class:`NoneType`
        """
        self.print_progress()
        self.reported_cnt += 1

    def print_progress(self) -> None:
        """
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        """
        extra = {
            "event_type": EventType.PROGRESS,
            "subtask": self.message,
            "current": math.ceil(self.current),
            "total": math.ceil(self.total) if self.total > 0 else math.ceil(self.current),
        }

        if self.is_size:
            extra["current_label"] = self.current_label
            extra["total_label"] = self.total_label

        self.logger.info("progress", extra=extra)
        if self.need_info_log is True:
            self.logger.info(f"{self.message} [{self.current_label} / {self.total_label}]")

    def need_report(self) -> bool:
        if (
            (self.current >= self.total)
            or (self.current % self.report_every == 0)
            or ((self.reported_cnt - 1) < (self.current // self.report_every))
        ):
            return True
        return False

    def report_if_needed(self) -> None:
        """
        Determines whether the message should be logged depending on current number of iterations
        """
        if self.need_report():
            self.report_progress()

    def iter_done_report(self) -> None:  # finish & report
        """
        Increments the current iteration counter by 1 and logs a message depending on current number of iterations.

        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            progress = sly.Progress("Processing:", len(img_infos))
            for img_info in img_infos:
                img_names.append(img_info.name)
                progress.iter_done_report()

            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 0, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 1, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 2, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 3, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 4, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 5, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 6, "total": 6, "timestamp": "2021-03-17T14:29:33.207Z", "level": "info"}
        """
        self.iter_done()
        self.report_if_needed()

    def iters_done_report(self, count: int) -> None:  # finish & report
        """
        Increments the current iteration counter by given count and logs a message depending on current number of iterations.

        :param count: Counter.
        :type count: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            progress = sly.Progress("Processing:", len(img_infos))
            for img_info in img_infos:
                img_names.append(img_info.name)
                progress.iters_done_report(1)

            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 0, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 1, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 2, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 3, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 4, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 5, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
            # {"message": "progress", "event_type": "EventType.PROGRESS", "subtask": "Processing:",
            #  "current": 6, "total": 6, "timestamp": "2021-03-17T14:31:21.655Z", "level": "info"}
        """
        self.iters_done(count)
        self.report_if_needed()

    def set_current_value(self, value: int, report: Optional[bool] = True) -> None:
        """
        Increments the current iteration counter by this value minus the current value of the counter and logs a message depending on current number of iterations.

        :param value: Current value.
        :type value: int
        :param report: Defines whether to report to log or not.
        :type report: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        if report is True:
            self.iters_done_report(value - self.current)
        else:
            self.iters_done(value - self.current)

    def set(self, current: int, total: int, report: Optional[bool] = True) -> None:
        """
        Sets counter current value and total value and logs a message depending on current number of iterations.

        :param current: Current count.
        :type current: int
        :param total: Total count.
        :type total: int
        :param report: Defines whether to report to log or not.
        :type report: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        self.total = total
        if self.total != 0:
            self.is_total_unknown = False
        self.current = current
        self.reported_cnt = 0
        self.report_every = max(1, math.ceil(total / 100))
        self._refresh_labels()
        if report is True:
            self.report_if_needed()


def report_agent_rpc_ready() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("Ready to get events", extra={"event_type": EventType.TASK_DEPLOYED})


def report_import_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("import finished", extra={"event_type": EventType.IMPORT_APPLIED})


def report_inference_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("model applied", extra={"event_type": EventType.MODEL_APPLIED})


def report_dtl_finished() -> None:
    """
    Logs a message with level INFO on logger
    """
    logger.info("DTL finished", extra={"event_type": EventType.DTL_APPLIED})


def report_dtl_verification_finished(output: str) -> None:
    """
    Logs a message with level INFO on logger
    :param output: str
    """
    logger.info(
        "Verification finished.", extra={"output": output, "event_type": EventType.TASK_VERIFIED}
    )


def _report_metrics(m_type, epoch, metrics):
    logger.info(
        "metrics",
        extra={"event_type": EventType.METRICS, "type": m_type, "epoch": epoch, "metrics": metrics},
    )


def report_metrics_training(epoch, metrics):
    _report_metrics("train", epoch, metrics)


def report_metrics_validation(epoch, metrics):
    _report_metrics("val", epoch, metrics)


def report_checkpoint_saved(checkpoint_idx, subdir, sizeb, best_now, optional_data) -> None:
    logger.info(
        "checkpoint",
        extra={
            "event_type": EventType.CHECKPOINT,
            "id": checkpoint_idx,
            "subdir": subdir,
            "sizeb": sizeb,
            "best_now": best_now,
            "optional": optional_data,
        },
    )


class tqdm_sly(tqdm, Progress):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # for self._upload_monitor
        self._iteration_value = 0
        self._iteration_number = 0
        self._iteration_locked = False
        self._total_monitor_size = 0

        self.unit_divisor = 1024

        relevant_args = {
            "total": "total_cnt",
            "desc": "message",
            "unit": "is_size",
            "unit_scale": "is_size",
        }

        for _tqdm, _progress in relevant_args.items():
            if kwargs.get(_tqdm) is not None and kwargs.get(_progress) is not None:
                raise ValueError(
                    f"Ambiguity error: Please specify only one of arguments: '{_tqdm}' or '{_progress}'."
                )

        kwargs_tqdm = kwargs.copy()

        # pop and convert every possible (and relevant) kwarg from Progress
        if len(args) < 2:  # i.e. 'desc' not set as a positional argument
            if kwargs_tqdm.get("message") is not None:
                kwargs_tqdm.setdefault("desc", kwargs_tqdm["message"])
                kwargs_tqdm.pop("message")
            else:
                kwargs_tqdm.setdefault("desc", "Processing")
        if len(args) < 3:  # i.e. 'total' not set as a positional argument
            if kwargs_tqdm.get("total_cnt") is not None:
                kwargs_tqdm.setdefault("total", kwargs_tqdm["total_cnt"])
                kwargs_tqdm.pop("total_cnt")
        if len(args) < 12:  # i.e. 'unit' not set as a positional argument
            if kwargs_tqdm.pop("is_size", None) == True:
                kwargs_tqdm["unit"] = "B"
                kwargs_tqdm["unit_scale"] = True

        if is_development():
            tqdm.__init__(
                self,
                *args,
                **kwargs_tqdm,
            )
            self.offset = 0  # to prevent overfilling of tqdm in console
        else:
            # disable tqdm on prod but keep attributes
            kwargs_tqdm.setdefault("disable", True)
            tqdm.__init__(
                self,
                *args,
                **kwargs_tqdm,
            )
            # pop and convert every possible (and relevant) kwarg from tqdm
            # mention that tqdm is a prior parent class
            if len(args) < 2:  # i.e. 'desc' not set as a positional argument
                if kwargs.get("desc") is not None:
                    kwargs.setdefault("message", kwargs["desc"])
                    kwargs.pop("desc")
                else:
                    kwargs.setdefault("message", "Processing")
            else:
                kwargs.setdefault("message", args[1])  # args[1]==desc
            if len(args) < 3:  # i.e. 'total' not set as a positional argument
                if kwargs.get("total") is not None:
                    kwargs.setdefault("total_cnt", kwargs["total"])
                    kwargs.pop("total")
            else:
                kwargs.setdefault("total_cnt", args[2])  # args[2]==total
            if len(args) < 12:  # i.e. 'unit' not set as a positional argument
                if kwargs.get("unit") in [
                    "",
                    "B",
                    "k",
                    "M",
                    "G",
                    "T",
                    "P",
                    "E",
                    "Z",
                ] and kwargs.pop("unit_scale", None):
                    kwargs["is_size"] = True
                    kwargs.pop("unit")
            else:
                if (
                    args[11] in ["", "B", "k", "M", "G", "T", "P", "E", "Z"] and args[12] == True
                ):  # i.e. unit=="B" and unit_scale==True
                    kwargs["is_size"] = True

            tqdm_init_params = inspect.signature(tqdm.__init__).parameters.keys()
            for keyword in tqdm_init_params:
                if keyword in kwargs:
                    kwargs.pop(keyword)

            Progress.__init__(
                self,
                **kwargs,
            )
            # self.disable = True/
            self.n = 0

    def update(self, count):
        if is_development():
            tqdm.update(
                self,
                min(count, self.total - self.offset),
            )
            self.offset += count

            if self.n == self.total:
                self.close()
        else:
            Progress.iters_done_report(
                self,
                count,
            )
            self.n += count

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        return self.update(
            *args,
            **kwargs,
        )

    def _progress_monitor(self, monitor):
        if is_development() and self.n >= self.total:
            self.refresh()
            self.close()

        if monitor.bytes_read == 8192:
            self._total_monitor_size += monitor.len

        if self._total_monitor_size > self.total:
            self.total = self._total_monitor_size

        if not self._iteration_locked:
            if is_development():
                super().update(self._iteration_value + monitor.bytes_read - self.n)
            else:
                self.set_current_value(self._iteration_value + monitor.bytes_read, report=False)

        if is_production() and self.need_report():
            self.report_progress()

        if monitor.bytes_read == monitor.len and not self._iteration_locked:
            self._iteration_value += monitor.len
            self._iteration_number += 1
            self._iteration_locked = True
            if is_development():
                self.refresh()

        if monitor.bytes_read < monitor.len:
            self._iteration_locked = False

    def get_partial(self):
        return partial(self._progress_monitor)
