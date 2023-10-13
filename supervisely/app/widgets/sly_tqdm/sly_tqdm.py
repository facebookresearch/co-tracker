import asyncio
import copy
import math
import re
import sys
import weakref
from functools import partial
from typing import Any, Union

from tqdm import tqdm

from supervisely import is_development, is_production

# from supervisely import _original_tqdm as tqdm
from supervisely.app import DataJson
from supervisely.app.fastapi import run_sync
from supervisely.app.singleton import Singleton
from supervisely.app.widgets import Widget
from supervisely.sly_logger import EventType, logger


def extract_by_regexp(regexp, string):
    result = re.search(regexp, string)
    if result is not None:
        return result.group(0)
    else:
        return ""


UNITS = ["B", "KB", "MB", "GB", "TB"]


class _slyProgressBarIO:
    def __init__(self, widget_id, message=None, total=None, unit=None, unit_scale=None):
        self.widget_id = widget_id

        self.progress = {
            "percent": 0,
            "info": "",
            "message": message,
            "status": None,
            "n": -1,
        }

        self.prev_state = self.progress.copy()
        self.total = total

        self.unit_scale = unit_scale
        self.unit = unit
        if self.unit_scale and self.unit == "B":
            self.unit = self.get_unit()

        self._n = 0

        self._done = False

    def print_progress_to_supervisely_tasks_section(self):
        """
        Logs a message with level INFO on logger. Message contain type of progress, subtask message, currtnt and total number of iterations
        """

        if self._n == self.progress["n"]:
            return

        self.progress["n"] = self._n

        extra = {
            "event_type": EventType.PROGRESS,
            "subtask": self.progress.get("message", None),
            "current": self._n,
            "total": self.total,
        }

        if self.unit_scale and self.unit != "it":
            extra["current_label"] = f"{self.bytes_to_unit(self._n)} {self.unit}"
            extra["total_label"] = f"{self.bytes_to_unit(self.total)} {self.unit}"

        gettrace = getattr(sys, "gettrace", None)
        in_debug_mode = gettrace is not None and gettrace()

        if not in_debug_mode:
            logger.info("progress", extra=extra)

    def write(self, s):
        new_text = s.strip().replace("\r", "")
        if len(new_text) != 0:
            if self.total is not None:
                if self.total == 0:  # to avoid ZeroDivisionError
                    self.progress["percent"] = 100
                else:
                    self.progress["percent"] = int(self._n / self.total * 100)
                self.progress["info"] = extract_by_regexp(r"(\d+(?:\.\d+\w+)?)*\w*/.*\]", new_text)
            else:
                self.progress["percent"] = int(self._n)
                self.progress["info"] = extract_by_regexp(r"(\d+(?:\.\d+\w+)?)*.*\]", new_text)

    def flush(self, synchronize_changes=True):
        if self.prev_state != self.progress:
            if self.progress["percent"] != "" and self.progress["info"] != "":
                self.print_progress_to_supervisely_tasks_section()

                if self.progress["percent"] == 100 and self.total is not None:
                    self.progress["status"] = "success"

                for key, value in self.progress.items():
                    DataJson()[f"{self.widget_id}"][key] = value

                if synchronize_changes is True:
                    run_sync(DataJson().synchronize_changes())

                self.prev_state = copy.deepcopy(self.progress)

    def __del__(self):
        self.progress["status"] = "success"
        self.progress["percent"] = 100

        self.flush(synchronize_changes=False)
        self.print_progress_to_supervisely_tasks_section()

    def get_unit(self) -> str:
        """Returns the unit of the progress bar according to the total number of bytes.

        :return: unit of the progress bar
        :rtype: str
        """
        total = self.total
        for unit in UNITS:
            if total < 1000:
                return unit
            total /= 1000

    def bytes_to_unit(self, bytes: int) -> Union[float, Any]:
        """Returns a human-readable string representation of bytes according to the
        self.unit.

        :param bytes: number of bytes
        :type bytes: int
        :return: converted size from bytes to self.unit
        :rtype: Union[float, Any]
        """
        if not isinstance(bytes, int) or self.unit not in UNITS:
            return bytes

        for idx, unit in enumerate(UNITS):
            if self.unit == unit:
                return round(bytes / (1000**idx), 2)


class CustomTqdm(tqdm):
    def __init__(self, widget_id, message, *args, **kwargs):
        extracted_total = copy.copy(
            tqdm(iterable=kwargs["iterable"], total=kwargs["total"], disable=True).total
        )
        unit_scale = kwargs.get("unit_scale")
        unit = kwargs.get("unit")

        self._iteration_value = 0
        self._iteration_number = 0
        self._iteration_locked = False
        self._total_monitor_size = 0

        self.unit_divisor = 1024

        # self.n = 0
        # self.reported_cnt = 0
        # self.report_every = max(1, math.ceil(self.extracted_total / 100))

        super().__init__(
            file=_slyProgressBarIO(widget_id, message, extracted_total, unit, unit_scale),
            *args,
            **kwargs,
        )

    def refresh(self, *args, **kwargs):
        if self.fp is not None:
            self.fp._n = self.n
        super().refresh(*args, **kwargs)

    def close(self):
        self.refresh()
        super(CustomTqdm, self).close()
        if self.fp is not None:
            self.fp.__del__()

    def __del__(self):
        super(CustomTqdm, self).__del__()
        if self.fp is not None:
            self.fp.__del__()

    def _progress_monitor(self, monitor):
        if is_development() and self.n >= self.total:
            self.refresh()
            self.close()

        if monitor.bytes_read == 8192:
            self._total_monitor_size += monitor.len

        if self._total_monitor_size > self.total:
            self.total = self._total_monitor_size

        if not self._iteration_locked:
            # if is_development():
            super().update(self._iteration_value + monitor.bytes_read - self.n)
            # else:
            #     self.set_current_value(self._iteration_value + monitor.bytes_read, report=False)

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


class SlyTqdm(Widget):
    # @TODO: track all active sessions for one object and close them if new object inited
    def __init__(self, message: str = None, show_percents: bool = False, widget_id: str = None):
        """
        Wrapper for classic tqdm progress bar.

            Parameters
            ----------
            identifier  : int, required
                HTML element identifier
            message  : int, optional
                Text message which displayed in HTML


            desc, total, leave, ncols, ... :
                Like in tqdm

        """
        self.message = message
        self.show_percents = show_percents

        self._active_session = None
        self._sly_io = None
        self._hide_on_finish = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _close_active_session(self):
        if self._active_session is not None:
            try:
                self._active_session.__del__()
                self._active_session = None
            except ReferenceError:
                pass

    def __call__(
        self,
        iterable=None,
        message=None,
        desc=None,
        total=None,
        leave=None,
        ncols=None,
        mininterval=1.0,
        maxinterval=10.0,
        miniters=None,
        ascii=False,
        disable=False,
        unit="it",
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        gui=False,
        **kwargs,
    ):
        return CustomTqdm(
            widget_id=self.widget_id,
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            message=message,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            gui=gui,
            **kwargs,
        )

    def get_json_data(self):
        return {
            "percent": 0,
            "info": None,
            "message": self.message,
            "status": None,
            "show_percents": self.show_percents,
        }

    def get_json_state(self):
        return None

    def set_message(self, message):
        self.message = message
        DataJson()[self.widget_id]["message"] = message
        DataJson().send_changes()


class Progress(SlyTqdm):
    def __init__(
        self,
        message: str = None,
        show_percents: bool = False,
        hide_on_finish=True,
        widget_id: str = None,
    ):
        self.hide_on_finish = hide_on_finish
        super().__init__(message=message, show_percents=show_percents, widget_id=widget_id)

    pass
