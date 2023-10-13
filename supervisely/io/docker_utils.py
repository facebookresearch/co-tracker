# coding: utf-8
from __future__ import annotations
from enum import Enum
import json
from typing import Optional
from supervisely.task.progress import Progress


class PullPolicy(Enum):
    def __str__(self):
        return str(self.value)

    ALWAYS = "Always".lower()
    IF_AVAILABLE = "IfAvailable".lower()
    IF_NOT_PRESENT = "IfNotPresent".lower()
    NEVER = "Never".lower()


class PullStatus(Enum):
    START = "Pulling fs layer"
    DOWNLOAD = "Downloading"
    EXTRACT = "Extracting"
    COMPLETE_LOAD = "Download complete"
    COMPLETE_PULL = "Pull complete"
    OTHER = "Other (unknown)"

    def is_equal(self, status: str) -> bool:
        return status == self.value

    @classmethod
    def from_str(cls, status: Optional[str]) -> PullStatus:
        dct = {
            "Pulling fs layer": PullStatus.START,
            "Downloading": PullStatus.DOWNLOAD,
            "Extracting": PullStatus.EXTRACT,
            "Download complete": PullStatus.COMPLETE_LOAD,
            "Pull complete": PullStatus.COMPLETE_PULL,
        }
        return dct.get(status, PullStatus.OTHER)


def docker_pull_if_needed(docker_api, docker_image_name, policy, logger, progress=True):
    logger.info(
        "docker_pull_if_needed args",
        extra={
            "policy": policy,
            "type(policy)": type(policy),
            "policy == PullPolicy.ALWAYS": str(policy) == str(PullPolicy.ALWAYS),
            "policy == PullPolicy.NEVER": str(policy) == str(PullPolicy.NEVER),
            "policy == PullPolicy.IF_NOT_PRESENT": str(policy) == str(PullPolicy.IF_NOT_PRESENT),
            "policy == PullPolicy.IF_AVAILABLE": str(policy) == str(PullPolicy.IF_AVAILABLE),
        },
    )
    if str(policy) == str(PullPolicy.ALWAYS):
        if progress is False:
            _docker_pull(docker_api, docker_image_name, logger)
        else:
            _docker_pull_progress(docker_api, docker_image_name, logger)
    elif str(policy) == str(PullPolicy.NEVER):
        pass
    elif str(policy) == str(PullPolicy.IF_NOT_PRESENT):
        if not _docker_image_exists(docker_api, docker_image_name):
            if progress is False:
                _docker_pull(docker_api, docker_image_name, logger)
            else:
                _docker_pull_progress(docker_api, docker_image_name, logger)
    elif str(policy) == str(PullPolicy.IF_AVAILABLE):
        if progress is False:
            _docker_pull(docker_api, docker_image_name, logger, raise_exception=True)
        else:
            _docker_pull_progress(docker_api, docker_image_name, logger, raise_exception=True)
    else:
        raise RuntimeError(f"Unknown pull policy {str(policy)}")
    if not _docker_image_exists(docker_api, docker_image_name):
        raise RuntimeError(
            f"Docker image {docker_image_name} not found. Agent's PULL_POLICY is {str(policy)}"
        )


def _docker_pull(docker_api, docker_image_name, logger, raise_exception=True):
    from docker.errors import DockerException

    logger.info("Docker image will be pulled", extra={"image_name": docker_image_name})
    progress_dummy = Progress("Pulling image...", 1, ext_logger=logger)
    progress_dummy.iter_done_report()
    try:
        pulled_img = docker_api.images.pull(docker_image_name)
        logger.info(
            "Docker image has been pulled",
            extra={"pulled": {"tags": pulled_img.tags, "id": pulled_img.id}},
        )
    except DockerException as e:
        if raise_exception is True:
            raise DockerException(
                "Unable to pull image: see actual error above. "
                "Please, run the task again or contact support team."
            )
        else:
            logger.warn("Pulling step is skipped. Unable to pull image: {!r}.".format(str(e)))


def _docker_pull_progress(docker_api, docker_image_name, logger, raise_exception=True):
    logger.info("Docker image will be pulled", extra={"image_name": docker_image_name})
    from docker.errors import DockerException

    try:
        layers_total_load = {}
        layers_current_load = {}
        layers_total_extract = {}
        layers_current_extract = {}
        started = set()
        loaded = set()
        pulled = set()

        progress_full = Progress("Preparing dockerimage", 1, ext_logger=logger)
        progres_ext = Progress("Extracting layers", 1, is_size=True, ext_logger=logger)
        progress_load = Progress("Downloading layers", 1, is_size=True, ext_logger=logger)

        for line in docker_api.api.pull(docker_image_name, stream=True, decode=True):
            status = PullStatus.from_str(line.get("status", None))
            layer_id = line.get("id", None)
            progress_details = line.get("progressDetail", {})
            need_report = True

            if status is PullStatus.START:
                started.add(layer_id)
                need_report = False
            elif status is PullStatus.DOWNLOAD:
                layers_total_load[layer_id] = progress_details["total"]
                layers_current_load[layer_id] = progress_details["current"]
                total_load = sum(layers_total_load.values())
                current_load = sum(layers_current_load.values())
                if total_load > progress_load.total:
                    progress_load.set(current_load, total_load)
                elif (current_load - progress_load.current) / total_load > 0.01:
                    progress_load.set(current_load, total_load)
                else:
                    need_report = False
            elif status is PullStatus.COMPLETE_LOAD:
                loaded.add(layer_id)
            elif status is PullStatus.EXTRACT:
                layers_total_extract[layer_id] = progress_details["total"]
                layers_current_extract[layer_id] = progress_details["current"]
                total_ext = sum(layers_total_extract.values())
                current_ext = sum(layers_current_extract.values())
                if total_ext > progres_ext.total:
                    progres_ext.set(current_ext, total_ext)
                elif (current_ext - progres_ext.current) / total_ext > 0.01:
                    progres_ext.set(current_ext, total_ext)
                else:
                    need_report = False
            elif status is PullStatus.COMPLETE_PULL:
                pulled.add(layer_id)
            
            if (started != pulled):
                if need_report:
                    if started == loaded:
                        progres_ext.report_progress()
                    else:
                        progress_load.report_progress()
            elif len(pulled) > 0:
                progress_full.report_progress()

        progress_full.iter_done()
        progress_full.report_progress()
        logger.info("Docker image has been pulled", extra={"image_name": docker_image_name})
    except DockerException as e:
        if raise_exception is True:
            raise e
            # raise DockerException(
            #     "Unable to pull image: see actual error above. "
            #     "Please, run the task again or contact support team."
            # )
        else:
            logger.warn("Pulling step is skipped. Unable to pull image: {!r}.".format(repr(e)))


def _docker_image_exists(docker_api, docker_image_name):
    from docker.errors import ImageNotFound

    try:
        docker_img = docker_api.images.get(docker_image_name)
    except ImageNotFound:
        return False
    return True
