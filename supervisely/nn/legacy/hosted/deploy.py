# coding: utf-8
from copy import deepcopy

from supervisely import logger
from supervisely.function_wrapper import function_wrapper
from supervisely.io.json import load_json_file
from supervisely.nn.config import update_recursively
from supervisely.nn.hosted.constants import CONFIG, MODEL
from supervisely.task.paths import TaskPaths

from supervisely.worker_api.rpc_servicer import AgentRPCServicer
from supervisely.worker_api.agent_rpc import SimpleCache


class ModelDeploy:

    config = {
        'cache_limit': 500,
        'connection': {
            'server_address': None,
            'token': None,
            'task_id': None,
        },
        MODEL: {
        }
    }

    def __init__(self, model_applier_cls, rpc_servicer_cls=AgentRPCServicer):
        self.model_applier_cls = model_applier_cls
        self.rpc_servicer_cls = rpc_servicer_cls
        self.load_config()
        self._create_serv_instance()

    def load_config(self):
        self.config = deepcopy(ModelDeploy.config)
        new_config = load_json_file(TaskPaths.TASK_CONFIG_PATH)
        logger.info('Input config', extra={CONFIG: new_config})
        update_recursively(self.config, new_config)
        logger.info('Full config', extra={CONFIG: self.config})

    def _create_applier(self):
        model_applier = function_wrapper(self.model_applier_cls)
        return model_applier

    def _create_serv_instance(self):
        model_applier = self._create_applier()
        image_cache = SimpleCache(self.config['cache_limit'])
        self.serv_instance = self.rpc_servicer_cls(logger=logger,
                                                   model_applier=model_applier,
                                                   conn_config=self.config['connection'],
                                                   cache=image_cache)

    def run(self):
        self.serv_instance.run_inf_loop()
