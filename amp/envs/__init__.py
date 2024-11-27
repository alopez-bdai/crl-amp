from amp.utils.task_registry import task_registry

from .amp.amp_task import AMPTask
from amp.cfg.amp.go1_config import Go1AMPCfg, Go1AMPTrainCfg

task_registry.register("go1_amp", AMPTask, Go1AMPCfg(), Go1AMPTrainCfg())
