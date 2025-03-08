from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseEnvCfg, BaseAgentCfg
from legged_lab.envs.h1.h1_config import H1FlatEnvCfg, H1FlatAgentCfg
from legged_lab.envs.g1.g1_config import G1FlatEnvCfg, G1FlatAgentCfg
from legged_lab.utils.task_registry import task_registry


task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())