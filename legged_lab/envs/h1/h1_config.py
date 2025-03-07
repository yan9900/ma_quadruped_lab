
from legged_lab.envs import BaseEnvConfig, BaseAgentConfig
from legged_lab.assets.unitree import H1_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


class H1FlatEnvCfg(BaseEnvConfig):
    class scene(BaseEnvConfig.scene):
        robot = H1_CFG
        terrain_type = "generator"
        terrain_generator = GRAVEL_TERRAINS_CFG


class H1FlatAgentCfg(BaseAgentConfig):
    experiment_name = "h1_falt"
    wandb_project = "h1_flat"
