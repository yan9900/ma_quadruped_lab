
from legged_lab.envs import BaseEnvConfig, BaseAgentConfig
from legged_lab.assets.unitree import H1_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


class H1FlatEnvCfg(BaseEnvConfig):
    class scene(BaseEnvConfig.scene):
        class height_scanner(BaseEnvConfig.scene.height_scanner):
            prim_body_name = "torso_link"

        robot = H1_CFG
        terrain_type = "generator"
        terrain_generator = GRAVEL_TERRAINS_CFG

    class robot(BaseEnvConfig.robot):
        penalize_contacts_body_names = [".*knee.*"]
        terminate_contacts_body_names = [".*torso.*"]

    class domain_rand(BaseEnvConfig.domain_rand):

        class add_rigid_body_mass:
            enable = True
            params = {"body_names": [".*torso.*"],
                      "mass_distribution_params": (-5.0, 5.0),
                      "operation": "add"}


class H1FlatAgentCfg(BaseAgentConfig):
    experiment_name = "h1_falt"
    wandb_project = "h1_flat"
