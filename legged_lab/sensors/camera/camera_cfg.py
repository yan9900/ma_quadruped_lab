# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""Configuration for the Camera"""


from isaaclab.sensors.camera import CameraCfg as BaseCameraCfg
from isaaclab.utils import configclass

from .camera import Camera


@configclass
class CameraCfg(BaseCameraCfg):

    class_type: type = Camera