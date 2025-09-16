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

from distutils.core import setup

from setuptools import find_packages

setup(
    name="LeggedLab",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        # 'isaacsim',
        "IsaacLab",
        # "rsl-rl-lib @ ./rsl_rl",  # use local rsl_rl
    ],
)
