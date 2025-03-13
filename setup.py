from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_lab',
    packages=find_packages(),
    install_requires=[
        # 'isaacsim',
        'IsaacLab',]
)
