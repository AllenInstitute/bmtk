import os
import json

import bmtk.simulator.utils.config as msdk_config
from bmtk.simulator.utils.sim_validator import SimConfigValidator


# TODO: Implement pointnet validator and create json schema for pointnet
def from_json(config_file, validate=False):
    return msdk_config.from_json(config_file)