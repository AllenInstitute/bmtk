import bmtk.simulator.utils.config as msdk_config


# TODO: Implement pointnet validator and create json schema for popnet
def from_json(config_file, validate=False):
    return msdk_config.from_json(config_file)
