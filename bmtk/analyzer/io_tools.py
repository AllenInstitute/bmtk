from six import string_types
from bmtk.simulator.utils.config import ConfigDict


def load_config(config):
    if isinstance(config, string_types):
        return ConfigDict.from_json(config)
    elif isinstance(config, dict):
        return ConfigDict.from_dict(config)
    else:
        raise Exception('Could not convert {} (type "{}") to json.'.format(config, type(config)))