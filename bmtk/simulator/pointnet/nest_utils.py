import nest
import re


def get_version():
    """Trys to get NEST version major, minor, and patch (optional) of the current running version of nest. Will return
    as a list of ints [major, minor, patch], although patch may be None.

    :return: [major, minor, patch] if able to parse version, None if fails. 'patch' may be None value.
    """

    # Try to get the version string
    version_str = None
    try:
        # NEST 2.* uses .version() to get version string
        version_str = nest.version()
    except AttributeError:
        pass

    if version_str is None:
        try:
            # For NEST 3.1 it uses __version__ attribute to store string
            version_str = nest.__version__
            if version_str.upper() == 'UNKNOWN':
                return [3, None, None]

        except AttributeError:
            pass

    if version_str is None:
        return None

    # parse the version string to get major, minor and patch numbers
    try:
        version_pattern = re.compile(r'.*nest-(\d+)\.(\d+)(?:\.(\d+))?.*')
        m = re.match(version_pattern, version_str)
        n_groups = len(m.groups())
        ver_major = int(m.group(1))
        ver_minor = int(m.group(2))
        ver_patch = int(m.group(3)) if n_groups >= 3 and m.group(3) is not None else None
        return [ver_major, ver_minor, ver_patch]

    except (AttributeError, IndexError, ValueError, TypeError) as err:
        return None


nest_version = get_version()
if nest_version is None:
    # For now default to assume NEST 3.*.* is being used
    nest_version = [3, None, None]


NEST_SYNAPSE_MODEL_PROP = 'model' if nest_version[0] == 2 else 'synapse_model'
NEST_SPIKE_DETECTOR = 'spike_detector' if nest_version[0] == 2 else 'spike_recorder'
