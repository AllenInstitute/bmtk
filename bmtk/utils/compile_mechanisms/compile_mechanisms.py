import os
import logging
from subprocess import call
from distutils.dir_util import copy_tree


logger = logging.getLogger(__name__)


def compile_mechanisms(mechanisms_dir):
    logger.info('Attempting to compile NEURON mechanims under "{}"'.format(mechanisms_dir))
    cwd = os.getcwd()
    try:
        os.chdir(os.path.join(mechanisms_dir))
        call(['nrnivmodl', 'modfiles'])
        logger.info('  Success.')
    except Exception as e:
        logger.error('  Was unable to compile mechanism in {}'.format(mechanisms_dir))
    os.chdir(cwd)


def copy_modfiles(mechanisms_dir, cached_dir=None):
    if not os.path.exists(mechanisms_dir):
        logger.info('Creating mechanisms directory {}'.format(mechanisms_dir))

    if cached_dir is None:
        local_path = os.path.dirname(os.path.realpath(__file__))
        cached_dir = os.path.join(local_path, '..', 'scripts/bionet/mechanisms')

    logger.info('Copying mod files from {} to {}'.format(cached_dir, mechanisms_dir))
    copy_tree(cached_dir, mechanisms_dir)
