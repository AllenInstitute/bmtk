import logging
from optparse import OptionParser

from .compile_mechanisms import compile_mechanisms, copy_modfiles


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(module)s [%(levelname)s] %(message)s')
    logging.basicConfig(level=logging.INFO, format='compile_mechanisms [%(levelname)s] %(message)s')

    parser = OptionParser(usage="Usage: python %prog [--copy-modfiles] mechanisms-dir")
    parser.add_option('-c', '--copy-modfiles', dest='copy_modfiles', action='store_true', default=False,
                      help='Also copy over Allen Cell Type modfiles/ directory into mechanism-dir.')

    options, args = parser.parse_args()

    mechanisms_dir = args[0]
    if options.copy_modfiles:
        copy_modfiles(mechanisms_dir=mechanisms_dir)

    compile_mechanisms(mechanisms_dir=mechanisms_dir)
