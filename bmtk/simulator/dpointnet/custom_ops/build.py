import os
import subprocess
import sys
from pathlib import Path


def main():
    script = Path(__file__).with_name('build.sh')
    environment = os.environ.copy()
    environment['PYTHON'] = sys.executable
    subprocess.run(['bash', str(script)], check=True, env=environment)


if __name__ == '__main__':
    main()
