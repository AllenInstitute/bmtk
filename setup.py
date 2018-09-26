from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys
import bmtk

package_name = 'bmtk'
package_version = bmtk.__version__


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def prepend_find_packages(*roots):
    """Recursively traverse nested packages under the root directories"""
    packages = []
    
    for root in roots:
        packages += [root]
        packages += [root + '.' + s for s in find_packages(root)]
        
    return packages


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--ignore=examples', '--ignore=nicholasc', '--cov-report=html']
        self.test_args_cov = self.test_args + ['--cov=%s'.format(package_name), '--cov-report=term']
        self.test_suite = True

    def run_tests(self):
        import pytest
        
        try:
            errcode = pytest.main(self.test_args_cov)
        except:
            errcode = pytest.main(self.test_args)
        sys.exit(errcode)


with open('README.md', 'r') as fhandle:
    long_description = fhandle.read()


setup(
    name=package_name,
    version=package_version,
    description='Brain Modeling Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/AllenInstitute/bmtk',
    author='Kael Dai',
    author_email='kaeld@alleninstitute.org',
    package_data={'': ['*.md', '*.txt', '*.cfg', '**/*.json', '**/*.hoc']},
    tests_require=['pytest'],
    install_requires=[
        'jsonschema',
        'pandas',
        'numpy',
        'six',
        'h5py',
        'matplotlib'
    ],
    extras_require={
        'bionet': ['NEURON'],
        'mintnet': ['tensorflow'],
        'pointnet': ['NEST'],
        'popnet': ['DiPDE']
    },
    cmdclass={'test': PyTest},
    packages=prepend_find_packages(package_name),
    include_package_data=True,
    platforms='any',
    keywords=['neuroscience', 'scientific', 'modeling', 'simulation'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ]
)
