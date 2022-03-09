from setuptools import setup, find_packages
import io
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],

    # setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    install_requires=[
        'jsonschema',
        'pandas',
        'numpy',
        'six',
        'h5py',
        'matplotlib',
        'enum; python_version <= "2.7"',
        'scipy',
        'scikit-image',  # Only required for filternet, consider making optional
        'sympy'  # For FilterNet
    ],
    extras_require={
        'bionet': ['NEURON'],
        'mintnet': ['tensorflow'],
        'pointnet': ['NEST'],
        'popnet': ['DiPDE']
    },
    packages=find_packages(exclude=['bmtk.tests', 'bmtk.tests.*']),
    # package_data={'': ['*.md', '*.txt', '*.cfg', '**/*.json', '**/*.hoc']},
    include_package_data=True,
    platforms='any'

)
