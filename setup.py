#! /usr/bin/env python
#
# Copyright (C) 2015 Vinnie Monaco <contact@vmonaco.com>

import os, sys

try:
    import numpy as np
except ImportError:
    import os.path
    import sys

    # A dirty hack to get RTD running.
    class np:
        @staticmethod
        def get_include():
            return os.path.join(sys.prefix, 'include')

from setuptools import setup, Extension

# Extract the version, README, and CHANGES
here = os.path.abspath(os.path.dirname(__file__))

try:
    import pohmm
    version = pohmm.__version__
except Exception:
    version = ''

try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()
except IOError:
    README = CHANGES = ''


install_requires = [
    'numpy',
    'scipy',
    'Cython'
]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
]

docs_require = [
    'Sphinx',
    'sphinx-gallery',
]


setup_options = dict(
    name='pohmm',
    version=version,
    description='Partially observable hidden Markov model',
    long_description='\n\n'.join([README, CHANGES]),
    maintainer='Vinnie Monaco',
    maintainer_email='contact@vmonaco.com',
    license='new BSD',
    url='https://github.com/vmonaco/pohmm',
    packages=['pohmm'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    ext_modules=[
        Extension('pohmm._hmmc', ['pohmm/_hmmc.pyx'],
                  extra_compile_args=['-O3'],
                  include_dirs=[np.get_include()])
    ],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require
    },
)

if __name__ == '__main__':
    setup(**setup_options)
