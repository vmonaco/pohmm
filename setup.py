#! /usr/bin/env python
#
# Copyright (C) 2015 Vinnie Monaco <contact@vmonaco.com>

import os, sys
from setuptools import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(compile(open('pohmm/version.py').read(),
                  'pohmm/version.py', 'exec'))

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
    version=__version__,
    author='Vinnie Monaco',
    author_email='contact@vmonaco.com',
    description='Partially observable hidden Markov model',
    license='new BSD',
    keywords='hidden Markov model data analysis',
    url='https://github.com/vmonaco/pohmm',
    packages=['pohmm'],
    long_description=read('README.txt'),
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
    package_data={
        "pohmm": [
            "../README.rst",
            "../README.txt",
            "../LICENSE",
            "../MANIFEST.in",
        ]
    },
)

if __name__ == '__main__':
    setup(**setup_options)
