#! /usr/bin/env python
#
# Copyright (C) 2016 Vinnie Monaco <contact@vmonaco.com>

import os, sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(compile(open('pohmm/version.py').read(),
                  'pohmm/version.py', 'exec'))

install_requires = [
    'Cython',
    'numpy',
    'scipy',
    'pandas',
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
    license='MIT',
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
        'Programming Language :: Python :: 3.6',
    ],
    ext_modules=cythonize("pohmm/_hmmc.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require
    },
    package_data={
        "pohmm": [
            "../README.md",
            "../README.txt",
            "../LICENSE",
            "../MANIFEST.in",
        ]
    },
)

if __name__ == '__main__':
    setup(**setup_options)
