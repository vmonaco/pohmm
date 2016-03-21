| [![Build Status](https://api.travis-ci.org/vmonaco/pohmm.png?branch=master)](https://travis-ci.org/vmonaco/pohmm) | [![Coverage](https://coveralls.io/repos/vmonaco/pohmm/badge.svg?branch=master&service=github)](https://coveralls.io/github/vmonaco/pohmm?branch=master) | [![pip version](https://img.shields.io/pypi/v/pohmm.svg)](https://pypi.python.org/pypi/pohmm) | [![pip downloads](https://img.shields.io/pypi/dm/pohmm.svg)](https://pypi.python.org/pypi/pohmm) |

# pohmm 

``pohmm`` is an implementation of the partially observable hidden Markov model, a generalization of the hidden Markov model in which the underlying system state is partially observable through event metadata at each time step.

An application that motivates usage of such a model is keystroke biometrics where the user can be in either a passive or active hidden state at each time step, and the time between key presses depends on the hidden state. In addition, the hidden state depends on the key that was pressed; thus the keys are observed symbols that partially reveal the hidden state of the user.

## Install

Development version:

    $ pip install git+git://github.com/vmonaco/pohmm.git

## Usage

The dependency graph of the POHMM is shown below.

![POHMM structure](figures/pohmm-structure.png)

## Examples

Keystroke example

In the root source directory,

    $ python examples/keystroke.py

## How to cite

