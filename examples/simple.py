#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POHMM consistency example.
"""

from __future__ import print_function

import numpy as np
import pohmm

TRAIN_OBS = np.array([4, 3, 4, 2, 3, 1])[:, np.newaxis]
TRAIN_PSTATES = list('babaca')

TEST_OBS = np.array([2, 1, 6])[:, np.newaxis]
TEST_PSTATES = list('cad')


def main():
    print('Train data ...')
    print('Observations:', TRAIN_OBS)
    print('p-states:', TRAIN_PSTATES)

    hmm = pohmm.Pohmm(n_hidden_states=2,
                      init_spread=2,
                      emissions=['lognormal'],
                      init_method='obs',
                      smoothing='freq',
                      random_state=1234)

    print('Fitting model ...')
    hmm.fit([TRAIN_OBS], [TRAIN_PSTATES])

    print('Fitted model ...')
    print(hmm)

    print('Test data ...')
    print('Observations:\n', TEST_OBS)
    print('p-states:', TEST_PSTATES)

    print('Test loglikelihood ...')
    print(hmm.score(TEST_OBS, TEST_PSTATES))

    sample_obs, sample_pstates, sample_hstates = hmm.sample(n_obs=10)
    print('Generated data ...')
    print('Observations:\n', sample_obs)
    print('p-states:', sample_pstates)
    print('h-states:', sample_hstates)

    return


if __name__ == '__main__':
    main()
