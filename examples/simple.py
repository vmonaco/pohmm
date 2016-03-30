#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POHMM consistency example.
"""

from __future__ import print_function

import numpy as np
import pohmm

SEED = 1234

TRAIN_OBS = [4, 3, 4, 2, 3, 1]
TRAIN_PSTATES = list('babaca')

TEST_OBS = [2, 1, 6]
TEST_PSTATES = list('cad')


def main():
    np.set_printoptions(precision=3)
    print('Train data ...')
    print('Observations:', TRAIN_OBS)
    print('p-states:', TRAIN_PSTATES)

    model = pohmm.Pohmm(n_hidden_states=2,
                        init_spread=2,
                        emissions=['lognormal'],
                        init_method='obs',
                        smoothing='freq',
                        random_state=SEED)

    print('Fitting model ...')
    model.fit([np.c_[TRAIN_OBS]], [TRAIN_PSTATES])

    print('Fitted model ...')
    print(model)

    print('Test data ...')
    print('Observations:\n', TEST_OBS)
    print('p-states:', TEST_PSTATES)

    print('Test loglikelihood ...')
    print(model.score(np.c_[TEST_OBS], TEST_PSTATES))

    sample_obs, sample_pstates, sample_hstates = model.sample(n_obs=10, random_state=SEED)
    print('Generated data ...')
    print('Observations:\n', sample_obs)
    print('p-states:', sample_pstates)
    print('h-states:', sample_hstates)

    return


if __name__ == '__main__':
    main()
