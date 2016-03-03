#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POHMM predictions example.
"""

from __future__ import print_function

import numpy as np
import pohmm

OBS_DISTR = ['normal']
PSTATES = ['a', 'b', 'c']
SEED = 1234


def gen_pohmm():
    hmm = pohmm.Pohmm(n_hidden_states=2,
                      init_spread=2,
                      emissions=OBS_DISTR,
                      smoothing='freq',
                      init_method='obs',
                      thresh=1e-2)
    return hmm


def main(n_obs=50, reps=30):
    sampler = gen_pohmm()
    sampler.rand(PSTATES, random_state=SEED)

    baseline_residuals = np.zeros(shape=(reps, len(OBS_DISTR)))
    pohmm_residuals = np.zeros(shape=(reps, len(OBS_DISTR)))

    for i in range(reps):
        # Generate a sample
        df = sampler.sample_df(n_obs=n_obs + 1, random_state=i, hstate_col=None)

        next_obs = df[sampler.emission_name].iloc[-1].values
        next_pstate = df['event'].iloc[-1]
        df_pred = df[:-1]

        # Fit a model to the sample
        hmm = gen_pohmm()
        hmm.fit_df([df_pred])

        # Make a prediction

        # With unknown next pstate
        # prediction = hmm.predict_df(df_pred)

        # With known next pstate
        prediction = hmm.predict_df(df_pred, next_pstate=next_pstate)

        baseline_prediction = df_pred[hmm.emission_name].values.mean(axis=0)

        pohmm_residuals[i] = np.abs(prediction - next_obs)
        baseline_residuals[i] = np.abs(baseline_prediction - next_obs)

    print('Mean baseline prediction residual ...')
    print(baseline_residuals.mean(axis=0))

    print('Mean POHMM prediction residual ...')
    print(pohmm_residuals.mean(axis=0))

    return


if __name__ == '__main__':
    main()
