#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POHMM classification example.
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


def main(n_classes=10, n_samples=1, n_obs=50):
    models = {i: gen_pohmm() for i in range(n_classes)}

    train_labels, train_samples = [], []
    test_labels, test_samples = [], []
    for i, model in models.items():
        model.rand(PSTATES)
        for _ in range(n_samples):
            train_labels.append(i)
            train_samples.append(model.sample_df(n_obs=n_obs, hstate_col=None))

            test_labels.append(i)
            test_samples.append(model.sample_df(n_obs=n_obs, hstate_col=None))

    pc = pohmm.PohmmClassifier(pohmm_factory=gen_pohmm)
    pc.fit_df(train_labels, train_samples)

    predictions = []
    for test_label, test_sample in zip(test_labels, test_samples):
        prediction, _ = pc.predict_df(test_sample)
        predictions.append(prediction)

    test_labels = np.array(test_labels)
    predictions = np.array(predictions)

    print('ACC', (test_labels == predictions).sum() / len(test_labels))
    return


if __name__ == '__main__':
    main()
