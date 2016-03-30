from operator import itemgetter

from .pohmm import Pohmm, PSTATE_COL


class PohmmClassifier(object):
    """
    Train a POHMM for each label and make predictions by choosing the maximum likelihood model
    """

    def __init__(self, pohmm_factory):
        self.pohmms = {}
        self.pohmm_factory = pohmm_factory
        return

    def fit(self, labels, samples, pstates):
        """
        Fit the classifier with labels y and observations X
        """
        assert len(labels) == len(samples) == len(pstates)

        for label in set(labels):
            label_samples = [s for l,s in zip(labels, samples) if l == label]
            label_pstates = [p for l,p in zip(labels, pstates) if l == label]

            pohmm = self.pohmm_factory()
            pohmm.fit(label_samples, label_pstates)
            self.pohmms[label] = pohmm

        return self

    def fit_df(self, labels, dfs, pstate_col=PSTATE_COL):
        """
        Fit the classifier with labels y and DataFrames dfs
        """
        assert len(labels) == len(dfs)

        for label in set(labels):
            label_dfs = [s for l,s in zip(labels, dfs) if l == label]

            pohmm = self.pohmm_factory()
            pohmm.fit_df(label_dfs, pstate_col=pstate_col)
            self.pohmms[label] = pohmm

        return self

    def predict(self, sample, pstates):
        """
        Predict the class label of X
        """
        scores = {}
        for label, pohmm in self.pohmms.items():
            scores[label] = pohmm.score(sample, pstates)
        max_score_label = max(scores.items(), key=itemgetter(1))[0]
        return max_score_label, scores

    def predict_df(self, df, pstate_col=PSTATE_COL):
        """
        Predict the class label of DataFrame df
        """
        scores = {}
        for label, pohmm in self.pohmms.items():
            scores[label] = pohmm.score_df(df, pstate_col=pstate_col)
        max_score_label = max(scores.items(), key=itemgetter(1))[0]
        return max_score_label, scores
