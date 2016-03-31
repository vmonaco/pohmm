import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc
from sklearn.metrics.ranking import _binary_clf_curve

from pohmm import Pohmm, PohmmClassifier

# CMU Keystroke Dynamics Benchmark Dataset
# See: http://www.cs.cmu.edu/~keystroke/
# Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics"
DATASET_URL = 'http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv'


def stratified_kfold(df, n_folds):
    """
    Create stratified k-folds from an indexed dataframe
    """
    sessions = pd.DataFrame.from_records(list(df.index.unique())).groupby(0).apply(lambda x: x[1].unique())
    sessions.apply(lambda x: np.random.shuffle(x))
    folds = []
    for i in range(n_folds):
        idx = sessions.apply(lambda x: pd.Series(x[i * (len(x) / n_folds):(i + 1) * (len(x) / n_folds)]))
        idx = pd.DataFrame(idx.stack().reset_index(level=1, drop=True)).set_index(0, append=True).index.values
        folds.append(df.loc[idx])
    return folds


def user_folds(df, target):
    users = df.index.get_level_values(0).unique()
    return [df.loc[u].reset_index().set_index([target, 'session']) for u in users]


def preprocess(df):
    """Convert the CMU dataset from row vectors into time/duration row observations"""

    def process_row(idx_row):
        idx, row = idx_row

        # press-press latency
        tau = 1000 * row[4::3].astype(float).values
        tau = np.r_[np.median(tau), tau]

        # duration
        duration = 1000 * row[3::3].astype(float).values

        keyname = list('.tie5Roanl') + ['enter']

        return pd.DataFrame.from_items([
            ('user', [row['subject']] * 11),
            ('session', [row['sessionIndex'] * 100 + row['rep']] * 11),
            ('tau', tau),
            ('duration', duration),
            ('event', keyname)
        ])

    df = pd.concat(map(process_row, df.iterrows())).set_index(['user', 'session'])
    return df


def roc_curve(y_true, y_score):
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=None, sample_weight=None)

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1e-2, thresholds]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return fpr, 1 - tpr, thresholds


def ROC(scores):
    # Generate an ROC curve for each fold, ordered by increasing threshold
    roc = scores.groupby('user').apply(lambda x: pd.DataFrame(np.c_[roc_curve(x['genuine'], x['score'])][::-1],
                                                              columns=['far', 'frr', 'threshold']))

    # interpolate to get the same threshold values in each fold
    thresholds = np.sort(roc['threshold'].unique())
    roc = roc.groupby(level='user').apply(lambda x: pd.DataFrame(np.c_[thresholds,
                                                                       interp(thresholds, x['threshold'], x['far']),
                                                                       interp(thresholds, x['threshold'], x['frr'])],
                                                                 columns=['threshold', 'far', 'frr']))
    roc = roc.reset_index(level=1, drop=True).reset_index()
    return roc


def EER(roc):
    far, frr = roc['far'].values, roc['frr'].values

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    def seg_intersect(a1, a2, b1, b2):
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom) * db + b1

    d = far <= frr
    idx = np.diff(d).nonzero()[0][0]
    return seg_intersect(np.array([idx, far[idx]]),
                         np.array([idx + 1, far[idx + 1]]),
                         np.array([idx, frr[idx]]),
                         np.array([idx + 1, frr[idx + 1]]))[1]


def AUC(roc):
    return auc(roc['frr'].values, roc['far'].values)


def keystroke_model():
    """Generates a 2-state model with lognormal emissions and frequency smoothing"""
    model = Pohmm(n_hidden_states=2,
                  init_spread=2,
                  emissions=['lognormal', 'lognormal'],
                  smoothing='freq',
                  init_method='obs',
                  thresh=1)
    return model


def identification(df, n_folds=10, seed=1234):
    # Obtain identification results using k-fold cross validation
    np.random.seed(seed)

    folds = stratified_kfold(df, n_folds)

    identification_results = []
    for i in range(n_folds):
        print('Fold %d of %d' % (i + 1, n_folds))

        test_idx, test_samples = zip(*folds[i].groupby(level=[0, 1]))
        train_idx, train_samples = zip(*pd.concat(folds[:i] + folds[i + 1:]).groupby(level=[0, 1]))

        test_labels, _ = zip(*test_idx)
        train_labels, _ = zip(*train_idx)

        cl = PohmmClassifier(keystroke_model)
        cl.fit_df(train_labels, train_samples)

        for test_label, test_sample in zip(test_labels, test_samples):
            result, _ = cl.predict_df(test_sample)
            identification_results.append((i, test_label, result))

    identification_results = pd.DataFrame.from_records(identification_results,
                                                       columns=['fold', 'label', 'prediction'])
    acc_summary = identification_results.groupby('fold').apply(
        lambda x: (x['label'] == x['prediction']).sum() / len(x)).describe()

    print('Identification summary')
    print('ACC: %.3f +/- %.3f' % (acc_summary['mean'], acc_summary['std']))
    return


def verification(df):
    verification_results = []
    users = set(df.index.get_level_values(level='user').unique())
    for genuine_user in users:
        impostor_users = users.difference([genuine_user])
        genuine_samples = df.loc[genuine_user]

        _, genuine_samples = zip(*genuine_samples.groupby(level='session'))

        train, test = genuine_samples[150:200], genuine_samples[200:]

        pohmm = keystroke_model()
        pohmm.fit_df(train)

        # train_scores = np.array([pohmm.score_df(sample) for sample in train])
        scores = []
        for sample in test:
            score = pohmm.score_df(sample)
            scores.append(score)
            verification_results.append((genuine_user, True, score))

        for imposter_user in impostor_users:
            _, impostor_samples = zip(*df.loc[imposter_user].groupby(level='session'))
            for sample in impostor_samples[:5]:
                score = pohmm.score_df(sample)
                scores.append(score)
                verification_results.append((genuine_user, False, score))

    verification_results = pd.DataFrame.from_records(verification_results,
                                                     columns=['user', 'genuine', 'score'])

    verification_ROC = verification_results.groupby('user').apply(ROC)
    eer_summary = verification_ROC.groupby('user').apply(EER).describe()
    auc_summary = verification_ROC.groupby('user').apply(AUC).describe()

    print('Verification summary')
    print('EER: %.3f +/- %.3f' % (eer_summary['mean'], eer_summary['std']))
    print('AUC: %.3f +/- %.3f' % (auc_summary['mean'], auc_summary['std']))
    return


if __name__ == '__main__':
    print('This example takes about 15 minutes to run on an Intel i5...')

    # Download and preprocess the CMU dataset
    df = pd.read_csv(DATASET_URL)
    df = preprocess(df)

    # Verification results obtained using the 4th session as training data,
    # sessions 5-8 as genuine and reps 1-5 as impostor
    verification(df)

    # Identification results obtained by 10-fold stratified cross validation using only the last session
    identification(df.groupby(level=0).apply(lambda x: x[-(11 * 50):]).reset_index(level=0, drop=True))
