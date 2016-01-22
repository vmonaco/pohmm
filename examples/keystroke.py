import numpy as np
import pandas as pd

from pohmm import PohmmClassifier, PohmmVerifier
from pohmm.utils import download

# CMU Keystroke Dynamics Benchmark Dataset
# See: http://www.cs.cmu.edu/~keystroke/
# Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics"
DATASET_URL = 'http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv'


def preprocess(df):
    """
    Convert the CMU dataset from row vectors into time/duration row observations
    """
    def process_row(idx_row):
        idx, row = idx_row
        timepress = 1000*np.r_[0, row[4::3].astype(float).values]
        duration = 1000*row[3::3].astype(float).values
        keyname = list('.tie5Roanl') + ['enter']

        return pd.DataFrame.from_items([
            ('user', [row['subject']]*11),
            ('session', [row['sessionIndex']*100 + row['rep']]*11),
            ('time', timepress),
            ('duration', duration),
            ('keyname', keyname)
        ])

    df = pd.concat(map(process_row, df.iterrows())).set_index(['user','session'])
    return df


def main():
    # Download and preprocess the CMU dataset
    df = pd.read_csv(DATASET_URL)
    df = preprocess(df)

    from IPython import embed
    embed()
    raise Exception()

    # Obtain identification results using a 10-fold cross validation



    # Obtain verification results using the cross validation procedure described at:
    # http://www.cs.cmu.edu/~keystroke/

    return


if __name__ == '__main__':
    main()
