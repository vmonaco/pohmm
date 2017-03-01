import numpy as np
import scipy.stats as stats
from collections import defaultdict, Counter
from . import _hmmc
from .utils import *

NEGINF = -np.inf

# Label for the missing/unknown p-states
UNKNOWN_PSTATE = 'UNKNOWN'

# There are many places where a 0 probability must be avoided. Assume a very small proba instead
MIN_PROBA = 1e-5

# Tolerance for assertion statements where a float array must, e.g., sum to 1
TOLERANCE = 1e-10

HSTATE_COL = 'state'
PSTATE_COL = 'event'

_DISTRIBUTIONS = {
    'normal': ['mu', 'sigma'],
    'lognormal': ['logmu', 'logsigma'],
}

_SMOOTHING_TYPES = [
    None,
    'fixed',
    'freq',
    'proba',
    'exp',
]

_INIT_METHODS = [
    'obs',
    'rand',
]

_SMOOTHING_DEFAULTS = {
    'startprob': 'fixed',
    'transmat': 'freq',
    'mu': 'freq',
    'logmu': 'freq',
    'logsigma': 'proba',
    'sigma': 'proba'
}

_RANDINIT = {
    'normal': {
        'mu': (100, 250),
        'sigma': (10, 50)
    },
    'lognormal': {
        'logmu': (4.5, 6),
        'logsigma': (0.10, 1)
    },
}


class Pohmm(object):
    """
    Partially observable hidden Markov model
    """

    def __init__(self,
                 n_hidden_states=2,
                 emissions=['normal'],
                 max_iter=1000,
                 thresh=1e-6,
                 init_method='obs',
                 init_spread=2,
                 smoothing=None,
                 random_state=None):

        if type(n_hidden_states) is int:
            self.n_hidden_states = n_hidden_states
        else:
            raise Exception('Wrong type for n_hidden_states. Must be int')

        if type(emissions[0]) is tuple:
            emission_name, emission_distr = zip(*emissions)
        elif type(emissions[0]) is str:
            emission_name, emission_distr = np.arange(len(emissions)), emissions

        for distr in emission_distr:
            if distr not in _DISTRIBUTIONS.keys():
                raise ValueError('Emission distribution must be one of', _DISTRIBUTIONS.keys())

        self.emission_name = emission_name
        self.emission_distr = emission_distr
        self.emission_name_distr = dict(zip(emission_name, emission_distr))
        self.n_features = len(emissions)

        # Set up the emission parameters
        # emission: {'feature':{'param': np.array(shape=(n_partial_states, n_hidden_states))}}
        self.emission = defaultdict(dict)
        for name, distr in zip(self.emission_name, self.emission_distr):
            for param in _DISTRIBUTIONS[distr]:
                self.emission[name][param] = None
        self.emission = dict(self.emission)

        assert max_iter >= 0
        assert thresh >= 0
        self.max_iter = max_iter
        self.thresh = thresh

        assert init_spread >= 0
        self.init_spread = init_spread

        if init_method not in _INIT_METHODS:
            raise ValueError('init_method must be one of', _INIT_METHODS)
        self.init_method = init_method

        if smoothing is None:
            smoothing = {'transmat': None, 'startprob': None}

            for name, distr in zip(self.emission_name, self.emission_distr):
                for param in _DISTRIBUTIONS[distr]:
                    smoothing['%s:%s' % (name, param)] = None

        elif type(smoothing) is str:
            s = smoothing
            smoothing = {'transmat': s, 'startprob': s}

            for name, distr in zip(self.emission_name, self.emission_distr):
                for param in _DISTRIBUTIONS[distr]:
                    smoothing['%s:%s' % (name, param)] = s

        elif type(smoothing) is dict:
            assert 'transmat' in smoothing.keys()
            assert 'startprob' in smoothing.keys()

            for name, distr in zip(self.emission_name, self.emission_distr):
                for param in _DISTRIBUTIONS[distr]:
                    assert param in smoothing.keys() or '%s:%s' % (name, param) in smoothing.keys()
                    if param in smoothing.keys() and '%s:%s' % (name, param) not in smoothing.keys():
                        smoothing['%s:%s' % (name, param)] = smoothing[param]
        else:
            raise Exception('Wrong type for smoothing. Must be None, str, or dict')

        self.smoothing = smoothing

        self.random_state = random_state

        # Number of unique partial states is unknown until fit
        self.n_partial_states = None

        # Results after fitting the model
        self.logprob_ = None
        self.n_iter_performed_ = None
        self.logprob_delta_ = None

        # Mapping between p-states and a unique index
        # Defaults to 0 for unknown or missing p-states
        self.e = defaultdict(int)

    def _get_startprob(self):
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.ones(shape=(self.n_partial_states, self.n_hidden_states)) / self.n_hidden_states
        else:
            startprob = np.asarray(startprob, dtype=np.float)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(startprob):
            startprob = normalize(startprob, axis=1)

        if len(startprob) != self.n_partial_states:
            raise ValueError('startprob must have length n_partial_states')
        if not np.allclose(np.sum(startprob, axis=1), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = np.log(np.asarray(startprob).copy())

    startprob = property(_get_startprob, _set_startprob)

    def _get_steadyprob(self):
        return np.exp(self._log_steadyprob)

    def _set_steadyprob(self, steadyprob):
        if steadyprob is None:
            steadyprob = np.ones(shape=(self.n_partial_states, self.n_hidden_states)) / self.n_hidden_states
        else:
            steadyprob = np.asarray(steadyprob, dtype=np.float)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(steadyprob):
            steadyprob = normalize(steadyprob, axis=1)

        if len(steadyprob) != self.n_partial_states:
            raise ValueError('steadyprob must have length n_partial_states')
        if not np.allclose(np.sum(steadyprob, axis=1), 1.0):
            raise ValueError('steadyprob must sum to 1.0')

        self._log_steadyprob = np.log(np.asarray(steadyprob).copy())

    steadyprob = property(_get_steadyprob, _set_steadyprob)

    def _get_transmat(self):
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat):
        if transmat is None:
            transmat = np.ones(shape=(self.n_partial_states, self.n_partial_states, self.n_hidden_states,
                                      self.n_hidden_states)) / self.n_hidden_states

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(transmat):
            transmat = normalize(transmat, axis=3)

        if (np.asarray(transmat).shape
                != (self.n_partial_states, self.n_partial_states, self.n_hidden_states, self.n_hidden_states)):
            raise ValueError('transmat must have shape '
                             '(n_partial_states,n_partial_states,n_hidden_states,n_hidden_states)')
        if not np.all(np.allclose(np.sum(transmat, axis=3), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = np.log(np.asarray(transmat).copy())
        underflow_idx = np.isnan(self._log_transmat)
        self._log_transmat[underflow_idx] = NEGINF

    transmat = property(_get_transmat, _set_transmat)

    def _compute_log_likelihood(self, obs, pstates_idx):

        q = np.zeros(shape=(len(obs), self.n_hidden_states, self.n_features))

        for col, (feature_name, feature_distr) in enumerate(zip(self.emission_name, self.emission_distr)):
            if feature_distr == 'normal':
                mu = self.emission[feature_name]['mu'][pstates_idx]
                sigma = self.emission[feature_name]['sigma'][pstates_idx]
                for j in range(self.n_hidden_states):
                    q[:, j, col] = np.log(
                            np.maximum(MIN_PROBA, stats.norm.pdf(obs[:, col], loc=mu[:, j], scale=sigma[:, j])))
            if feature_distr == 'lognormal':
                logmu = self.emission[feature_name]['logmu'][pstates_idx]
                logsigma = self.emission[feature_name]['logsigma'][pstates_idx]
                for j in range(self.n_hidden_states):
                    q[:, j, col] = np.log(np.maximum(MIN_PROBA,
                                                     stats.lognorm.pdf(obs[:, col], logsigma[:, j], loc=0,
                                                                       scale=np.exp(logmu[:, j]))))

        q = q.sum(axis=2)
        return q

    def _generate_sample_from_state(self, hidden_state, pstates_idx, random_state=None):
        sample = np.zeros(self.n_features)
        for col, (feature_name, feature_distr) in enumerate(zip(self.emission_name, self.emission_distr)):
            if feature_distr == 'normal':
                mu = self.emission[feature_name]['mu'][pstates_idx][hidden_state]
                sigma = self.emission[feature_name]['sigma'][pstates_idx][hidden_state]
                sample[col] = stats.norm.rvs(loc=mu, scale=sigma, random_state=random_state)
            if feature_distr == 'lognormal':
                logmu = self.emission[feature_name]['logmu'][pstates_idx][hidden_state]
                logsigma = self.emission[feature_name]['logsigma'][pstates_idx][hidden_state]
                sample[col] = stats.lognorm.rvs(logsigma, loc=0, scale=np.exp(logmu), random_state=random_state)
        return sample

    def _init_pstates(self, unique_pstates):
        # Map events to a unique index. The unknown p-state is at idx 0
        self.e = defaultdict(int)
        self.e.update(dict(zip(np.sort(unique_pstates), range(1, len(unique_pstates) + 1))))
        self.er = {v: k for k, v in self.e.items()}  # Reverse lookup
        self.er[0] = UNKNOWN_PSTATE
        self.n_partial_states = len(self.e.keys()) + 1  # Add one for the unknown state
        return

    def _init_pstate_freqs(self, pstates_idx):
        # Partial state frequencies
        self.pstate_freq = Counter([idx for seq in pstates_idx for idx in seq])
        self.pstate_trans_freq = Counter([(idx1, idx2) for seq in pstates_idx for idx1, idx2 in zip(seq[:-1], seq[1:])])

        # Store freqs for the meta state
        self.pstate_freq[0] = len(np.concatenate(pstates_idx))

        for seq in pstates_idx:
            self.pstate_startprob[seq[0]] += 1
            self.pstate_steadyprob += np.bincount(seq, minlength=self.n_partial_states)
            for idx1, idx2 in zip(seq[:-1], seq[1:]):
                self.pstate_trans_freq[(0, 0)] += 1
                self.pstate_trans_freq[(idx1, 0)] += 1
                self.pstate_trans_freq[(0, idx2)] += 1

                self.pstate_transmat[idx1, idx2] += 1

        # TODO: separate probas from freqs
        # Normalize to get the probabilities, ignore the meta state at idx 0
        self.pstate_startprob[1:] = normalize(self.pstate_startprob[1:])
        self.pstate_transmat[1:, 1:] = normalize(self.pstate_transmat[1:, 1:], axis=1)
        self.pstate_steadyprob[1:] = normalize(self.pstate_steadyprob[1:])

        return

    def _init_from_obs(self, obs, pstates_idx):
        # Partial state probabilities
        self.pstate_startprob = np.zeros(self.n_partial_states)
        self.pstate_transmat = np.zeros((self.n_partial_states, self.n_partial_states))
        self.pstate_steadyprob = np.zeros(self.n_partial_states)

        # obs should be (N*T, n_features)
        # N is the number of samples
        # T is the size of each sample 
        obs = np.concatenate(obs)
        pstates_idx = np.concatenate(pstates_idx)

        # Initialize starting and transition probas
        self.startprob = np.ones(shape=(self.n_partial_states, self.n_hidden_states)) / self.n_hidden_states
        self.steadyprob = np.ones(shape=(self.n_partial_states, self.n_hidden_states)) / self.n_hidden_states
        self.transmat = np.ones(shape=(self.n_partial_states, self.n_partial_states, self.n_hidden_states,
                                       self.n_hidden_states)) / self.n_hidden_states

        # Initialize emission parameters
        # Hidden states are ordered by the first feature
        feature1 = self.emission_name[0]
        for col, (feature_name, feature_distr) in enumerate(zip(self.emission_name, self.emission_distr)):
            if feature_distr == 'normal':
                self.emission[feature_name]['mu'] = np.zeros(shape=(self.n_partial_states, self.n_hidden_states))
                self.emission[feature_name]['sigma'] = np.zeros(shape=(self.n_partial_states, self.n_hidden_states))
                for idx in range(1, self.n_partial_states):
                    idx_pstate = (pstates_idx == idx)

                    if not np.any(idx_pstate):
                        idx_pstate = np.arange(len(pstates_idx))

                    if feature_name == feature1:
                        self.emission[feature_name]['sigma'][idx, :] = np.maximum(obs[idx_pstate, col].std(), MIN_PROBA)
                        self.emission[feature_name]['mu'][idx] = obs[idx_pstate, col].mean() + obs[:,
                                                                                               col].std() * np.linspace(
                                -self.init_spread, self.init_spread, self.n_hidden_states)
                    else:
                        self.emission[feature_name]['sigma'][idx, :] = np.maximum(obs[idx_pstate, col].std(), MIN_PROBA)
                        self.emission[feature_name]['mu'][idx, :] = obs[idx_pstate, col].mean()

            if feature_distr == 'lognormal':
                self.emission[feature_name]['logmu'] = np.zeros(shape=(self.n_partial_states, self.n_hidden_states))
                self.emission[feature_name]['logsigma'] = np.zeros(shape=(self.n_partial_states, self.n_hidden_states))

                for idx in range(1, self.n_partial_states):
                    idx_pstate = (pstates_idx == idx)

                    if not np.any(idx_pstate):
                        idx_pstate = np.arange(len(pstates_idx))

                    if feature_name == feature1:
                        self.emission[feature_name]['logsigma'][idx, :] = np.maximum(np.log(obs[idx_pstate, col]).std(),
                                                                                     MIN_PROBA)
                        self.emission[feature_name]['logmu'][idx] = np.log(obs[idx_pstate, col]).mean() + np.log(
                                obs[:, col]).std() * np.linspace(-self.init_spread, self.init_spread,
                                                                 self.n_hidden_states)
                    else:
                        self.emission[feature_name]['logsigma'][idx, :] = np.maximum(np.log(obs[idx_pstate, col]).std(),
                                                                                     MIN_PROBA)
                        self.emission[feature_name]['logmu'][idx] = np.log(obs[idx_pstate, col]).mean()

        return

    def _init_random(self, random_state=None):

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        self.pstate_startprob = np.zeros(self.n_partial_states)
        self.pstate_transmat = np.zeros((self.n_partial_states, self.n_partial_states))
        self.pstate_steadyprob = np.zeros(self.n_partial_states)

        self.pstate_startprob[1:] = gen_stochastic_matrix(size=self.n_partial_states - 1, random_state=random_state)
        self.pstate_transmat[1:, 1:] = gen_stochastic_matrix(
                size=(self.n_partial_states - 1, self.n_partial_states - 1), random_state=random_state)
        self.pstate_steadyprob[1:] = steadystate(self.pstate_transmat[1:, 1:])

        self.startprob = gen_stochastic_matrix(size=(self.n_partial_states, self.n_hidden_states),
                                               random_state=random_state)

        transmat = np.zeros((self.n_partial_states, self.n_partial_states, self.n_hidden_states, self.n_hidden_states))
        for i, j in product(range(1, self.n_partial_states), range(1, self.n_partial_states)):
            transmat[i, j] = gen_stochastic_matrix(size=(self.n_hidden_states, self.n_hidden_states),
                                                   random_state=random_state)
        self.transmat = normalize(transmat, axis=3)

        # Initialize emission parameters
        for feature_name, feature_distr in zip(self.emission_name, self.emission_distr):
            if feature_distr == 'normal':
                self.emission[feature_name]['mu'] = random_state.uniform(*_RANDINIT[feature_distr]['mu'], size=(
                    self.n_partial_states, self.n_hidden_states))
                self.emission[feature_name]['sigma'] = random_state.uniform(*_RANDINIT[feature_distr]['sigma'], size=(
                    self.n_partial_states, self.n_hidden_states))
            if feature_distr == 'lognormal':
                self.emission[feature_name]['logmu'] = random_state.uniform(*_RANDINIT[feature_distr]['logmu'], size=(
                    self.n_partial_states, self.n_hidden_states))
                self.emission[feature_name]['logsigma'] = random_state.uniform(*_RANDINIT[feature_distr]['logsigma'],
                                                                               size=(self.n_partial_states,
                                                                                     self.n_hidden_states))

        if self.emission_distr[0] == 'normal':
            self.emission[self.emission_name[0]]['mu'] = np.sort(self.emission[self.emission_name[0]]['mu'], axis=1)
        elif self.emission_distr[0] == 'lognormal':
            self.emission[self.emission_name[0]]['logmu'] = np.sort(self.emission[self.emission_name[0]]['logmu'],
                                                                    axis=1)

        return

    def _smooth(self):
        self._compute_marginals()

        startprob = self.startprob
        for j in range(1, self.n_partial_states):
            if 'freq' == self.smoothing['startprob']:
                w_ = 1 / (1 + self.pstate_freq[j])
                w_j = 1 - w_
            elif 'proba' == self.smoothing['startprob']:
                w_j = self.pstate_steadyprob[j]
                w_ = 1 - w_j
            elif 'exp' == self.smoothing['startprob']:
                w_ = np.exp(-self.pstate_freq[j])
                w_j = 1 - w_
            elif 'fixed' == self.smoothing['startprob']:
                w_ = 1
                w_j = 0
            elif self.smoothing['startprob'] is None:
                w_ = 0
                w_j = 1
            else:
                raise Exception('Wrong smoothing for startprob: ' + self.smoothing['startprob'])
            startprob[j] = w_j * self.startprob[j] + w_ * self.startprob[0]
        self.startprob = startprob

        transmat = self.transmat
        for i, j in product(range(1, self.n_partial_states), range(1, self.n_partial_states)):
            if 'freq' == self.smoothing['transmat']:
                w_i0 = 1 / (1 + self.pstate_trans_freq[i, j] + self.pstate_trans_freq[0, j])
                w_0j = 1 / (1 + self.pstate_trans_freq[i, j] + self.pstate_trans_freq[i, 0])
                w_ij = 1 - (w_i0 + w_0j)
                w_ = 0
            elif 'proba' == self.smoothing['transmat']:
                denom = self.pstate_transmat[i, j] + self.pstate_transmat[i, :].sum() + self.pstate_transmat[:, j].sum()
                w_i0 = self.pstate_transmat[i, :].sum() / denom
                w_0j = self.pstate_transmat[:, j].sum() / denom
                w_ij = self.pstate_transmat[i, j] / denom
                w_ = 0
            elif 'exp' == self.smoothing['transmat']:
                w_i0 = self.pstate_trans_freq[i, 0] * np.exp(-self.pstate_trans_freq[i, j])
                w_0j = self.pstate_trans_freq[0, j] * np.exp(-self.pstate_trans_freq[i, j])
                w_ = self.pstate_trans_freq[0, 0] * np.exp(
                        -(self.pstate_trans_freq[i, 0] + self.pstate_trans_freq[0, j]))
                w_ij = self.pstate_trans_freq[i, j]
                w_ij, w_i0, w_0j, w_ = normalize(np.array([w_ij, w_i0, w_0j, w_]))
            elif 'fixed' == self.smoothing['transmat']:
                w_i0 = 0
                w_0j = 0
                w_ = 1
                w_ij = 0
            elif self.smoothing['transmat'] is None:
                w_i0 = 0
                w_0j = 0
                w_ = 0
                w_ij = 1
            else:
                raise Exception('Wrong smoothing for transmat: ' + self.smoothing['transmat'])

            assert (w_i0 + w_0j + w_ + w_ij) - 1 < TOLERANCE
            transmat[i, j] = w_ij * self.transmat[i, j] + w_i0 * self.transmat[i, 0] + w_0j * self.transmat[0, j] + w_ * \
                                                                                                                    self.transmat[
                                                                                                                        0, 0]
        self.transmat = transmat

        assert np.all(self.startprob.sum(axis=1) - 1 < TOLERANCE)
        assert np.all(self.steadyprob.sum(axis=1) - 1 < TOLERANCE)
        assert np.all(self.transmat.sum(axis=3) - 1 < TOLERANCE)

        for feature_name, feature_distr in zip(self.emission_name, self.emission_distr):
            for param in _DISTRIBUTIONS[feature_distr]:
                key = '%s:%s' % (feature_name, param)
                for j in range(1, self.n_partial_states):
                    if 'freq' == self.smoothing[key]:
                        w_ = 1 / (1 + self.pstate_freq[j])
                        w_j = 1 - w_
                    elif 'proba' == self.smoothing[key]:
                        w_j = self.pstate_steadyprob[j]
                        w_ = 1 - w_j
                    elif 'exp' == self.smoothing[key]:
                        w_ = np.exp(-self.pstate_freq[j])
                        w_j = 1 - w_
                    elif 'fixed' == self.smoothing[key]:
                        w_ = 1
                        w_j = 0
                    elif self.smoothing[key] is None:
                        w_ = 0
                        w_j = 1
                    else:
                        raise Exception('Wrong smoothing for ' + key)
                    self.emission[feature_name][param][j] = w_j * self.emission[feature_name][param][j] + w_ * \
                                                                                                          self.emission[
                                                                                                              feature_name][
                                                                                                              param][0]

        return

    def _initialize_sufficient_statistics(self):
        stats = {
            'nobs': 0,
            'post': np.zeros((self.n_partial_states, self.n_hidden_states)),
            'obs': np.zeros((self.n_partial_states, self.n_hidden_states, self.n_features)),
            'obs**2': np.zeros((self.n_partial_states, self.n_hidden_states, self.n_features)),
            'lnobs': np.zeros((self.n_partial_states, self.n_hidden_states, self.n_features)),
            'lnobs**2': np.zeros((self.n_partial_states, self.n_hidden_states, self.n_features)),
            'start': np.zeros((self.n_partial_states, self.n_hidden_states)),
            'steady': np.zeros((self.n_partial_states, self.n_hidden_states)),
            'trans': np.zeros(
                    (self.n_partial_states, self.n_partial_states, self.n_hidden_states, self.n_hidden_states))
        }

        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, pstates_idx, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        stats['nobs'] += 1
        n_observations, n_hidden_states = framelogprob.shape

        stats['start'][0] += posteriors[0]
        for i in range(self.n_partial_states):
            if len(np.where(pstates_idx == i)[0]) > 0:
                stats['start'][i] += posteriors[np.where(pstates_idx == i)[0].min()]

        if n_observations > 1:
            lneta = np.zeros((n_observations - 1, n_hidden_states, n_hidden_states))
            _hmmc._compute_lneta(n_observations, n_hidden_states, pstates_idx, fwdlattice,
                                 self._log_transmat, bwdlattice, framelogprob,
                                 lneta)

            for i, j in unique_rows(np.c_[pstates_idx[:-1], pstates_idx[1:]]):
                if ((pstates_idx[:-1] == i) & (pstates_idx[1:] == j)).sum() > 0:
                    stats['trans'][i, j] += np.exp(
                            logsumexp(lneta[(pstates_idx[:-1] == i) & (pstates_idx[1:] == j)], axis=0))

        for i in range(self.n_partial_states):
            stats['post'][i] += posteriors[pstates_idx == i].sum(axis=0)
            stats['obs'][i] += np.dot(posteriors[pstates_idx == i].T, obs[pstates_idx == i])
            stats['obs**2'][i] += np.dot(posteriors[pstates_idx == i].T, obs[pstates_idx == i] ** 2)
            stats['lnobs'][i] += np.dot(posteriors[pstates_idx == i].T, np.log(obs[pstates_idx == i]))
            stats['lnobs**2'][i] += np.dot(posteriors[pstates_idx == i].T, np.log(obs[pstates_idx == i]) ** 2)

        return

    def _compute_marginals(self):
        # TODO: cythonize some of this

        # Start prob, weighted by p-state start probs
        self.startprob[0] = (self.pstate_startprob[1:, np.newaxis] * self.startprob[1:]).sum(axis=0)

        # Use the p-state transmat and transmat to get the full transmat
        full_transmat = ph2full(self.pstate_transmat[1:, 1:], self.transmat[1:, 1:])
        full_steadyprob = steadystate(full_transmat)

        # Steady state probas are determined by the full trans mat, need to be updated
        steadyprob = np.zeros(shape=(self.n_partial_states, self.n_hidden_states))
        steadyprob[0] = full_steadyprob.reshape(-1, self.n_hidden_states).sum(axis=0)
        for i in range(self.n_partial_states - 1):
            steadyprob[i + 1] = normalize(
                    full_steadyprob[i * self.n_hidden_states:i * self.n_hidden_states + self.n_hidden_states])

        self.steadyprob = steadyprob

        # Update the transations to/from the marginal state
        transmat = self.transmat
        # Group the hidden states within each partial state
        for hidx1, hidx2 in product(range(self.n_hidden_states), range(self.n_hidden_states)):
            transmat[0, 0][hidx1, hidx2] = full_transmat[hidx1::self.n_hidden_states, hidx2::self.n_hidden_states].sum()
            for pidx in range(self.n_partial_states - 1):
                transmat[pidx + 1, 0][hidx1, hidx2] = full_transmat[pidx * self.n_hidden_states + hidx1,
                                                      hidx2::self.n_hidden_states].sum()
                transmat[0, pidx + 1][hidx1, hidx2] = full_transmat[hidx1::self.n_hidden_states,
                                                      pidx * self.n_hidden_states + hidx2].sum()
        self.transmat = normalize(transmat, axis=3)

        pweights = self.pstate_steadyprob[1:, np.newaxis]
        # Update emission parameters
        for feature_name, feature_distr in zip(self.emission_name, self.emission_distr):
            if feature_distr == 'normal':
                # Marginal state is a mixture of normals
                mu = self.emission[feature_name]['mu'][1:, :]
                sigma = self.emission[feature_name]['sigma'][1:, :]

                # Weighted mean and var
                mu_0 = (pweights * mu).sum(axis=0)
                self.emission[feature_name]['mu'][0, :] = mu_0
                self.emission[feature_name]['sigma'][0, :] = np.sqrt(
                        (pweights * ((mu - mu_0) ** 2 + sigma ** 2)).sum(axis=0))

            if feature_distr == 'lognormal':
                # Marginal state is a mixture of normals
                mu = self.emission[feature_name]['logmu'][1:, :]
                sigma = self.emission[feature_name]['logsigma'][1:, :]

                # Weighted mean and var
                mu_0 = (pweights * mu).sum(axis=0)
                self.emission[feature_name]['logmu'][0, :] = mu_0
                self.emission[feature_name]['logsigma'][0, :] = np.sqrt(
                        (pweights * ((mu - mu_0) ** 2 + sigma ** 2)).sum(axis=0))

        return

    def _do_mstep(self, stats):
        self.startprob = normalize(np.maximum(stats['start'], MIN_PROBA), axis=1)
        self.transmat = normalize(np.maximum(stats['trans'], MIN_PROBA), axis=3)

        for col, (feature_name, feature_distr) in enumerate(zip(self.emission_name, self.emission_distr)):
            if feature_distr == 'normal':
                denom = np.maximum(stats['post'], MIN_PROBA)
                mu = stats['obs'][:, :, col] / denom
                cv_num = (stats['obs**2'][:, :, col]
                          - 2 * mu * stats['obs'][:, :, col]
                          + mu ** 2 * denom)
                sigma = np.sqrt(cv_num / denom)
                sigma[np.isnan(sigma)] = MIN_PROBA
                self.emission[feature_name]['mu'] = mu
                self.emission[feature_name]['sigma'] = np.maximum(sigma, MIN_PROBA)
            if feature_distr == 'lognormal':
                denom = np.maximum(stats['post'], MIN_PROBA)
                mu = stats['lnobs'][:, :, col] / denom
                cv_num = (stats['lnobs**2'][:, :, col]
                          - 2 * mu * stats['lnobs'][:, :, col]
                          + mu ** 2 * denom)
                sigma = np.sqrt(cv_num / denom)
                sigma[np.isnan(sigma)] = MIN_PROBA
                self.emission[feature_name]['logmu'] = mu
                self.emission[feature_name]['logsigma'] = np.maximum(sigma, MIN_PROBA)

        return

    def _do_forward_pass(self, framelogprob, event_idx):
        n_observations, n_hidden_states = framelogprob.shape
        fwdlattice = np.zeros((n_observations, n_hidden_states))
        _hmmc._forward(n_observations, n_hidden_states,
                       event_idx, self._log_startprob,
                       self._log_transmat, framelogprob, fwdlattice)

        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob, event_idx):
        n_observations, n_hidden_states = framelogprob.shape
        bwdlattice = np.zeros((n_observations, n_hidden_states))
        _hmmc._backward(n_observations, n_hidden_states,
                        event_idx, self._log_startprob,
                        self._log_transmat, framelogprob, bwdlattice)
        return bwdlattice

    def _do_viterbi_pass(self, framelogprob, event_idx):
        n_observations, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
                n_observations, n_components,
                event_idx, self._log_startprob,
                self._log_transmat, framelogprob)
        return logprob, state_sequence

    def rand(self, unique_pstates, random_state=None):
        """
        Randomize the POHMM parameters
        """
        self._init_pstates(unique_pstates)
        self._init_random(random_state=random_state)
        self._compute_marginals()
        return self

    def fit(self, obs, pstates, unique_pstates=None):
        """
        Estimate model parameters.
        """
        obs = [np.array(o) for o in obs]
        pstates = [np.array(p) for p in pstates]

        # List or array of observation sequences
        assert len(obs) == len(pstates)
        assert obs[0].ndim == 2
        assert pstates[0].ndim == 1

        if unique_pstates is not None:
            self._init_pstates(unique_pstates)
        else:
            self._init_pstates(list(set(np.concatenate(pstates))))

        # Map the partial states to a unique index
        pstates_idx = [np.array([self.e[p] for p in seq]) for seq in pstates]

        if self.init_method == 'rand':
            self._init_random()
        elif self.init_method == 'obs':
            self._init_from_obs(obs, pstates_idx)

        self._init_pstate_freqs(pstates_idx)
        self._smooth()

        logprob = []
        for i in range(self.max_iter):
            # Expectation step
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for obs_i, pstates_idx_i in zip(obs, pstates_idx):
                framelogprob = self._compute_log_likelihood(obs_i, pstates_idx_i)
                lpr, fwdlattice = self._do_forward_pass(framelogprob, pstates_idx_i)
                bwdlattice = self._do_backward_pass(framelogprob, pstates_idx_i)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                curr_logprob += lpr

                self._accumulate_sufficient_statistics(stats, obs_i,
                                                       pstates_idx_i, framelogprob, posteriors, fwdlattice, bwdlattice)

            logprob.append(curr_logprob)
            self.logprob_ = curr_logprob

            # Check for convergence.
            self.n_iter_performed_ = i
            if i > 0:
                self.logprob_delta = logprob[-1] - logprob[-2]
                if self.logprob_delta < self.thresh:
                    break

            # Maximization step
            self._do_mstep(stats)

            # Mix the parameters
            self._smooth()

        return self

    def score(self, obs, pstates):
        """
        Compute the log probability under the model.
        """
        pstates_idx = np.array([self.e[p] for p in pstates])
        framelogprob = self._compute_log_likelihood(obs, pstates_idx)
        logprob, _ = self._do_forward_pass(framelogprob, pstates_idx)
        return logprob

    def score_events(self, obs, pstates):
        """
        Compute the log probability of each event under the model.
        """
        pstates_idx = np.array([self.e[p] for p in pstates])

        framelogprob = self._compute_log_likelihood(obs, pstates_idx)
        _, fwdlattice = self._do_forward_pass(framelogprob, pstates_idx)
        L = logsumexp(fwdlattice, axis=1)
        return np.concatenate([L[[0]], np.diff(L)])

    def predict_states(self, obs, pstates):
        pstates_idx = np.array([self.e[p] for p in pstates])
        framelogprob = self._compute_log_likelihood(obs, pstates_idx)
        viterbi_logprob, state_sequence = self._do_viterbi_pass(framelogprob, pstates_idx)
        return viterbi_logprob, state_sequence

    def predict(self, obs, pstates, next_pstate=None):
        """
        Predict the next observation
        """
        assert len(obs) == len(pstates)
        pstates_idx = np.array([self.e[ei] for ei in pstates])
        next_pstate_idx = self.e[next_pstate]

        if len(obs) == 0:
            # No history, use the starting probas
            next_hstate_prob = self.startprob[next_pstate_idx]
        else:
            # With a history, determine the hidden state posteriors using
            # the last posteriors and transition matrix
            framelogprob = self._compute_log_likelihood(obs, pstates_idx)
            _, fwdlattice = self._do_forward_pass(framelogprob, pstates_idx)

            next_hstate_prob = np.zeros(self.n_hidden_states)

            alpha_n = fwdlattice[-1]
            vmax = alpha_n.max(axis=0)
            alpha_n = np.exp(alpha_n - vmax)
            alpha_n = alpha_n / alpha_n.sum()

            trans = self.transmat[pstates_idx[-1], next_pstate_idx]

            for i in range(self.n_hidden_states):
                next_hstate_prob[i] = np.sum([alpha_n[j] * trans[j, i] for j in range(self.n_hidden_states)])

        assert next_hstate_prob.sum() - 1 < TOLERANCE

        # Make the prediction
        prediction = np.array(
                [self.expected_value(feature, pstate=next_pstate, hstate_prob=next_hstate_prob) for feature in
                 self.emission_name])

        # next_hstate = np.argmax(next_hstate_prob)
        # prediction = np.array(
        #     [self.expected_value(feature, pstate=next_pstate, hstate=next_hstate) for feature in
        #      self.emission_name])

        return prediction

    def gen_pstates_idx(self, n, random_state=None):

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.pstate_startprob)
        transmat_cdf = np.cumsum(self.pstate_transmat, 1)

        # Initial state.
        rand = random_state.rand()
        curr_pstate = (startprob_cdf > rand).argmax()
        pstates = [curr_pstate]

        for _ in range(1, n):
            rand = random_state.rand()
            curr_pstate = (transmat_cdf[curr_pstate] > rand).argmax()
            pstates.append(curr_pstate)

        return np.array(pstates, dtype=int)

    def sample(self, pstates=None, n_obs=None, random_state=None):
        """

        """
        random_state = check_random_state(random_state)

        if pstates is None and n_obs is None:
            raise Exception('Must provide either pstates or n_obs')

        if pstates is not None and n_obs is not None:
            raise Exception('Must provide either pstates or n_obs but not both')

        gen_pstates = False
        rand = random_state.rand()
        if pstates is None:
            gen_pstates = True
            pstartprob_cdf = np.cumsum(self.pstate_startprob)
            ptransmat_cdf = np.cumsum(self.pstate_transmat, 1)

            # Initial pstate
            currpstate = (pstartprob_cdf > rand).argmax()
            pstates_idx = [currpstate]
            pstates = [self.er[currpstate]]
        else:
            n_obs = len(pstates)
            pstates_idx = np.array([self.e[p] for p in pstates])

        startprob_pdf = self.startprob[pstates_idx[0]]
        startprob_cdf = np.cumsum(startprob_pdf)
        transmat_cdf = np.cumsum(self.transmat[0, pstates_idx[0]], 1)

        # Initial hidden state
        rand = random_state.rand()
        currstate = (startprob_cdf > rand).argmax()
        hidden_states = [currstate]
        obs = [self._generate_sample_from_state(currstate, pstates_idx[0], random_state)]

        for i in range(1, n_obs):
            rand = random_state.rand()

            if gen_pstates:
                currpstate = (ptransmat_cdf[currpstate] > rand).argmax()
                pstates_idx.append(currpstate)
                pstates.append(self.er[currpstate])

            transmat_cdf = np.cumsum(self.transmat[pstates_idx[i - 1], pstates_idx[i]], 1)
            rand = random_state.rand()
            currstate = (transmat_cdf[currstate] > rand).argmax()
            hidden_states.append(currstate)
            obs.append(self._generate_sample_from_state(currstate, pstates_idx[i], random_state))

        return np.array(obs), np.array(pstates), np.array(hidden_states, dtype=int)

    def fit_df(self, dfs, pstate_col=PSTATE_COL):
        """
        Convenience function to fit a model from a list of dataframes
        """
        obs_cols = list(self.emission_name)
        obs = [df[df.columns.difference([pstate_col])][obs_cols].values for df in dfs]
        pstates = [df[pstate_col].values for df in dfs]
        return self.fit(obs, pstates)

    def score_df(self, df, pstate_col=PSTATE_COL):
        """

        """
        obs_cols = list(self.emission_name)
        obs = df[df.columns.difference([pstate_col])][obs_cols].values
        pstates = df[pstate_col].values
        return self.score(obs, pstates)

    def score_events_df(self, df, pstate_col=PSTATE_COL, score_col='score'):
        """

        """
        df = df.copy()
        obs_cols = list(self.emission_name)
        obs = df[df.columns.difference([pstate_col])][obs_cols].values
        pstates = df[pstate_col].values
        df[score_col] = self.score_events(obs, pstates)
        return df

    def predict_states_df(self, df, pstate_col=PSTATE_COL, hstate_col=HSTATE_COL):
        df = df.copy()
        obs_cols = list(self.emission_name)
        obs = df[df.columns.difference([pstate_col])][obs_cols].values
        pstates = df[pstate_col].values
        _, df[hstate_col] = self.predict_states(obs, pstates)
        return df

    def predict_df(self, df, next_pstate=None, pstate_col=PSTATE_COL):
        obs_cols = list(self.emission_name)
        obs = df[df.columns.difference([pstate_col])][obs_cols].values
        pstates = df[pstate_col].values
        return self.predict(obs, pstates, next_pstate)

    def sample_df(self, pstates=None, n_obs=None, random_state=None, pstate_col=PSTATE_COL, hstate_col=HSTATE_COL):
        """
        Convenience function to generate samples a model and create a dataframe
        """
        try:
            import pandas as pd
        except Exception as e:
            raise e

        obs, pstates, hstates = self.sample(pstates, n_obs, random_state)
        items = []

        if pstate_col is not None:
            items.append((pstate_col, pstates))

        if hstate_col is not None:
            items.append((hstate_col, hstates))

        items = items + [(self.emission_name[i], obs[:, i]) for i in range(self.n_features)]
        df = pd.DataFrame.from_items(items)
        return df

    def __str__(self):
        pstates = sorted(self.e.keys())
        sep = '-' * 80 + '\n'
        sep2 = '_' * 40 + '\n'

        out = 'POHMM\n'
        out += 'H-states: %d\n' % self.n_hidden_states
        out += 'P-states: (%d) %s\n' % (len(pstates), str(pstates))
        out += 'Emission: %s\n' % (self.emission_distr)

        out += sep
        out += 'Transition probas\n'
        out += sep2
        out += '. -> .\n%s\n' % str(self.transmat[0, 0])
        for pstate in pstates:
            out += sep2
            out += '%s -> .\n%s\n' % (pstate, str(self.transmat[self.e[pstate], 0]))
            out += sep2
            out += '. -> %s\n%s\n' % (pstate, str(self.transmat[0, self.e[pstate]]))
        for pstate1, pstate2 in product(pstates, pstates):
            out += sep2
            out += '%s -> %s\n%s\n' % (pstate1, pstate2, str(self.transmat[self.e[pstate1], self.e[pstate2]]))

        out += sep
        out += 'Starting probas\n'
        out += '.: %s\n' % str(self.startprob[0])
        for pstate in pstates:
            out += '%s: %s\n' % (pstate, str(self.startprob[self.e[pstate]]))

        out += sep
        out += 'Steady probas\n'
        out += '.: %s\n' % str(self.steadyprob[0])
        for pstate in pstates:
            out += '%s: %s\n' % (pstate, str(self.steadyprob[self.e[pstate]]))

        out += sep
        out += 'Emissions\n'
        for feature_name, feature_distr in zip(self.emission_name, self.emission_distr):
            out += sep2
            out += 'Feature %s: %s\n' % (feature_name, feature_distr)
            for param in _DISTRIBUTIONS[feature_distr]:
                out += '.: %s = %s\n' % (param, str(self.emission[feature_name][param][0]))
                for pstate in pstates:
                    out += '%s: %s = %s\n' % (pstate, param, str(self.emission[feature_name][param][self.e[pstate]]))

        out += sep
        return out

    def expected_value(self, feature=None, pstate=None, hstate=None, pstate_prob=None, hstate_prob=None):
        """
        Determine the joint maximum likelihood estimate
        """
        # Use the first feature by default
        if feature is None:
            feature = self.emission_name[0]

        # Will default to marginal pstate if pstate is unknown or None
        pstate_idx = self.e[pstate]

        if pstate is not None and pstate_prob is not None:
            raise Exception('Must provide either pstate or pstate_proba but not both')

        if hstate is not None and hstate_prob is not None:
            raise Exception('Must provide either hstate or hstate_proba but not both')

        # Marginalize pstate using the steady state probas
        if pstate_prob is None:
            pstate_prob = self.pstate_steadyprob

        # Marginalize hstate using the steady state probas
        if hstate_prob is None:
            hstate_prob = self.steadyprob[pstate_idx]

        if pstate is None and hstate is None:
            # Marginalize both pstate and hstate
            w = (pstate_prob[:, np.newaxis] * hstate_prob).flatten()

            if self.emission_name_distr[feature] == 'lognormal':
                return np.sum(w * expected_lognormal(self.emission[feature]['logsigma'].flatten(),
                                                     self.emission[feature]['logmu'].flatten()))
            elif self.emission_name_distr[feature] == 'normal':
                return np.sum(w * expected_normal(self.emission[feature]['mu'].flatten(),
                                                  self.emission[feature]['sigma'].flatten()))
        elif hstate is None:
            # Marginalize hstate
            if self.emission_name_distr[feature] == 'lognormal':
                return np.sum(hstate_prob * expected_lognormal(self.emission[feature]['logsigma'][pstate_idx, :],
                                                               self.emission[feature]['logmu'][pstate_idx, :]))
            elif self.emission_name_distr[feature] == 'normal':
                return np.sum(hstate_prob * expected_normal(self.emission[feature]['mu'][pstate_idx, :],
                                                            self.emission[feature]['sigma'][pstate_idx, :]))

        elif pstate is None:
            # Marginalize pstate
            if self.emission_name_distr[feature] == 'lognormal':
                return np.sum(pstate_prob * expected_lognormal(self.emission[feature]['logsigma'][:, hstate],
                                                               self.emission[feature]['logmu'][:, hstate]))
            elif self.emission_name_distr[feature] == 'normal':
                return np.sum(pstate_prob * expected_normal(self.emission[feature]['mu'][:, hstate],
                                                            self.emission[feature]['sigma'][:, hstate]))
        else:
            if self.emission_name_distr[feature] == 'lognormal':
                return expected_lognormal(self.emission[feature]['logsigma'][pstate_idx, hstate],
                                          self.emission[feature]['logmu'][pstate_idx, hstate])
            elif self.emission_name_distr[feature] == 'normal':
                return expected_normal(self.emission[feature]['mu'][pstate_idx, hstate],
                                       self.emission[feature]['sigma'][pstate_idx, hstate])

        return

    def pdf_fn(self, feature=None, pstate=None, hstate=None, pstate_prob=None, hstate_prob=None):
        # Use the first feature by default
        if feature is None:
            feature = self.emission_name[0]

        # Will default to marginal pstate if pstate is unknown or None
        pstate_idx = self.e[pstate]

        if pstate is not None and pstate_prob is not None:
            raise Exception('Must provide either pstate or pstate_proba but not both')

        if hstate is not None and hstate_prob is not None:
            raise Exception('Must provide either hstate or hstate_proba but not both')

        # Marginalize pstate using the steady state probas
        if pstate_prob is None:
            pstate_prob = self.pstate_steadyprob

        # Marginalize hstate using the steady state probas
        if hstate_prob is None:
            hstate_prob = self.steadyprob[pstate_idx]

        if pstate is None and hstate is None:
            # Marginalize both pstate and hstate
            w = (pstate_prob[:, np.newaxis] * hstate_prob).flatten()

            if self.emission_name_distr[feature] == 'lognormal':
                pdf = lambda x: stats.lognorm.pdf(x, self.emission[feature]['logsigma'].flatten(), 0,
                                                  np.exp(self.emission[feature]['logmu'].flatten()))
            elif self.emission_name_distr[feature] == 'normal':
                pdf = lambda x: stats.norm.pdf(x, self.emission[feature]['mu'].flatten(),
                                               self.emission[feature]['sigma'].flatten())
        elif hstate is None:
            # Marginalize hstate
            w = hstate_prob

            if self.emission_name_distr[feature] == 'lognormal':
                pdf = lambda x: stats.lognorm.pdf(x, self.emission[feature]['logsigma'][pstate_idx, :], 0,
                                                  np.exp(self.emission[feature]['logmu'][pstate_idx, :]))
            elif self.emission_name_distr[feature] == 'normal':
                pdf = lambda x: stats.norm.pdf(x, self.emission[feature]['mu'][pstate_idx, :],
                                               self.emission[feature]['sigma'][pstate_idx, :])
        elif pstate is None:
            # Marginalize pstate
            w = pstate_prob

            if self.emission_name_distr[feature] == 'lognormal':
                pdf = lambda x: stats.lognorm.pdf(x, self.emission[feature]['logsigma'][:, hstate], 0,
                                                  np.exp(self.emission[feature]['logmu'][:, hstate]))
            elif self.emission_name_distr[feature] == 'normal':
                pdf = lambda x: stats.norm.pdf(x, self.emission[feature]['mu'][:, hstate],
                                               self.emission[feature]['sigma'][:, hstate])
        else:
            w = 1

            if self.emission_name_distr[feature] == 'lognormal':
                pdf = lambda x: stats.lognorm.pdf(x, self.emission[feature]['logsigma'][pstate_idx, hstate], 0,
                                                  np.exp(self.emission[feature]['logmu'][pstate_idx, hstate]))
            elif self.emission_name_distr[feature] == 'normal':
                pdf = lambda x: stats.norm.pdf(x, self.emission[feature]['mu'][pstate_idx, hstate],
                                               self.emission[feature]['sigma'][pstate_idx, hstate])

        def fn(x):
            if np.isscalar(x):
                p = np.sum(w * pdf(x))
            else:
                x = np.array(x)
                p = np.zeros(len(x))
                for i, xi in enumerate(x):
                    p[i] = np.sum(w * pdf(xi))
            return p

        return fn

    def cdf_fn(self, feature=None, pstate=None, hstate=None, pstate_prob=None, hstate_prob=None):
        # Use the first feature by default
        if feature is None:
            feature = self.emission_name[0]

        # Will default to marginal pstate if pstate is unknown or None
        pstate_idx = self.e[pstate]

        if pstate is not None and pstate_prob is not None:
            raise Exception('Must provide either pstate or pstate_proba but not both')

        if hstate is not None and hstate_prob is not None:
            raise Exception('Must provide either hstate or hstate_proba but not both')

        # Marginalize pstate using the steady state probas
        if pstate_prob is None:
            pstate_prob = self.pstate_steadyprob

        # Marginalize hstate using the steady state probas
        if hstate_prob is None:
            hstate_prob = self.steadyprob[pstate_idx]

        if pstate is None and hstate is None:
            # Marginalize both pstate and hstate
            w = (pstate_prob[:, np.newaxis] * hstate_prob).flatten()

            if self.emission_name_distr[feature] == 'lognormal':
                cdf = lambda x: stats.lognorm.cdf(x, self.emission[feature]['logsigma'].flatten(), 0,
                                                  np.exp(self.emission[feature]['logmu'].flatten()))
            elif self.emission_name_distr[feature] == 'normal':
                cdf = lambda x: stats.norm.cdf(x, self.emission[feature]['mu'].flatten(),
                                               self.emission[feature]['sigma'].flatten())
        elif hstate is None:
            # Marginalize hstate
            w = hstate_prob

            if self.emission_name_distr[feature] == 'lognormal':
                cdf = lambda x: stats.lognorm.cdf(x, self.emission[feature]['logsigma'][pstate_idx, :], 0,
                                                  np.exp(self.emission[feature]['logmu'][pstate_idx, :]))
            elif self.emission_name_distr[feature] == 'normal':
                cdf = lambda x: stats.norm.cdf(x, self.emission[feature]['mu'][pstate_idx, :],
                                               self.emission[feature]['sigma'][pstate_idx, :])
        elif pstate is None:
            # Marginalize pstate
            w = pstate_prob

            if self.emission_name_distr[feature] == 'lognormal':
                cdf = lambda x: stats.lognorm.cdf(x, self.emission[feature]['logsigma'][:, hstate], 0,
                                                  np.exp(self.emission[feature]['logmu'][:, hstate]))
            elif self.emission_name_distr[feature] == 'normal':
                cdf = lambda x: stats.norm.cdf(x, self.emission[feature]['mu'][:, hstate],
                                               self.emission[feature]['sigma'][:, hstate])
        else:
            w = 1

            if self.emission_name_distr[feature] == 'lognormal':
                cdf = lambda x: stats.lognorm.cdf(x, self.emission[feature]['logsigma'][pstate_idx, hstate], 0,
                                                  np.exp(self.emission[feature]['logmu'][pstate_idx, hstate]))
            elif self.emission_name_distr[feature] == 'normal':
                cdf = lambda x: stats.norm.cdf(x, self.emission[feature]['mu'][pstate_idx, hstate],
                                               self.emission[feature]['sigma'][pstate_idx, hstate])

        def fn(x):
            if np.isscalar(x):
                p = np.sum(w * cdf(x))
            else:
                x = np.array(x)
                p = np.zeros(len(x))
                for i, xi in enumerate(x):
                    p[i] = np.sum(w * cdf(xi))
            return p

        return fn

    def params(self, pstates=None):
        if pstates is None:
            pstates = [None] + sorted(set(self.er.values()))  # TODO: self.e caches any unknown value, maybe it shouldn't?

        params = []

        # emission parameters
        for hstate, pstate_label in product(range(self.n_hidden_states), pstates):
            for feature, distr in zip(self.emission_name, self.emission_distr):
                for feature_param in _DISTRIBUTIONS[distr]:
                    params.append(self.emission[feature][feature_param][self.e[pstate_label], hstate])

        # transition parameters, diagonals only assuming 2 state
        for hstate, pstate_label in product(range(self.n_hidden_states), pstates):
            params.append(self.transmat[self.e[pstate_label], self.e[pstate_label], hstate, hstate])

        return np.array(params)
