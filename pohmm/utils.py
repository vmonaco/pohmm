import numbers
import numpy as np
from itertools import product


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def logsumexp(arr, axis=0):
    """
    Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


def normalize(A, axis=None, inplace=False):
    """
    Normalize the input array so that it sums to 1.

    Parameters
    ----------
    A: array, shape (n_samples, n_features)
        Non-normalized input data.
    axis: int
        Dimension along which normalization is performed.

    Returns
    -------
    normalized_A: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis
    """
    if not inplace:
        A = A.copy()

    A += np.finfo(float).eps
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    A /= Asum
    return A


def ph2full(ptrans, htrans):
    """
    Convert a p-state transition matrix and h-state matrices to the full transation matrix

    The full transmat hase N=n_pstates*n_hstates states
    """
    n_pstates = len(ptrans)
    n_hstates = len(htrans[0, 0])
    N = n_pstates * n_hstates
    trans = np.zeros((N, N))
    for pidx in range(n_pstates):
        for hidx in range(n_hstates):
            trans[pidx * n_hstates + hidx] = (ptrans[pidx, :, np.newaxis] * htrans[pidx, :, hidx]).flatten()

    return trans


def full2ph(trans, n_pstates):
    """
    Convert a full transmat to the respective p-state and h-state transmats
    """
    n_hstates = len(trans) / n_pstates

    htrans = np.zeros((n_pstates, n_pstates, n_hstates, n_hstates))
    for pidx1, pidx2 in product(range(n_pstates), range(n_pstates)):
        idx1 = pidx1 * n_hstates
        idx2 = pidx2 * n_hstates
        htrans[pidx1, pidx2] = trans[idx1:idx1 + n_hstates, idx2:idx2 + n_hstates]

    ptrans = normalize(htrans.sum(axis=-1).sum(axis=-1), axis=1)
    htrans = normalize(htrans, axis=3)

    return ptrans, htrans


def gen_stochastic_matrix(size, random_state=None):
    """
    Generate a unfiformly-random stochastic array or matrix
    """
    if not type(size) is tuple:
        size = (1, size)

    assert len(size) == 2

    n = random_state.uniform(size=(size[0], size[1] - 1))
    n = np.concatenate([np.zeros((size[0], 1)), n, np.ones((size[0], 1))], axis=1)

    A = np.diff(np.sort(n))
    return A.squeeze()


def steadystate(A, max_iter=100):
    """
    Empirically determine the steady state probabilities from a stochastic matrix
    """
    P = np.linalg.matrix_power(A, max_iter)

    # Determine the unique rows in A
    v = []
    for i in range(len(P)):
        if not np.any([np.allclose(P[i], vi, ) for vi in v]):
            v.append(P[i])

    return normalize(np.sum(v, axis=0))


def unique_rows(a):
    """
    Get the unique row values in matrix a
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def expected_lognormal(logsigma, logmu):
    return np.exp(logmu + (logsigma ** 2) / 2)


def expected_normal(logmu, logsigma):
    return logmu
