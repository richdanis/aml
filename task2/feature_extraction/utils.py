import numpy as np

def hfd(a, k_max):

    """
    Compute Higuchi Fractal Dimension of a time series.
    Vectorised version of the eponymous [PYEEG]_ function.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appears to have implemented an erroneous formulae.
        [HIG88]_ defines the normalisation factor as:
        .. math::
            \frac{N-1}{[\frac{N-m}{k} ]\dot{} k}
        [PYEEG]_ implementation uses:
        .. math::
            \frac{N-1}{[\frac{N-m}{k}]}
        The latter does *not* give the expected fractal dimension of approximately `1.50` for brownian motion (see example bellow).
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param k_max: the maximal value of k
    :type k_max: int
    :return: Higuchi's fractal dimension; a scalar
    :rtype: float
    Example from [HIG88]_. This should produce a result close to `1.50`:
    """

    L = []
    x = []
    N = a.size

    # TODO this could be used to pregenerate k and m idxs ... but memory pblem?
    # km_idxs = np.triu_indices(k_max - 1)
    # km_idxs = k_max - np.flipud(np.column_stack(km_idxs)) -1
    # km_idxs[:,1] -= 1
    #

    for k in range(1,k_max):
        Lk = 0
        for m in range(0,k):
            # we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)

            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk

        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])

    (p, r1, r2, s)=np.linalg.lstsq(x, L, rcond=-1)

    return p[0]


