"""
Provides routines for calculating Lyapunov exponents.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import scipy.linalg as linalg


def calculate_FTLEs(dim, A, Nk, Q=None):
    """Calculate FTLEs for given system.

    The exponents are estimated by computing the approximate backward
    Lyapunov vectors (BLVs) using the QR algorithm.

    Parameters
    ----------
    dim : int
        Dimension of the system state-space.

    A : array, shape (dim, dim, T)
        Array containing the system matrices used to compute the cocycle.

    Nk : array, shape (n,)
        Points at which to orthogonalize current estimate for BLVs to the
        eigenspaces U. The maximum number of push-forward steps N used
        is expected to be given as N = Nk[-1].

    Q : array, shape (dim, dim), optional
        If given, an initial set of orthogonal vectors to push-forward.

    Returns
    -------
    Lyaps : array, shape (dim,)
        Array containing the approximate Lyapunov exponents.

    References
    ----------
    Dieci, L., Russell, R. D., and Van Vleck, E. S., "On the Computation of
    Lyapunov Exponents for Continuous Dynamical Systems", SIAM Journal on
    Numerical Analysis 34(1) (1997), 402 - 423,
    doi:10.1137/S0036142993247311
    """

    N = Nk[-1]
    if Q is None:
        Q = np.identity(dim)

    orthsteps = len(Nk)-1
    Rdiag = np.array(np.zeros((dim,orthsteps), dtype=np.float))

    k = 0
    for nn in np.arange(0,N):
        Q = np.matmul(A[:,:,nn],Q)

        if any(nk == nn+1 for nk in Nk):
            Q , Rj = linalg.qr(Q)
            Rj = np.diag(Rj)
            l = np.where(Rj<0)
            ind = l[0]
            for ii in ind:
                Q[:,ii] = -1*Q[:,ii]
            Rj = abs(Rj)
            Rdiag[:,k] = Rj
            k += 1

    Lambda = np.sum(np.log(Rdiag),axis=1)/N
    Lyaps = Lambda
    order2 = np.argsort(Lyaps)
    order2 = order2[::-1]
    Lyaps = Lyaps[order2]

    return Lyaps
