"""
Provides routines for calculating covariant Lyapunov vectors.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import scipy.linalg as linalg

import clustering_dynamics.utils as cu


def _calculate_clvs_svd_alg21(dim, A, N, M, nCLVs=None):
    """Calculate CLVs using SVD-based algorithm.

    Parameters
    ----------
    dim : int
        Dimension of the system state-space.

    A : array, shape (dim, dim, T)
        Array containing the system matrices used to compute the cocycle.
        The number of time-steps T must satisfy T >= max(M, N).

    N : int
        Number of push-forward steps used to compute the CLVs.

    M : int
        Number of push-forward steps used to compute eigenspaces of the
        far-future operator.

    nCLVs : int, optional
        Number of CLVs to compute. If not given, all CLVs are computed.

    Returns
    -------
    CLV : array, shape (dim, nCLVs)
        Array containing the approximate CLVs at time N as columns.

    References
    ----------
    Froyland, G. et al, "Computing covariant Lyapunov vectors, Oseledets
    vectors, and dichotomy projectors: A comparative numerical study",
    Physica D 247 (2013), 18 - 39, algorithm 2.1,
    doi:10.1016/j.physd.2012.12.005
    """

    if nCLVs is None:
        nCLVs = dim

    Psi = np.identity(dim)
    i = 0
    while i < M:
        Psi = np.matmul(A[:,:,i],Psi)
        Psi = Psi/linalg.norm(Psi)
        i += 1

    if nCLVs == dim:
        _, s, v = linalg.svd(Psi)
    else:
        _, s, v = cu.calculate_truncated_svd(Psi, nCLVs)

    v = v.T.conj()
    order = np.argsort(s)
    order = order[::-1]
    v = v[:,order]

    CLV = v / linalg.norm(v,axis=0)

    for nn in np.arange(N):

        CLV = np.matmul(A[:,:,nn],CLV)
        CLV = CLV/linalg.norm(CLV,axis=0)

    return CLV


def _calculate_clvs_svd_alg22(dim, A, Nk, M, nCLVs=None):
    """Calculate CLVs using improved SVD algorithm.

    Parameters
    ----------
    dim : int
        Dimension of the system state-space.

    A : array, shape (dim, dim, T)
        Array containing the system matrices used to compute the cocycle.
        The number of time-steps T must satisfy T >= max(M, Nk[-1]).

    Nk : array, shape (n,)
        Points at which to orthogonalize current estimate for CLVs to the
        eigenspaces U. The maximum number of push-forward steps N used
        for computing the CLVs is expected to be given as N = Nk[-1].

    M : int
        Number of push-forward steps used to compute eigenspaces of the
        far-future operator.

    nCLVs : int, optional
        Number of CLVs to compute. If not given, all CLVs are computed.

    Returns
    -------
    CLV : array, shape (dim, nCLVs)
        Array containing the approximate CLVs at time N as columns.

    References
    ----------
    Froyland, G. et al, "Computing covariant Lyapunov vectors, Oseledets
    vectors, and dichotomy projectors: A comparative numerical study",
    Physica D 247 (2013), 18 - 39, algorithm 2.2,
    doi:10.1016/j.physd.2012.12.005
    """

    if nCLVs is None:
        nCLVs = dim

    k = 0
    U = np.array(np.zeros((dim,nCLVs,len(Nk)-1), dtype=np.float))
    N = Nk[-1]

    for nn in Nk:
        Psi = np.identity(dim)
        i = 0
        while i < M:
            Psi = np.matmul(A[:,:,nn+i],Psi)
            Psi = Psi/linalg.norm(Psi)
            i += 1

        if nCLVs == dim:
            _, s, v = linalg.svd(Psi)
        else:
            _, s, v = cu.calculate_truncated_svd(Psi, nCLVs)

        v = v.T.conj()
        order = np.argsort(s)
        order = order[::-1]
        v = v[:,order]
        if nn == 0:
            CLV = v/linalg.norm(v,axis=0)
        else:
            U[:,:,k] = v/linalg.norm(v,axis=0)
            k +=1

    k = 0
    for nn in np.arange(N):
        CLV = np.matmul(A[:,:,nn],CLV)
        CLV = CLV/linalg.norm(CLV,axis=0)
        if any(nk == nn+1 for nk in Nk):
            for jj in np.arange(0,nCLVs):
                for ii in np.arange(0,jj):
                    CLV[:,jj] = CLV[:,jj]-np.dot(CLV[:,jj],U[:,ii,k])*U[:,ii,k]
                    CLV[:,jj] = CLV[:,jj]/linalg.norm(CLV[:,jj])
            k += 1

    return CLV


def calculate_CLV_numerically(dim, A, Nk, M, nCLVs=None, version='2.2'):
    """Calculate CLV numerically from cocycle.

    Parameters
    ----------
    dim : int
        Dimension of the system state-space.

    A : array, shape (dim, dim, T)
        Array containing the system matrices used to compute the cocycle.
        The number of time-steps T must satisfy T >= max(M, Nk[-1]).

    Nk : array, shape (n,)
        Points at which to orthogonalize current estimate for CLVs to the
        eigenspaces U. The maximum number of push-forward steps N used
        for computing the CLVs is expected to be given as N = Nk[-1].

    M : int
        Number of push-forward steps used to compute eigenspaces of the
        far-future operator.

    nCLVs : int, optional
        Number of CLVs to compute. If not given, all CLVs are computed.

    version : '2.1' | '2.2', default: '2.2'
        Version of the SVD-based algorithm to use.

    Returns
    -------
    CLV : array, shape (dim, nCLVs)
        Array containing the approximate CLVs at time N as columns.
    """

    if version == '2.1':
        N = np.max(Nk)
        return _calculate_clvs_svd_alg21(dim, A, N, M, nCLVs=nCLVs)

    if version == '2.2':
        return _calculate_clvs_svd_alg22(dim, A, Nk, M, nCLVs=nCLVs)

    raise ValueError("Invalid algorithm version '%r'" % version)
