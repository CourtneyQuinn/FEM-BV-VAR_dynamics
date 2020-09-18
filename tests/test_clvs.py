"""
Provides unit tests for calculation of CLVs.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np

from clustering_dynamics.dynamics import calculate_CLV_numerically


def test_simple_svd_algorithm_with_constant_jacobian():
    """Test simple SVD algorithm gives correct result with constant Jacobian."""

    tolerance = 1e-7

    delta_t = 1.0
    exact_clvs = np.array([[1.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 1.0]])
    D = np.diag([np.exp(delta_t), np.exp(-delta_t), np.exp(-3 * delta_t)])

    A = np.matmul(exact_clvs, np.matmul(D, np.linalg.inv(exact_clvs)))

    T = 1000
    At = np.repeat(A[:, :, np.newaxis], T, axis=-1)

    dim = A.shape[0]
    nCLVs = dim

    N = 10
    M = 2 * N

    Nk = np.arange(N + 1)

    version = '2.1'
    clvs = calculate_CLV_numerically(
        dim, At, Nk, M, nCLVs=nCLVs, version=version)

    assert clvs.shape == (dim, nCLVs)
    assert np.allclose(np.linalg.norm(clvs, axis=0), 1.0)

    for i in range(nCLVs):
        numerical_clv = clvs[:, i]
        exact_clv = exact_clvs[:, i]

        assert min(np.linalg.norm(exact_clv - numerical_clv),
                   np.linalg.norm(exact_clv + numerical_clv)) < tolerance


def test_improved_svd_algorithm_with_constant_jacobian():
    """Test improved SVD algorithm gives correct result with constant Jacobian."""

    tolerance = 1e-7

    delta_t = 1.0
    exact_clvs = np.array([[1.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 0.0],
                           [0.0, 1.0 / np.sqrt(3.0), 1.0]])
    D = np.diag([np.exp(delta_t), np.exp(-delta_t), np.exp(-3 * delta_t)])

    A = np.matmul(exact_clvs, np.matmul(D, np.linalg.inv(exact_clvs)))

    T = 1000
    At = np.repeat(A[:, :, np.newaxis], T, axis=-1)

    dim = A.shape[0]
    nCLVs = dim

    N = 10
    M = N

    Nk = np.arange(N + 1)

    version = '2.2'
    clvs = calculate_CLV_numerically(
        dim, At, Nk, M, nCLVs=nCLVs, version=version)

    assert clvs.shape == (dim, nCLVs)
    assert np.allclose(np.linalg.norm(clvs, axis=0), 1.0)

    for i in range(nCLVs):
        numerical_clv = clvs[:, i]
        exact_clv = exact_clvs[:, i]

        assert min(np.linalg.norm(exact_clv - numerical_clv),
                   np.linalg.norm(exact_clv + numerical_clv)) < tolerance


def test_simple_svd_algorithm_with_time_dependent_system():
    """Test simple SVD algorithm gives correct result for time-dependent system."""

    random_state = np.random.default_rng(0)

    tolerance = 1e-5

    d = 8
    lambdas = np.log(np.arange(1, d + 1)[::-1])
    epsilon = 0.1
    R = np.diag(np.exp(lambdas))

    N = 75
    n_window = 0

    Sn = np.zeros((2 * N + 2 + 2 * n_window, d, d))

    idx = 0
    for n in range(-(N + 1 + n_window), N + n_window + 1):

        if n == -1:
            z = random_state.uniform(size=(d - 1),)
            Sn[idx] = np.eye(d) + np.diag(z, k=-1)
        else:
            Z = random_state.uniform(size=(d, d))
            Sn[idx] = np.eye(d) + epsilon * Z

        idx += 1

    An = np.full((Sn.shape[0], d, d), np.NaN)
    for n in range(1, Sn.shape[0]):
        An[n] = np.matmul(Sn[n], np.matmul(R, np.linalg.inv(Sn[n - 1])))
    An = An[1:]

    Sn = Sn.transpose((1, 2, 0))
    An = An.transpose((1, 2, 0))

    Nk = np.arange(N + 1)
    M = 2 * N

    version = '2.1'
    clvs = calculate_CLV_numerically(
        d, An, Nk, M, version=version)

    # note that the CLV corresponding to index n of An (= index n - 1 of
    # Sn) is Sn[n]
    exact_clvs = Sn[..., N] / np.linalg.norm(Sn[..., N], axis=0, keepdims=True)

    n_clvs_to_check = 2
    for i in range(n_clvs_to_check):
        numerical_clv = clvs[:, i]
        exact_clv = exact_clvs[:, i]

        assert min(np.linalg.norm(exact_clv - numerical_clv),
                   np.linalg.norm(exact_clv + numerical_clv)) < tolerance


def test_improved_svd_algorithm_with_time_dependent_system():
    """Test improved SVD algorithm gives correct result for time-dependent system."""

    random_state = np.random.default_rng(0)

    tolerance = 1e-6

    d = 8
    lambdas = np.log(np.arange(1, d + 1)[::-1])
    epsilon = 0.1
    R = np.diag(np.exp(lambdas))

    N = 100
    n_window = 0

    Sn = np.zeros((2 * N + 2 + 2 * n_window, d, d))

    idx = 0
    for n in range(-(N + 1 + n_window), N + n_window + 1):

        if n == -1:
            z = random_state.uniform(size=(d - 1),)
            Sn[idx] = np.eye(d) + np.diag(z, k=-1)
        else:
            Z = random_state.uniform(size=(d, d))
            Sn[idx] = np.eye(d) + epsilon * Z

        idx += 1

    An = np.full((Sn.shape[0], d, d), np.NaN)
    for n in range(1, Sn.shape[0]):
        An[n] = np.matmul(Sn[n], np.matmul(R, np.linalg.inv(Sn[n - 1])))
    An = An[1:]

    Sn = Sn.transpose((1, 2, 0))
    An = An.transpose((1, 2, 0))

    Nk = np.arange(N + 1)
    M = N

    version = '2.2'
    clvs = calculate_CLV_numerically(
        d, An, Nk, M, version=version)

    # note that the CLV corresponding to index n of An (= index n - 1 of
    # Sn) is Sn[n]
    exact_clvs = Sn[..., N] / np.linalg.norm(Sn[..., N], axis=0, keepdims=True)

    n_clvs_to_check = 2
    for i in range(n_clvs_to_check):
        numerical_clv = clvs[:, i]
        exact_clv = exact_clvs[:, i]

        assert min(np.linalg.norm(exact_clv - numerical_clv),
                   np.linalg.norm(exact_clv + numerical_clv)) < tolerance
