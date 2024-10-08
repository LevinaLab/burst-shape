import multiprocessing
import numpy as np

n = 4000
size = n * (n - 1) // 2
# initialize shared memory array
# distance_matrix = multiprocessing.Array("d", n * (n - 1) // 2, lock=False)

"""# def _metric_from_index(X, i, j):
def _metric_from_index(i, j):
    k = i * (2 * n - i - 1) // 2 + j - i - 1
    k_alt = i * (2 * n - i - 3) // 2 + j - 1
    assert k == k_alt
    a = -1/2
    b = (2*n-3)/2 + 1
    c = - (1 + k)
    i_est = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    i_est = int(np.floor(i_est - 1e5 * np.finfo(float).eps))
    assert i_est == i
    j_est = k - (i_est * (2 * n - i_est - 1) // 2 - i_est - 1)
    assert j_est == j, (i, j, i_est, j_est)

# parallel computation of the distance matrix
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    pool.starmap(
        _metric_from_index,
        [
            (i, j)
            for i in range(n - 1)
            for j in range(i + 1, n)
        ],
    )
"""

a = -1/2
b = (2*n-3)/2 + 1
def _metric_from_index_k(k):
    c = - (1 + k)
    i = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    i = int(np.floor(i - 1e5 * np.finfo(float).eps))
    j = k - (i * (2 * n - i - 1) // 2 - i - 1)

with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    pool.map(
        _metric_from_index_k,
        range(size),
    )