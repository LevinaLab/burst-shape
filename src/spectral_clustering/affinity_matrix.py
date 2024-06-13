import ray
import numpy as np
import scipy.stats
import scipy.sparse
from tqdm import tqdm


def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


@ray.remote
def _compute_connections_wasserstein(i, bursts, n_neighbors):
    distance_matrix_row = np.zeros(bursts.shape[0])
    for j in range(bursts.shape[0]):
        distance_matrix_row[j] = _wasserstein_distance(bursts[i], bursts[j])
    # get indices of the n_neighbors smallest distances
    connections = np.argpartition(distance_matrix_row, n_neighbors + 1)[
        : n_neighbors + 1
    ]
    return i, connections


@ray.remote
def _compute_connections_cosine(i, bursts, n_neighbors):
    # TODO consider subtracting the mean, this would compute pearson correlation
    distance_matrix_row = np.zeros(bursts.shape[0])
    for j in range(bursts.shape[0]):
        distance_matrix_row[j] = scipy.spatial.distance.cosine(bursts[i], bursts[j])
    # get indices of the n_neighbors smallest distances
    connections = np.argpartition(distance_matrix_row, n_neighbors + 1)[
        : n_neighbors + 1
    ]
    return i, connections


def compute_affinity_matrix(bursts, n_jobs=1, metric="wasserstein", n_neighbors=10):
    # see this website for the pattern used here
    # https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html
    match metric:
        case "wasserstein":
            _compute_connections = _compute_connections_wasserstein
        case "cosine":
            _compute_connections = _compute_connections_cosine
        case _:
            raise NotImplementedError(
                f"Metric '{metric}' not implemented, "
                f"options are 'wasserstein' and 'cosine'."
            )
    max_tasks = 3 * n_jobs
    # initialize connectivity matrix as sparse matrix
    ray.init(num_cpus=n_jobs, ignore_reinit_error=True)
    connectivity = scipy.sparse.lil_matrix((bursts.shape[0], bursts.shape[0]))
    result_refs = []
    for i in tqdm(
        range(bursts.shape[0]), desc="Computing connectivity matrix", smoothing=0
    ):
        if len(result_refs) > max_tasks:
            ready_ids, result_refs = ray.wait(result_refs, num_returns=1)
            index, connections = ray.get(ready_ids)[0]
            connectivity[index, connections] = 1
        result_refs.append(_compute_connections.remote(i, bursts, n_neighbors))

    for result_ref in result_refs:
        index, connections = ray.get(result_ref)
        connectivity[index, connections] = 1
    ray.shutdown()

    print("Build affinity matrix from connectivity matrix...")
    # convert to csr
    connectivity = connectivity.tocsr()
    # delete self connections
    connectivity.setdiag(0)
    connectivity.eliminate_zeros()
    affinity_matrix = 0.5 * (connectivity + connectivity.T)
    print("Done.")

    return affinity_matrix
