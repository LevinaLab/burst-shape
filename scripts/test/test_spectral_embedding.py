import numpy as np
from sklearn.manifold import spectral_embedding

from src.spectral_clustering import SpectralClusteringModified

# Generate a random adjacency matrix (symmetric, non-negative)
np.random.seed(42)
n_samples = 10
affinity_matrix = np.random.rand(n_samples, n_samples)
affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)  # Make it symmetric
np.fill_diagonal(affinity_matrix, 0)  # No self-loops

# Compute spectral clustering
n_clusters = 3
clustering = SpectralClusteringModified(
    n_clusters=n_clusters,
    n_components_max=n_clusters,
    affinity="precomputed",
    assign_labels="kmeans",
)
clustering.compute_maps(affinity_matrix)

# Extract embeddings
embedding_maps = clustering.maps_[:, 1:]
embedding_direct = spectral_embedding(
    affinity_matrix, n_components=n_clusters, random_state=42
)[:, :-1]

# Compare with Euclidean distance
difference = np.linalg.norm(embedding_maps - embedding_direct)

print("Euclidean distance between maps_ and spectral_embedding output:", difference)
