# Setup

## Installation
[Optional] Create a conda environment named `burst-clustering` and `python=3.11` with
```bash
conda create -n burst-clustering python=3.11
```
Activate the environment with
```bash
conda activate burst-clustering
```

Install `src` module with
```bash
pip install -e .
```
which will run `setup.py`, making the `src` module available.

[Optional] If this fails to install the dependencies, you can install them manually with
```bash
pip install -r requirements.txt
```

# Computation pipeline

## Download data
First, download the data and ensure that it is functioning.

Existing scripts: `001_download_wagenaar_data.py`
`001b_download_kapucu_data.py`
`001c_test_load_Hommersom.py`
`001d_preload_inhibblock.py`
`001e_test_load_mossink.py`

If you want to use your own data, create a similar script.

## Extract bursts
Identifying the bursts relies on two steps:
1. Extract the bursts
   (scripts: 
  `002_extract_bursts.py`
  `002b_extract_bursts_kapucu.py`
  `002_extract_bursts_hommersom.py`
  `002_extract_bursts_inhibblock.py`
  `002_extract_bursts_.py`
   )
2. Burst visualization (script: `002x_review_burst_detection`)
Iterate these steps until you have a satisfying burst detection.

### Example parameters
```python
params_burst_extraction = {
    "maxISIstart": 5,
    "maxISIb": 5,
    "minBdur": 40,
    "minIBI": 40,
    "minSburst": 50,
    "bin_size": None,
    "n_bins": 50,
    "extend_left": 0,
    "extend_right": 0,
    "burst_length_threshold": None,
    "pad_right": False,
}
```

### Identification
TODO @Oleg

### Extending and padding
Bursts time can be extended with `extend_left` and `extend_right` (default `0`).

### Binning
Two options for binning (with major implications):
- Fixed bin size with `bin_size` (default `None`):
    - Burst are represented in real time
    - results in a variable number of bins per burst and requires padding with `pad_right` (default `False`)
- Fixed number of bins`n_bins` (default `None`)
    - All burst have the same number of bins (features)
    - results in a different bin size for each burst

### Selection
Discard long bursts with `burst_length_threshold` (default `None`)

### Normalization
TODO @Tim

## Embedding and Clustering
The core steps for computation are
1. Split dataset for cross-validation: `003_split_cross_validation.py`
2. Compute spectral clustering: `005a_spectral_clustering.py` 
   (implicitly computes sparse affinity matrix (KNN-graph) and spectral embedding)
3. Compute agglomerative_clustering: `005b_agglomerative_clustering.py` (implicitly computes full distance matrix)
4. (optional) Compute t-SNE embedding `010_compute_tsne.py`
5. (TODO: remove because redundant with spectral clustering) Compute spectral embedding: `010a_spectral_embedding.py`

The core steps for evaluation:
1. Cross-validation of clustering: `020_cross-validation.py`
2. Visualize results interactively: `011_interactive_tsne.py`

### Spectral clustering
script: `005_spectral_clustering.py`

This is the most time-consuming step in the pipeline.
Consider setting `n_jobs` to your number of CPU cores to speed up the computation.
### Affinity matrix
Either use
- built-in method from sklearn (Euclidian distance)-> specifying `n_neighbors'` (default `10`)
- or use custom method (e.g. Wasserstein distance, Cosine distance)
-> additionally specifiy `affinity="precomputed"` and `metric` (e.g. `"wasserstein"`)

### Eigenvalue decomposition
uses a modified version of sklearn's spectral clustering algorithm,
this implicitly saves the eigenvectors up to `n_components_max` (default 30).
### Label assignment
Available methods:
- `assign_labels="kmeans"`
- `assign_labels="cluster_qr"` (default)

This requires setting `n_clusters_min` and `n_clusters_max` (default `2` and `30`)
### Example parameters
```python
clustering_params = {
    "n_components_max": 30,
    "affinity": "precomputed",
    "metric": "wasserstein",
    "n_neighbors": 10,
    "random_state": 0,
}
label_params = {
    "n_clusters_min": 2,
    "n_clusters_max": 30,
    "assign_labels": "cluster_qr",
    "random_state": 0,
}
```

## Cross-validation
script: `002_split_cross_validation.py`

This script splits the data into training and test set and saves the indices for each split to `df_bursts_cv_{cv_params}`.
All subsequent scripts will use will run cross-validation if `cv_params` is provided in these scripts.

### Example parameters
```python
cv_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}
```