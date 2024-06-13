# Computation pipeline

## Download data
script: `001_download_wagenaar_data.py`

This will download the data into the `data/raw` folder and extract it into `data/extracted`.

## Extract bursts
script: `002_extract_bursts.py`
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

## Spectral clustering
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

# Evaluation
TODO @Tim describe the scripts for plotting and evaluation

## Plot bursts
script: `004_plot_example_bursts.py`

## Plot clusters
script: `006_plot_clusters.py`

### PCA
script: `007_pca.py`

### t-SNE
script: `008_tsne.py`

## Cross-validation
script: `009_cross_validation.py`
