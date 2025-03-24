ARG DATASET=inhibblock
ARG APPLICATION=review

FROM python:3.10 AS base

# Set the working directory
WORKDIR /app

# Copy files to the container
# COPY . /app
COPY src /app/src
# COPY scripts/011_interactive_tsne.py /app/interactive_tsne.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for consistent paths (optional)
ENV RESULTS_FOLDER=/app/results
ENV DEBUG=False
ENV DATASET=${DATASET}

FROM base as inhibblock
ARG PATH_TO_DATA=results/burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85
ENV DATASET=inhibblock

FROM base as kapucu
ARG PATH_TO_DATA=results/burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150
ENV DATASET=kapucu

FROM base as hommersom
ARG PATH_TO_DATA=results/burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6
ENV DATASET=hommersom

FROM ${DATASET} AS intermediate
COPY ${PATH_TO_DATA}/df_cultures.pkl /app/${PATH_TO_DATA}/df_cultures.pkl

FROM intermediate AS review
# application-specific settings
COPY scripts/002x_review_burst_detection.py /app/main.py

FROM intermediate AS embedding
COPY scripts/011_interactive_tsne.py /app/main.py
COPY ${PATH_TO_DATA}/df_bursts.pkl /app/${PATH_TO_DATA}/df_bursts.pkl
COPY ${PATH_TO_DATA}/pca.npy /app/${PATH_TO_DATA}/pca.npy
COPY ${PATH_TO_DATA}/tsne.npy /app/${PATH_TO_DATA}/tsne.npy
COPY ${PATH_TO_EMBEDDING}/clustering_labels.pkl /app/${PATH_TO_EMBEDDING}/clustering_labels.pkl
COPY ${PATH_TO_EMBEDDING}/spectral_embedding.npy /app/${PATH_TO_EMBEDDING}/spectral_embedding.npy

FROM ${APPLICATION} AS final

# Expose the port Dash runs on
EXPOSE 8080

# Run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:server"]
