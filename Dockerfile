ARG DATASET=inhibblock
ARG APPLICATION=review

# uv-on-python image: smaller than python:3.x-slim + pip, and supports `uv sync`.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

# Re-declare the global ARG inside the stage so it's available for ENV substitution.
ARG DATASET

WORKDIR /app

# Install only the `web` dependency group (and the project itself).
# pyproject.toml + uv.lock first → cache layer; src copied after.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv sync --frozen --no-default-groups --group web

# Put the venv on PATH so gunicorn/python resolve from it.
ENV PATH="/app/.venv/bin:${PATH}"

ENV RESULTS_FOLDER=/app/results \
    DEBUG=False \
    RESAMPLE=True \
    DATASET=${DATASET}

FROM base AS inhibblock
ARG PATH_TO_DATA=results/burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85
ENV DATASET=inhibblock

FROM base AS kapucu
ARG PATH_TO_DATA=results/burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150
ENV DATASET=kapucu

FROM base AS hommersom_test
ARG PATH_TO_DATA=results/burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6
ENV DATASET=hommersom_test

FROM base AS hommersom_binary
ARG PATH_TO_DATA=results/burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_21
ENV DATASET=hommersom_binary

FROM base AS hommersom
ARG PATH_TO_DATA=results/burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30/
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_55
ENV DATASET=hommersom

FROM base AS mossink
ARG PATH_TO_DATA=results/burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30/
ARG PATH_TO_EMBEDDING=${PATH_TO_DATA}/spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85
ENV DATASET=mossink

FROM ${DATASET} AS intermediate
COPY ${PATH_TO_DATA}/df_cultures.pkl /app/${PATH_TO_DATA}/df_cultures.pkl

FROM intermediate AS review
# application-specific settings
COPY scripts/2_interactive_tools/002x_review_burst_detection.py /app/main.py

FROM intermediate AS embedding
COPY scripts/2_interactive_tools/011_explore_embeddings.py /app/main.py
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
