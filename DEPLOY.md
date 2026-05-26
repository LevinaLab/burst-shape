# Deploying the online tools

The two Dash apps under `scripts/2_interactive_tools/` (`002x_review_burst_detection.py` and `011_explore_embeddings.py`) are packaged into Docker images by the multi-stage `Dockerfile` and deployed to Google Cloud Run. The `makefile` provides the build/push commands.

## Dependency groups

The project's dependencies are split into PEP 735 groups so the Docker images can stay small:

| Group | What's in it | When you need it |
|---|---|---|
| (core, always installed) | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `met-brewer` | Always — imported throughout `burst_shape` |
| `web` | `dash`, `dash-extensions`, `flask`, `gunicorn`, `plotly-resampler`, `werkzeug` | The two interactive Dash apps |
| `analysis` | `scikit-learn`, `numba`, `ray`, `xgboost`, `shap`, `statsmodels`, `baycomp`, `tqdm` | Preprocessing, embedding, ML scripts |
| `dev` | `pytest`, `ruff`, `pre-commit` | Development only |

Locally, `uv sync` installs all four groups. The Docker image runs `uv sync --frozen --no-default-groups --group web`, which installs only the core + `web` set — about 56 packages instead of 93.

## Dockerfile structure

The `Dockerfile` is a parametric multi-stage build:

1. **`base`** — `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`, copies `pyproject.toml` + `uv.lock` + `src/`, then `uv sync --no-default-groups --group web`.
2. **Per-dataset stages** (`inhibblock`, `kapucu`, `hommersom`, `hommersom_test`, `hommersom_binary`, `mossink`) — set `DATASET` and the per-dataset `PATH_TO_DATA` / `PATH_TO_EMBEDDING` build args.
3. **`intermediate`** — copies `df_cultures.pkl` for the selected dataset.
4. **`review`** — copies the burst-detection-review script as `main.py`.
5. **`embedding`** — copies the embedding-exploration script as `main.py`, plus `df_bursts.pkl`, `pca.npy`, `tsne.npy`, `clustering_labels.pkl`, `spectral_embedding.npy`.
6. **`final`** — picks `review` or `embedding` based on `APPLICATION`, exposes port 8080, runs `gunicorn -b 0.0.0.0:8080 main:server`.

The relevant pickle/numpy artifacts must exist on disk under `results/burst_dataset_<dataset>_.../` before building — they get baked into the image.

## Prerequisites

- [Docker](https://www.docker.com/) running locally. On macOS we recommend [OrbStack](https://orbstack.dev/) (`brew install --cask orbstack`) — much lower memory footprint than Docker Desktop, drop-in `docker` CLI.
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) authenticated against the GCP project `burstier-review`.

## Makefile variables

Override on the command line, e.g. `make docker-build APPLICATION=embedding DATASET=mossink`:

| Variable | Default | Meaning |
|---|---|---|
| `APPLICATION` | `review` | Which Dash app: `review` or `embedding` |
| `DATASET` | `inhibblock` | Which dataset to bake in: `inhibblock`, `kapucu`, `hommersom`, `hommersom_test`, `hommersom_binary`, `mossink` |
| `TAG` | `latest` | Docker tag |
| `PLATFORM` | `linux/amd64` | Image platform. Keep `linux/amd64` for Cloud Run (built-via-emulation on Apple Silicon). Override to your native arch for fast local runs. |
| `PROJECT_ID` | `burstier-review` | GCP project |
| `REGION` | `europe-west1` | Cloud Run region |
| `REPO` | `eu.gcr.io` | Registry host (see "GCR deprecation" below) |

## Workflow

End-to-end deploy in one go:
```bash
make docker-deploy APPLICATION=review DATASET=inhibblock
```

Or run the steps individually:
```bash
make docker-build  APPLICATION=review DATASET=inhibblock   # build linux/amd64 image
make docker-run    APPLICATION=review DATASET=inhibblock   # smoke-test locally on :8080
make docker-tag    APPLICATION=review DATASET=inhibblock   # tag for the registry
make gcloud-auth                                           # one-off per machine
make docker-push   APPLICATION=review DATASET=inhibblock   # push to the registry
```

After pushing, deploy the new image to Cloud Run from the GCP console, or:
```bash
gcloud run deploy review-inhibblock \
    --image eu.gcr.io/burstier-review/review-inhibblock:latest \
    --region europe-west1 --platform managed --allow-unauthenticated
```

## Notes

- **Apple Silicon → Cloud Run.** Cloud Run runs `linux/amd64` by default, so the makefile builds with `--platform linux/amd64`. On an arm64 Mac this uses Docker's amd64 emulation, which is several times slower than a native build. Acceptable for an occasional deploy. If you ever switch the Cloud Run service to arm64 (gen2 supports it), set `PLATFORM=linux/arm64` for native-speed builds.
- **gcloud-auth.** Running `gcloud auth login` interactively from `make` is awkward. Do it once per machine, then use `make docker-deploy-no-auth` (which skips the auth step) for subsequent deploys.
- **GCR deprecation.** The makefile still pushes to `eu.gcr.io` (Container Registry), which Google is shutting down in favor of Artifact Registry (`europe-west1-docker.pkg.dev/<PROJECT_ID>/<REPO>/...`). The commented-out lines in `makefile` show the new format. When you migrate, also update `gcloud-auth` to call `gcloud auth configure-docker europe-west1-docker.pkg.dev` (already there) instead of relying on the legacy gcr.io credential helper.