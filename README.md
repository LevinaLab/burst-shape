# Data-driven burst shape analysis for functional phenotyping of neuronal cultures

corresponding to [Schäfer et al., 2025, bioRxiv: Data-driven burst shape analysis for functional phenotyping of neuronal cultures](https://doi.org/10.1101/2025.09.29.679256)

```
@article{schaefer2025data-driven,
	author = {Sch{\"a}fer, Tim J. and Giannakakis, Emmanouil and Schmidt-Barbo, Paul and Levina, Anna and Vinogradov, Oleg},
	title = {Data-driven burst shape analysis for functional phenotyping of neuronal cultures},
	year = {2025},
	doi = {10.1101/2025.09.29.679256},
	journal = {bioRxiv},
}
```

# Tutorial
`notebooks/tutorial.ipynb` walks you through the basic pipeline step-by-step.

# Online tools
You can also try out the analysis pipeline without installing anything using the following online tools.

## Burst visualization
<a href="https://review-inhibblock-659951261078.europe-west1.run.app" target="_blank">Try burst visualization (10s loading time)!</a>
This is used to visualize all recordings and for adjusting burst detection hyperparameters.

<img src="figures/Figure_Suppl_interactive_tools/inhibblock_burst_review.png" width="500"/>

## Embedding visualization
<a href="https://embedding-inhibblock-659951261078.europe-west1.run.app" target="_blank">Try embedding visualization (10s loading time)!</a>
This is used for visualizing the spectral embedding (of individual burst shapes) and exploring this burst shape space.

<img src="figures/Figure_Suppl_interactive_tools/inhibblock_embedding.png" width="500"/>

## Links for other datasets
<ul style="display: inline-block; text-align: left;">
    <li>Blocked inhibition --- Bicuculline (data: <a href="https://doi.org/10.1101/2024.08.21.608974" target="_blank">Vinogradov et al., 2024</a>)</li>
        <ul>
            <li><a href="https://review-inhibblock-659951261078.europe-west1.run.app" target="_blank">Burst visualization</a></li>
            <li><a href="https://embedding-inhibblock-659951261078.europe-west1.run.app" target="_blank">Embedding visualization</a> </li>
        </ul>
    <li>Kleefstra syndrom (hPSC) (data: <a href="https://doi.org/10.17632/bvt5swtc5h.1" target="_blank">Mossink et al., 2021</a>)
        <ul>
            <li><a href="https://review-mossink-659951261078.europe-west1.run.app" target="_blank">Burst visualization</a></li>
            <li><a href="https://embedding-mossink-659951261078.europe-west1.run.app" target="_blank">Embedding visualization</a> </li>
        </ul>
    </li>
    <li>CACNA1A mutation (data: <a href="https://doi.org/10.1101/2024.03.18.585506" target="_blank">Hommersom et al., 2024</a>)</li>
        <ul>
            <li>Burst visualization (data not public yet)</li>
            <li>Embedding visualization (data not public yet)</li>
        </ul>
    <li>Developing cultures (data: <a href="https://doi.org/10.1186/1471-2202-7-11" target="_blank">Wagenaar et al., 2006</a>)</li>
        <ul>
            <li>Burst visualization (dataset too large)</li>
            <li>Embedding visualization (dataset too large)</li>
        </ul>
</ul>


# Setup

## Installation
The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv with
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or via Homebrew (`brew install uv`).

Then, from the repo root, run
```bash
uv sync
```
This creates a `.venv/` with Python 3.13, installs `burst_shape` editable, and pulls in every PEP 735 dependency group declared in `pyproject.toml` (`web`, `analysis`, `dev`). Activate the venv with `source .venv/bin/activate`, or prepend `uv run` to any command (e.g. `uv run pytest`, `uv run python scripts/...`).

## Pre-commit hooks
To enable the ruff format/lint hooks, run once after `uv sync`:
```bash
uv run pre-commit install
```

## Deployment
See [DEPLOY.md](DEPLOY.md) for how to build the Docker images and push the online tools to Google Cloud Run.