import os

import dash
import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from flask import Flask

from src.folders import get_fig_folder
from src.persistence import (
    load_clustering_labels,
    load_df_bursts,
    load_df_cultures,
    load_pca,
    load_spectral_embedding,
    load_tsne,
)
from src.persistence.agglomerative_clustering import get_agglomerative_labels
from src.persistence.spike_times import (
    get_hommersom_spike_times,
    get_inhibblock_spike_times,
    get_kapucu_spike_times,
)
from src.plot import get_cluster_colors, get_group_colors

if "DEBUG" in os.environ:
    debug = os.environ["DEBUG"] == "True"
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to debug mode")
    debug = True


###############################################################################
#                           Parameters                                        #
###############################################################################
if "DATASET" in os.environ:
    match os.environ["DATASET"]:
        case "wagenaar":
            burst_extraction_params = "burst_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
        case "kapucu":
            burst_extraction_params = "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
        case "hommersom":
            burst_extraction_params = "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        case "inhibblock":
            burst_extraction_params = "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        case "mossink":
            burst_extraction_params = "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
        case _:
            raise NotImplementedError(
                f"Unknown environment variable DATASET: {os.environ['DATASET']}"
            )
else:
    burst_extraction_params = (
        # "burst_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
        # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
        # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
        # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
        # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
        # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
        # "burst_dataset_hommersom_minIBI_50_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_hommersom_minIBI_50_n_bins_50_normalization_integral_min_length_30_min_firing_rate_1585"
        # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    )
citation = "the relevant literature"
doi_link = None
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
    citation = "Kapucu et al. (2022)"
    doi_link = "https://doi.org/10.1038/s41597-022-01242-4"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
    citation = "Hommersom et al. (2024)"
    doi_link = "https://doi.org/10.1101/2024.03.18.585506"
    clustering_params = "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    citation = "Vinogradov et al. (2024)"
    doi_link = "https://doi.org/10.1101/2024.08.21.608974"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
    citation = "Mossink et al. (2021)"
    doi_link = "https://doi.org/10.17632/bvt5swtc5h.1"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
    )
else:
    dataset = "wagenaar"
    citation = "Wagenaar et al. (2006)"
    doi_link = "https://doi.org/10.1186/1471-2202-7-11"
    clustering_params = (
        "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
    )
print(f"Detected dataset: {dataset}")

# clustering_params = (
# "agglomerating_clustering_linkage_complete"
# "agglomerating_clustering_linkage_ward"
# "agglomerating_clustering_linkage_average"
# "agglomerating_clustering_linkage_single"
# "spectral_affinity_precomputed_metric_wasserstein"
# "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_150"
# "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_60"
# "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_6"
# "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_85"
# )
clustering_type = clustering_params.split("_")[0]
labels_params = None  #  needed for spectral clustering if not default "labels"
n_clusters = np.arange(2, 21, 1)

# initial settings
n_clusters_init = 4  # initial number of clusters
color_by_init = "cluster"  # initial color by
marker_size_init = 3  # initial marker size
embedding_type_init = ["tsne", "pca", "spectral"][2]

###############################################################################
#                           Prepare data                                      #
###############################################################################
# get df_bursts and labels
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
match clustering_type:
    case "agglomerating":
        # load labels
        for n_clusters_ in n_clusters:
            labels = get_agglomerative_labels(
                n_clusters_, burst_extraction_params, clustering_params
            )
            df_bursts[f"cluster_{n_clusters_}"] = [
                f"Cluster {label}" for label in labels
            ]
        # df_bursts["cluster"] = [f"Cluster {label}" for label in labels]
    case "spectral":
        if labels_params is None:
            labels_params = "labels"
        clustering = load_clustering_labels(
            clustering_params,
            burst_extraction_params,
            labels_params,
            None,
            i_split=None,
        )
        for n_clusters_ in n_clusters:
            df_bursts[f"cluster_{n_clusters_}"] = [
                f"Cluster {label + 1}" for label in clustering.labels_[n_clusters_]
            ]


def _load_embedding(embedding_type_):
    match embedding_type_:
        case "tsne":
            embedding = load_tsne(burst_extraction_params)
        case "pca":
            embedding = load_pca(burst_extraction_params)
        case "spectral":
            embedding = load_spectral_embedding(
                burst_extraction_params, clustering_params
            )
    # df_bursts["embedding_x"] = embedding[:, 0]
    # df_bursts["embedding_y"] = embedding[:, 1]
    return embedding


for embedding_type in ["tsne", "pca", "spectral"]:
    embedding = _load_embedding(embedding_type)
    for i in range(1, 3):
        df_bursts[f"{embedding_type}_{i}"] = embedding[:, i - 1]
del embedding, embedding_type


# prepare for plotly
df_bursts.reset_index(inplace=True)

df_bursts["firing_rate"] = df_bursts["integral"] / 50  # df_bursts["time_orig"]
match dataset:
    case "wagenaar":
        df_bursts["batch_culture"] = (
            df_bursts["batch"].astype(str) + "-" + df_bursts["culture"].astype(str)
        )
    case "kapucu":
        df_bursts["batch"] = (
            df_bursts["culture_type"].astype(str)
            + "-"
            + df_bursts["mea_number"].astype(str)
        )
        df_bursts["batch_culture"] = (
            df_bursts["batch"].astype(str) + "-" + df_bursts["well_id"].astype(str)
        )
    case "hommersom":
        df_bursts["batch_culture"] = (
            df_bursts["batch"].astype(str) + "-" + df_bursts["clone"].astype(str)
        )
    case "inhibblock":
        df_bursts["batch_culture"] = (
            df_bursts["drug_label"].astype(str) + "-" + df_bursts["div"].astype(str)
        )
    case "mossink":
        df_bursts["batch_culture"] = (
            df_bursts["group"].astype(str) + "-" + df_bursts["subject_id"].astype(str)
        )
    case _:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

# confirm burst_matrix and df_bursts["burst"] are the same
# for i in range(len(burst_matrix)):
#     assert np.allclose(burst_matrix[i], df_bursts["burst"].iloc[i])
# print("Burst matrix and df_bursts['burst'] are the same")
print("Starting Dash app...")

###############################################################################
#                       Interactive t-SNE Plot with Dash                      #
###############################################################################
# Dash App
server = Flask(__name__)  # Create a Flask app
app = Dash(__name__, server=server)  # Attach Dash to Flask

# Create a color map based on cluster labels
# color_map = cluster_colors  # px.colors.qualitative.Plotly[:n_clusters]


ID_EMBEDDING = "tsne-plot"
ID_EMBEDDING_TYPE = "embedding-type"
ID_COLOR_BY = "color-by"
ID_N_CLUSTERS = "n-clusters-slider"
ID_DOWNLOAD_FORMAT = "download-format"
ID_MARKER_SIZE = "marker-size-slider"
ID_BURST_SHAPE = "timeseries-plot"
ID_BURST_RASTER = "raster-plot"
ID_FIRING_RATE = "firing-rate"

# Create the initial t-SNE plot
# tsne_fig = update_tsne_plot(df_bursts, n_clusters_init)
# tsne_plot = dcc.Graph(id=ID_EMBEDDING, figure=tsne_fig, style={"flex": "1"})
tsne_plot = dcc.Graph(id=ID_EMBEDDING, style={"flex": "1"})

# Define layout of the Dash app
app.layout = html.Div(
    [
        html.P(
            [
                "If you use the data presented here, please cite ",
                citation
                if doi_link is None
                else html.A(
                    citation,
                    href=doi_link,
                    target="_blank",
                    style={"textDecoration": "none", "color": "blue"},
                ),
                ".",
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        # drop-down menu for selecting what to plot
                        dcc.Dropdown(
                            id=ID_EMBEDDING_TYPE,
                            options=[
                                {"label": "Spectral", "value": "spectral"},
                                {"label": "t-SNE", "value": "tsne"},
                                {"label": "PCA", "value": "pca"},
                            ],
                            value=embedding_type_init,
                        ),
                        # drop-down menu for selecting which column to color by
                        dcc.Dropdown(
                            id=ID_COLOR_BY,
                            options=[
                                {"label": "Cluster", "value": "cluster"},
                                {"label": "Batch", "value": "batch"},
                                {"label": "Culture", "value": "batch_culture"},
                                {"label": "Duration", "value": "time_orig"},
                                {"label": "Firing rate", "value": "firing_rate"},
                                {"label": "Day", "value": "day"},
                            ],
                            value=color_by_init,
                        ),
                        # slider for selecting the number of clusters with label "slide to select number of clusters"
                        dcc.Slider(
                            id=ID_N_CLUSTERS,
                            min=n_clusters[0],
                            max=n_clusters[-1],
                            step=1,
                            value=n_clusters_init,
                            tooltip={"placement": "bottom", "always_visible": True},
                            vertical=True,
                        ),
                        html.Button("Save to figures/", id="download-button"),
                        # dcc.Download(id="download-pdf"),
                        dcc.Dropdown(
                            id=ID_DOWNLOAD_FORMAT,
                            options=[
                                {"label": "PDF", "value": "pdf"},
                                {"label": "SVG", "value": "svg"},
                            ],
                            value="svg",
                        ),
                    ],
                    style={
                        "flex": "0 0 100px",
                        "display": "flex",
                        "flex-direction": "column",
                    },
                ),
                html.Div(
                    [
                        # t-SNE plot
                        tsne_plot,
                        dcc.Slider(
                            id=ID_MARKER_SIZE,
                            min=1,
                            max=10,
                            step=1,
                            value=marker_size_init,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"display": "flex", "flex-direction": "column", "flex": "1"},
                ),
                # Time series plot of selected point
                dcc.Graph(id=ID_BURST_SHAPE, style={"flex": "1"}),
                # Raster plot of selected point
                dcc.Graph(id=ID_BURST_RASTER, style={"flex": "1"}),
            ],
            style={"display": "flex", "flex-direction": "row", "height": "50vh"},
        ),
        dcc.Graph(id=ID_FIRING_RATE, style={"flex": "1"}),
    ],
    style={"display": "flex", "flex-direction": "column"},
)


# Create the t-SNE scatter plot
@app.callback(
    Output(ID_EMBEDDING, "figure"),
    [
        Input(ID_N_CLUSTERS, "value"),
        Input(ID_COLOR_BY, "value"),
        Input(ID_EMBEDDING_TYPE, "value"),
        Input(ID_MARKER_SIZE, "value"),
    ],
)
def update_tsne_plot(n_clusters_current, color_by, embedding_type, marker_size):
    global tsne_fig
    color_discrete_sequence = None
    color_discrete_map = None
    color_continuous_scale = None
    category_orders = None
    legend_title = None
    color_log = None

    match color_by:
        case "cluster":
            color = f"cluster_{n_clusters_current}"
            color_discrete_sequence = get_cluster_colors(n_clusters_current)
            category_orders = {
                f"cluster_{n_clusters_current}": [
                    f"Cluster {i}" for i in range(1, n_clusters_current + 1)
                ]
            }
            legend_title = "Cluster"
        case "batch":
            color = "batch"
            match dataset:
                case "inhibblock":
                    color = "drug_label"
                    color_discrete_map = get_group_colors(dataset)
                case "wagenaar":
                    color_discrete_map_load = get_group_colors(dataset)
                    color_discrete_map = {}
                    for key, value in color_discrete_map_load.items():
                        color_discrete_map[str(key)] = value
                case "kapucu":
                    color_discrete_map_load = get_group_colors(dataset)
                    color_discrete_map = {}
                    for key, value in color_discrete_map_load.items():
                        color_discrete_map["-".join(key)] = value
                case "mossink":
                    color = "group"
            df_bursts[color] = df_bursts[color].astype(str)
            color_discrete_sequence = px.colors.qualitative.Set1
            match dataset:
                case "wagenaar":
                    category_orders = {
                        color: sorted(df_bursts["batch"].unique(), key=int)
                    }
                case "kapucu" | "hommersom" | "inhibblock":
                    category_orders = {color: sorted(df_bursts[color].unique())}
                case _:
                    pass
                    # raise NotImplementedError(f"Dataset {dataset} not implemented.")
            legend_title = "Batch"
        case "batch_culture":
            color = "batch_culture"
            df_bursts[color] = df_bursts[color].astype(str)
            color_discrete_sequence = px.colors.qualitative.Set1
            match dataset:
                case "wagenaar":
                    category_orders = {
                        color: sorted(
                            df_bursts["batch_culture"].unique(),
                            key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])),
                        )
                    }
                case "kapucu" | "hommersom" | "inhibblock" | "mossink":
                    category_orders = {
                        color: sorted(df_bursts["batch_culture"].unique())
                    }
                case _:
                    raise NotImplementedError(f"Dataset {dataset} not implemented.")
            legend_title = "Batch-Culture"
        case "day":
            color = "day" if dataset == "wagenaar" else "DIV"
            df_bursts[color] = df_bursts[color].astype(str)
            unique_days = sorted(df_bursts[color].unique(), key=int)
            category_orders = {color: unique_days}
            viridis_colors = plotly.colors.sample_colorscale(
                px.colors.sequential.Viridis,
                [i / len(unique_days) for i in range(len(unique_days))],
            )
            color_discrete_map = {
                str(day): viridis_colors[i] for i, day in enumerate(unique_days)
            }
            legend_title = "Day"
        case "time_orig":
            color = np.log10(pd.to_numeric(df_bursts["time_orig"], errors="coerce"))
            # df_bursts[color] = pd.to_numeric(df_bursts[color], errors='coerce')
            color_continuous_scale = px.colors.sequential.Viridis
            legend_title = "Duration"
            color_log = True
        case "firing_rate":
            color = np.log10(pd.to_numeric(df_bursts["firing_rate"], errors="coerce"))
            color_continuous_scale = px.colors.sequential.Viridis
            legend_title = "Firing rate"
            color_log = True
        case _:
            print(f"Invalid color_by: {color_by}")
            raise ValueError(f"Invalid color_by: {color_by}")

    n_clusters_ = n_clusters_current
    tsne_fig = px.scatter(
        df_bursts,
        x=f"{embedding_type}_1",
        y=f"{embedding_type}_2",
        color=color,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        category_orders=category_orders,
        hover_data={
            f"{embedding_type}_1": False,
            f"{embedding_type}_2": False,
            f"cluster_{n_clusters_}": True,
            # "batch": dataset == "wagenaar",
            # "culture": dataset == "wagenaar",
            # "day": dataset == "wagenaar",
            "start_orig": True,
            "time_orig": True,
        },
        title="t-SNE plot",
        custom_data=[df_bursts.index],
    )
    tsne_fig.update_traces(marker=dict(size=marker_size))

    if color_log is True:
        c_min_max = [color.min(), color.max()]
        tickvals = np.linspace(c_min_max[0], c_min_max[1], 5, endpoint=True)
        tickvalues = [f"{10**v:.0f}" for v in tickvals]
        tsne_fig.update_coloraxes(
            colorbar=dict(
                title=legend_title,
                tickvals=tickvals,
                ticktext=tickvalues,
            )
        )

    tsne_fig.update_layout(
        legend_title=legend_title,
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
        plot_bgcolor="white",
    )
    return tsne_fig


@app.callback(
    # Output("download-pdf", "data"),
    Input("download-button", "n_clicks"),
    Input(ID_EMBEDDING, "figure"),
    Input(ID_DOWNLOAD_FORMAT, "value"),
    prevent_initial_call=True,
)
def download_pdf(n_clicks, figure, format):
    ctx = dash.callback_context

    # Check if the download button was clicked
    if (
        not ctx.triggered
        or ctx.triggered[0]["prop_id"].split(".")[0] != "download-button"
    ):
        raise dash.exceptions.PreventUpdate

    def _cm2px(cm):
        return cm * 37.8

    # Create a BytesIO buffer to hold the PDF data
    fig = go.Figure(figure)
    fig.update_layout(
        title=None,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,  # Remove tick labels
            ticks="",  # Remove ticks
            title="",  # Remove axis title
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""
        ),
        font=dict(family="Helvetica", size=12),
        width=_cm2px(8),
        height=_cm2px(8),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.write_image(os.path.join(get_fig_folder(), f"embedding.{format}"))
    return


# Update time series plot based on the selected point in the t-SNE plot
@app.callback(
    Output(ID_BURST_SHAPE, "figure"),
    [
        Input(ID_EMBEDDING, "clickData"),
        Input(ID_FIRING_RATE, "clickData"),
        Input(ID_N_CLUSTERS, "value"),
    ],
    prevent_initial_call=True,
)
def update_timeseries(tsne_click_data, firing_rate_click_data, n_clusters_current):
    # Determine which plot was clicked
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == ID_FIRING_RATE:
        selected_data = firing_rate_click_data
    else:
        selected_data = tsne_click_data
    if selected_data is None:
        return px.line(pd.DataFrame(), title="Select a point to view time series")

    # Extract index of selected point
    if "customdata" in selected_data["points"][0]:
        point_index = selected_data["points"][0]["customdata"][0]
    else:
        return dash.no_update

    # Create time series plot and title
    # Title should be the same as hovering info in t-SNE plot
    match dataset:
        case "wagenaar":
            title = "Time series for " + ", ".join(
                [
                    f"{key}: {df_bursts.iloc[point_index][key]}"
                    for key in [
                        f"cluster_{n_clusters_current}",
                        "batch",
                        "culture",
                        "day",
                        "start_orig",
                        "time_orig",
                    ]
                ]
            )
        case "kapucu":
            title = "Time series for " + ", ".join(
                [
                    f"{key}: {df_bursts.iloc[point_index][key]}"
                    for key in [
                        f"cluster_{n_clusters_current}",
                        "batch",
                        "culture_type",
                        "mea_number",
                        "well_id",
                        "start_orig",
                        "time_orig",
                    ]
                ]
            )
        case "hommersom":
            title = "Time series for " + ", ".join(
                [
                    f"{key}: {df_bursts.iloc[point_index][key]}"
                    for key in [
                        f"cluster_{n_clusters_current}",
                        "batch",
                        "clone",
                        "well_idx",
                        # "well_id",
                        "start_orig",
                        "time_orig",
                    ]
                ]
            )
        case "inhibblock":
            # print(df_bursts.iloc[point_index])
            title = "Time series for " + ", ".join(
                [
                    f"{key}: {df_bursts.iloc[point_index][key]}"
                    for key in [
                        f"cluster_{n_clusters_current}",
                        "drug_label",
                        "div",
                        "well_idx",
                        "i_burst",
                        # "well_id",
                        "start_orig",
                        "time_orig",
                    ]
                ]
            )
        case "mossink":
            title = "Time series for " + ", ".join(
                [
                    f"{key}: {df_bursts.iloc[point_index][key]}"
                    for key in [
                        f"cluster_{n_clusters_current}",
                        "group",
                        "subject_id",
                        "well_idx",
                        "i_burst",
                        # "well_id",
                        "start_orig",
                        "time_orig",
                    ]
                ]
            )
        case _:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
    timeseries_fig = px.line(
        df_bursts.iloc[point_index]["burst"],
        title=title,
        line_shape="linear",
        line_dash_sequence=["solid"],
    )
    timeseries_fig.update_traces(line=dict(color="black"))
    timeseries_fig.update_layout(
        xaxis_title="Time [ms]",
        yaxis_title="Amplitude [a.u.]",
        plot_bgcolor="white",
    )
    return timeseries_fig


@app.callback(
    [
        Output(ID_BURST_RASTER, "figure"),
        Output(ID_FIRING_RATE, "figure"),
    ],
    [
        Input(ID_EMBEDDING, "clickData"),
        Input(ID_FIRING_RATE, "clickData"),
        Input(ID_N_CLUSTERS, "value"),
    ],
    prevent_initial_call=True,
)
def update_raster(tsne_click_data, firing_rate_click_data, n_clusters_current):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == ID_FIRING_RATE:
        selected_data = firing_rate_click_data
    else:
        selected_data = tsne_click_data
    if selected_data is None:
        return px.line(pd.DataFrame(), title="Select a point to view time series")

    # Extract index of selected point
    if "customdata" in selected_data["points"][0]:
        point_index = selected_data["points"][0]["customdata"][0]
    else:
        return dash.no_update, dash.no_update

    match dataset:
        case "wagenaar":
            batch, culture, day = list(
                df_bursts.iloc[point_index][["batch", "culture", "day"]]
            )
            # print(f"batch: {batch}, culture: {culture}, day: {day}")

            # Load spike data
            st, gid = np.loadtxt(
                "../data/extracted/%s-%s-%s.spk.txt" % (batch, culture, day)
            ).T
            st = st * 1000
            title_firing_rate = (
                f"Firing rate for batch {batch}, culture {culture}, day {day}"
            )
        case "kapucu":
            culture_type, mea_number, well_id, div_day = list(
                df_bursts.iloc[point_index][
                    ["culture_type", "mea_number", "well_id", "DIV"]
                ]
            )
            st, gid = get_kapucu_spike_times(
                df_cultures, (culture_type, mea_number, well_id, div_day)
            )
            df_plot = df_bursts[
                (df_bursts["culture_type"] == culture_type)
                & (df_bursts["mea_number"] == mea_number)
                & (df_bursts["well_id"] == well_id)
                & (df_bursts["DIV"] == div_day)
            ]
            title_firing_rate = (
                f"Firing rate for {culture_type} {mea_number}, {well_id}, DIV {div_day}"
            )
        case "hommersom":
            batch, clone, well_idx = list(
                df_bursts.iloc[point_index][["batch", "clone", "well_idx"]]
            )
            st, gid = get_hommersom_spike_times(df_cultures, (batch, clone, well_idx))
            df_plot = df_bursts[
                (df_bursts["batch"] == batch)
                & (df_bursts["clone"] == clone)
                & (df_bursts["well_idx"] == well_idx)
            ]
            title_firing_rate = f"Firing rate for {batch} {clone}, well {well_idx}"
        case "inhibblock":
            drug_label, div, well_idx = list(
                df_bursts.iloc[point_index][["drug_label", "div", "well_idx"]]
            )
            st, gid = get_inhibblock_spike_times(
                df_cultures, (drug_label, div, well_idx)
            )
            df_plot = df_bursts[
                (df_bursts["drug_label"] == drug_label)
                & (df_bursts["div"] == div)
                & (df_bursts["well_idx"] == well_idx)
            ]
            title_firing_rate = (
                f"Firing rate for {drug_label}, div {div}, well {well_idx}"
            )
        case "mossink":
            group, subject_id, well_idx = list(
                df_bursts.iloc[point_index][["group", "subject_id", "well_idx"]]
            )
            st, gid = get_inhibblock_spike_times(
                df_cultures, (group, subject_id, well_idx)
            )
            df_plot = df_bursts[
                (df_bursts["group"] == group)
                & (df_bursts["subject_id"] == subject_id)
                & (df_bursts["well_idx"] == well_idx)
            ]
            title_firing_rate = (
                f"Firing rate for {group}, subject {subject_id}, well {well_idx}"
            )
        case _:
            raise NotImplementedError(f"Dataset {dataset} not implemented")

    # trace of firing rate
    bin_size = 100  # ms
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size / 1000)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])

    # cut out a window around the burst
    start_orig = df_bursts.iloc[point_index]["start_orig"]
    duration = df_bursts.iloc[point_index]["time_orig"]
    start = start_orig - max(800, duration)
    end = start_orig + max(2000, 2 * duration)
    # if start_orig + duration > end:
    #     end = start_orig + duration + 500
    st, gid = st[(st >= start) & (st <= end)], gid[(st >= start) & (st <= end)]

    # Create raster plot with '|' markers for spikes and no line connecting them
    fig_raster = go.Figure()
    # Raster plot with vertical line markers
    fig_raster.add_trace(
        go.Scatter(
            x=st,
            y=gid,
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=10,
                # color=palette[i_cluster - 1],
                line_width=1,
            ),
            name="Spikes",
        )
    )
    # Vertical reference line for start and start + duration
    fig_raster.add_vline(x=start_orig, line_dash="dash", line_color="green")
    fig_raster.add_vline(x=start_orig + duration, line_dash="dash", line_color="red")
    # add line indicating 1 second duration and add text "1s"
    fig_raster.add_shape(
        type="line",
        x0=start,
        y0=-1,
        x1=start + 1000,
        y1=-1,
        line=dict(
            color="black",
            width=5,
        ),
    )
    fig_raster.add_annotation(
        x=start + 500,
        y=-1,
        text="1s",
        showarrow=False,
        yshift=-10,
    )
    # Update layout
    fig_raster.update_layout(
        title="Raster plot",
        xaxis_title="Time [ms]",
        yaxis_title="Neuron ID",
        xaxis=dict(range=[start, end], showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="white",
        showlegend=False,
    )

    fig_firing_rate = go.Figure()
    # line plot of firing rate in black
    fig_firing_rate.add_trace(
        go.Scatter(
            x=times_all,
            y=firing_rate,
            mode="lines",
            name="Firing rate",
            line=dict(color="black"),
        )
    )
    # Collect all burst start and end times
    burst_starts = []
    burst_ends = []
    burst_colors = []
    burst_index = []

    color = f"cluster_{n_clusters_current}"
    color_discrete_sequence = get_cluster_colors(n_clusters_current)
    for i_burst in df_plot.index:
        burst_starts.append(df_plot.loc[i_burst].start_orig)
        burst_ends.append(df_plot.loc[i_burst].end_orig)
        burst_colors.append(
            color_discrete_sequence[int(df_plot.loc[i_burst][color].split(" ")[1]) - 1]
        )
        burst_index.append(i_burst)

    # Add vertical lines for all burst windows
    for start, end, color, index in zip(
        burst_starts, burst_ends, burst_colors, burst_index
    ):
        fig_firing_rate.add_trace(
            go.Scatter(
                x=[start, start, end, end],
                y=[0, max(firing_rate), max(firing_rate), 0],
                mode="lines",
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=color,
                opacity=0.5,
                name="Burst",
                hoverinfo="x+name",
                customdata=[index],
            )
        )

    # put a big arrow on the current burst
    fig_firing_rate.add_annotation(
        x=start_orig + 0.5 * duration,
        y=max(firing_rate) * 0.9,
        ax=start_orig + 0.5 * duration,
        ay=max(firing_rate) * 1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="black",
        text="Current burst",
        font=dict(family="Courier New, monospace", size=16, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8,
    )
    # update layout
    fig_firing_rate.update_layout(
        title=title_firing_rate + f", {len(burst_starts)} bursts",
        xaxis_title="Time [ms]",
        yaxis_title="Rate [Hz]",
        # xaxis=dict(range=[start, end], showgrid=False),
        # yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig_raster, fig_firing_rate


if __name__ == "__main__":
    print("Starting the app.")
    if debug is True:
        print("Running locally.")
        app.run(debug=debug, port=8050)
    else:
        print("Running on the internet.")
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
