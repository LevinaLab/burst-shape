import base64
import io
import os

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.colors
import seaborn as sns
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.cluster.hierarchy import fcluster

from src.folders import get_results_folder, get_data_kapucu_folder
from src.persistence import load_burst_matrix, load_df_bursts

###############################################################################
#                           Parameters                                        #
###############################################################################
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
    "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)

dataset = "kapucu" if "kapucu" in burst_extraction_params else "wagenaar"

clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
n_clusters = np.arange(2, 21, 1)
n_clusters_current = 9  # initial number of clusters
color_by = "cluster"  # initial color by
marker_size = 3  # initial marker size
embedding_type = ["tsne", "pca"][0]


###############################################################################
#                           Prepare data                                      #
###############################################################################
# define colors
def get_cluster_colors(n_clusters_):
    palette = sns.color_palette(n_colors=n_clusters_)
    cluster_colors = [palette[i - 1] for i in range(1, n_clusters_ + 1)]
    # convert colors to string (hex format)
    cluster_colors = [
        f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
        for c in cluster_colors
    ]
    return cluster_colors


cluster_colors = get_cluster_colors(n_clusters_current)

# load linkage -> labels
print(f"Loading linkage from {get_results_folder()}")
linkage_file = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "linkage.npy",
)
linkage = np.load(linkage_file)
# labels = fcluster(linkage, t=n_clusters, criterion="maxclust")

burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
file_idx = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "idx.npy",
)
if os.path.exists(file_idx):
    idx = np.load(file_idx)
    burst_matrix = burst_matrix[idx]
    df_bursts = df_bursts.iloc[idx]

def _update_embedding(embedding_type_):
    file_embedding = os.path.join(
        get_results_folder(),
        burst_extraction_params,
        clustering_params,
        "tsne.npy" if embedding_type_ == "tsne" else "pca.npy",
    )
    embedding = np.load(file_embedding)
    df_bursts["embedding_x"] = embedding[:, 0]
    df_bursts["embedding_y"] = embedding[:, 1]
    return


_update_embedding(embedding_type)

# prepare for plotly
for n_clusters_ in n_clusters:
    labels = fcluster(linkage, t=n_clusters_, criterion="maxclust")
    df_bursts[f"cluster_{n_clusters_}"] = [f"Cluster {label}" for label in labels]
# df_bursts["cluster"] = [f"Cluster {label}" for label in labels]
df_bursts.reset_index(inplace=True)

df_bursts["firing_rate"] = df_bursts["integral"] / 50  # df_bursts["time_orig"]
match dataset:
    case "wagenaar":
        df_bursts["batch_culture"] = df_bursts["batch"].astype(str) + "-" + df_bursts["culture"].astype(str)
    case "kapucu":
        df_bursts["batch"] = df_bursts["culture_type"].astype(str) + "-" + df_bursts["mea_number"].astype(str)
        df_bursts["batch_culture"] = df_bursts["batch"].astype(str) + "-" + df_bursts["well_id"].astype(str)

# confirm burst_matrix and df_bursts["burst"] are the same
# for i in range(len(burst_matrix)):
#     assert np.allclose(burst_matrix[i], df_bursts["burst"].iloc[i])
# print("Burst matrix and df_bursts['burst'] are the same")
print("Starting Dash app...")

###############################################################################
#                       Interactive t-SNE Plot with Dash                      #
###############################################################################
app = dash.Dash(__name__)

# Create a color map based on cluster labels
# color_map = cluster_colors  # px.colors.qualitative.Plotly[:n_clusters]


# Create the t-SNE scatter plot
def update_tsne_plot(df_bursts):
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
            df_bursts[color] = df_bursts[color].astype(str)
            color_discrete_sequence = px.colors.qualitative.Set1
            match dataset:
                case "wagenaar":
                    category_orders = {color: sorted(df_bursts["batch"].unique(), key=int)}
                case "kapucu":
                    category_orders = {color: sorted(df_bursts["batch"].unique())}
            legend_title = "Batch"
        case "batch_culture":
            color = "batch_culture"
            df_bursts[color] = df_bursts[color].astype(str)
            color_discrete_sequence = px.colors.qualitative.Set1
            match dataset:
                case "wagenaar":
                    category_orders = {color: sorted(
                        df_bursts["batch_culture"].unique(),
                        key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])),
                    )}
                case "kapucu":
                    category_orders = {color: sorted(df_bursts["batch_culture"].unique())}
            legend_title = "Batch-Culture"
        case "day":
            color = "day" if dataset == "wagenaar" else "DIV"
            df_bursts[color] = df_bursts[color].astype(str)
            unique_days = sorted(df_bursts[color].unique(), key=int)
            category_orders = {color: unique_days}
            viridis_colors = plotly.colors.sample_colorscale(px.colors.sequential.Viridis, [i / len(unique_days) for i in range(len(unique_days))])
            color_discrete_map = {
                str(day): viridis_colors[i]
                for i, day in enumerate(unique_days)
            }
            legend_title = "Day"
        case "time_orig":
            color = np.log10(pd.to_numeric(df_bursts["time_orig"], errors='coerce'))
            # df_bursts[color] = pd.to_numeric(df_bursts[color], errors='coerce')
            color_continuous_scale = px.colors.sequential.Viridis
            legend_title = "Duration"
            color_log = True
        case "firing_rate":
            color = np.log10(pd.to_numeric(df_bursts["firing_rate"], errors='coerce'))
            color_continuous_scale = px.colors.sequential.Viridis
            legend_title = "Firing rate"
            color_log = True
        case _:
            print(f"Invalid color_by: {color_by}")
            raise ValueError(f"Invalid color_by: {color_by}")

    n_clusters_ = n_clusters_current
    tsne_fig = px.scatter(
        df_bursts,
        x="embedding_x",
        y="embedding_y",
        color=color,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        category_orders=category_orders,
        hover_data={
            "embedding_x": False,
            "embedding_y": False,
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
    )
    return tsne_fig


# Create the initial t-SNE plot
tsne_fig = update_tsne_plot(df_bursts)
tsne_plot = dcc.Graph(id="tsne-plot", figure=tsne_fig, style={"flex": "1"})

# Define layout of the Dash app
app.layout = html.Div(
    [
        html.Div(
            [
                html.Button("Download as PDF", id="download-button"),
                dcc.Download(id="download-pdf"),
                # drop-down menu for selecting what to plot
                dcc.Dropdown(
                    id="embedding-type",
                    options=[
                        {"label": "t-SNE", "value": "tsne"},
                        {"label": "PCA", "value": "pca"},
                    ],
                    value="tsne",
                ),
                # drop-down menu for selecting which column to color by
                dcc.Dropdown(
                    id="color-by",
                    options=[
                        {"label": "Day", "value": "day"},
                        {"label": "Culture", "value": "batch_culture"},
                        {"label": "Batch", "value": "batch"},
                        {"label": "Cluster", "value": "cluster"},
                        {"label": "Duration", "value": "time_orig"},
                        {"label": "Firing rate", "value": "firing_rate"},
                    ],
                    value="cluster",
                ),
                # slider for selecting the number of clusters with label "slide to select number of clusters"
                dcc.Slider(
                    id="n-clusters-slider",
                    min=n_clusters[0],
                    max=n_clusters[-1],
                    step=1,
                    value=n_clusters_current,
                    tooltip={"placement": "bottom", "always_visible": True},
                    vertical=True,
                ),
            ],
            style={"flex": "0 0 100px", "display": "flex", "flex-direction": "column"},
        ),
        html.Div(
            [
                # t-SNE plot
                tsne_plot,
                dcc.Slider(
                    id="marker-size-slider",
                    min=1,
                    max=10,
                    step=1,
                    value=marker_size,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "flex": "1"},
        ),
        # Time series plot of selected point
        dcc.Graph(id="timeseries-plot", style={"flex": "1"}),
        # Raster plot of selected point
        dcc.Graph(id="raster-plot", style={"flex": "1"}),
    ],
    style={"display": "flex", "flex-direction": "row", "height": "80vh"},
)


@app.callback(
    Output("download-pdf", "data"),
    Input("download-button", "n_clicks"),
    Input("tsne-plot", "figure"),
    prevent_initial_call=True
)
def download_pdf(n_clicks, figure):
    ctx = dash.callback_context

    # Check if the download button was clicked
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'download-button':
        raise dash.exceptions.PreventUpdate

    # Create a BytesIO buffer to hold the PDF data
    pdf_buffer = io.BytesIO()
    fig = go.Figure(figure)
    fig.write_image(pdf_buffer, format="pdf")
    pdf_data = base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")

    return dcc.send_bytes(lambda x: x.write(pdf_buffer.getvalue()), "figure.pdf")


@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("embedding-type", "value")],
    prevent_initial_call=True,
)
def update_embedding_type(embedding_type_):
    _update_embedding(embedding_type_)
    return update_tsne_plot(df_bursts)


# Update t-SNE plot based on the selected column to color by
@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("color-by", "value")],
    prevent_initial_call=True,
)
def update_color_by(color_by_):
    print(f"Updating t-SNE plot with color by {color_by_}")
    global color_by
    color_by = color_by_
    return update_tsne_plot(df_bursts)

# Update t-SNE plot based on the selected number of clusters
@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("n-clusters-slider", "value")],
    prevent_initial_call=True,
)
def update_n_clusters(n_clusters_):
    print(f"Updating t-SNE plot with {n_clusters_} clusters")
    global n_clusters_current
    n_clusters_current = n_clusters_
    return update_tsne_plot(df_bursts)

@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("marker-size-slider", "value")],
    prevent_initial_call=True,
)
def update_marker_size(marker_size_):
    print(f"Updating t-SNE plot with marker size {marker_size_}")
    global marker_size
    marker_size = marker_size_
    return update_tsne_plot(df_bursts)

# Update time series plot based on the selected point in the t-SNE plot
@app.callback(Output("timeseries-plot", "figure"), [Input("tsne-plot", "clickData")])
def update_timeseries(selected_data):
    if selected_data is None:
        return px.line(pd.DataFrame(), title="Select a point to view time series")

    # Extract index of selected point
    point_index = selected_data["points"][0]["customdata"][0]

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
    timeseries_fig = px.line(
        df_bursts.iloc[point_index]["burst"],
        title=title,
    )
    return timeseries_fig


@app.callback(Output("raster-plot", "figure"), [Input("tsne-plot", "clickData")])
def update_raster(selected_data):
    if selected_data is None:
        return px.line(pd.DataFrame(), title="Select a point to view time series")

    # Extract index of selected point
    point_index = selected_data["points"][0]["customdata"][0]
    # print(f"point_index: {point_index}")
    # print(df_bursts.iloc[point_index])

    match dataset:
        case "wagenaar":
            batch, culture, day = list(df_bursts.iloc[point_index][['batch', 'culture', 'day']])
            # print(f"batch: {batch}, culture: {culture}, day: {day}")

            # Load spike data
            st, gid = np.loadtxt('../data/extracted/%s-%s-%s.spk.txt' % (batch, culture, day)).T
            st = st * 1000
        case "kapucu":
            culture_type, mea_number, well_id, div_day = list(df_bursts.iloc[point_index][['culture_type', 'mea_number', 'well_id', 'DIV']])
            # print(f"culture_type: {culture_type}, mea_number: {mea_number}, well_id: {well_id}, div_day: {div_day}")

            # Load spike data
            data_folder = get_data_kapucu_folder()
            path_time_series = os.path.join(data_folder, "_".join([culture_type, "20517" if culture_type == "hPSC" else "190617", mea_number, f"DIV{div_day}", "spikes.csv"]))
            # print(path_time_series)
            # print(os.path.exists(path_time_series))
            spikes = pd.read_csv(path_time_series)
            # print(spikes)
            spikes[['well', 'ch_n']] = spikes['Channel'].str.split('_', expand=True)
            # print(spikes)
            spikes = spikes[spikes['well'] == well_id]
            # print(spikes)
            st = spikes['Time'].values
            gid = spikes['ch_n'].values
            st = st * 1000

    # cut out a window around the burst
    start_orig = df_bursts.iloc[point_index]['start_orig']
    duration = df_bursts.iloc[point_index]['time_orig']
    start = start_orig - 800
    end = start_orig + 2000
    if start_orig + duration > end:
        end = start_orig + duration + 500
    st, gid = st[(st >= start) & (st <= end)], gid[(st >= start) & (st <= end)]




    # Create raster plot with '|' markers for spikes and no line connecting them
    fig = go.Figure()
    # Raster plot with vertical line markers
    fig.add_trace(
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
            name="Spikes"
        )
    )
    # Vertical reference line for start and start + duration
    fig.add_vline(x=start_orig, line_dash="dash", line_color="green")
    fig.add_vline(x=start_orig + duration, line_dash="dash", line_color="red")
    # add line indicating 1 second duration and add text "1s"
    fig.add_shape(
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
    fig.add_annotation(
        x=start + 500,
        y=-1,
        text="1s",
        showarrow=False,
        yshift=-10,
    )
    # Update layout
    fig.update_layout(
        title="Raster plot",
        xaxis_title="Time [ms]",
        yaxis_title="Neuron ID",
        xaxis=dict(range=[start, end], showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="white",
        showlegend=False
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
