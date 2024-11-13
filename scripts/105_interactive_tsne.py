import os

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.cluster.hierarchy import fcluster

from src.folders import get_results_folder
from src.persistence import load_burst_matrix, load_df_bursts

###############################################################################
#                           Parameters                                        #
###############################################################################
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
)
clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
n_clusters = np.arange(2, 21, 1)
n_clusters_current = 9  # initial number of clusters
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
    n_clusters_ = n_clusters_current
    tsne_fig = px.scatter(
        df_bursts,
        x="embedding_x",
        y="embedding_y",
        color=f"cluster_{n_clusters_}",
        color_discrete_sequence=get_cluster_colors(n_clusters_),
        category_orders={
            f"cluster_{n_clusters_}": [
                f"Cluster {i}" for i in range(1, n_clusters_ + 1)
            ]
        },
        labels={"color": "Cluster"},
        hover_data={
            "embedding_x": False,
            "embedding_y": False,
            f"cluster_{n_clusters_}": True,
            "batch": True,
            "culture": True,
            "day": True,
            "start_orig": True,
            "time_orig": True,
        },
        title="t-SNE plot",
        custom_data=[df_bursts.index],
    )

    tsne_fig.update_layout(
        legend_title="Clusters",
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
    )
    return tsne_fig


# Create the initial t-SNE plot
tsne_fig = update_tsne_plot(df_bursts)

# Define layout of the Dash app
app.layout = html.Div(
    [
        html.Div(
            [
                # drop-down menu for selecting what to plot
                dcc.Dropdown(
                    id="embedding-type",
                    options=[
                        {"label": "t-SNE", "value": "tsne"},
                        {"label": "PCA", "value": "pca"},
                    ],
                    value="tsne",
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
        # t-SNE plot
        dcc.Graph(id="tsne-plot", figure=tsne_fig, style={"flex": "1"}),
        # Time series plot of selected point
        dcc.Graph(id="timeseries-plot", style={"flex": "1"}),
        # Raster plot of selected point
        dcc.Graph(id="raster-plot", style={"flex": "1"}),
    ],
    style={"display": "flex", "flex-direction": "row", "height": "80vh"},
)
# ])

@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("embedding-type", "value")],
    prevent_initial_call=True,
)
def update_embedding_type(embedding_type_):
    _update_embedding(embedding_type_)
    return update_tsne_plot(df_bursts)


# Update t-SNE plot based on the selected number of clusters
@app.callback(
    Output("tsne-plot", "figure", allow_duplicate=True),
    [Input("n-clusters-slider", "value")],
    prevent_initial_call=True,
)
def update_tsne_plot_callback(n_clusters_):
    print(f"Updating t-SNE plot with {n_clusters_} clusters")
    global n_clusters_current
    n_clusters_current = n_clusters_
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

    batch, culture, day = list(df_bursts.iloc[point_index][['batch', 'culture', 'day']])
    # print(f"batch: {batch}, culture: {culture}, day: {day}")

    # Load spike data
    st, gid = np.loadtxt('../data/extracted/%s-%s-%s.spk.txt' % (batch, culture, day)).T
    st = st * 1000

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
