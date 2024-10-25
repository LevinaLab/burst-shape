import os

import dash
import numpy as np
import pandas as pd
import plotly.express as px
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
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
n_clusters = np.arange(2, 21, 1)
n_clusters_initial = 9
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


cluster_colors = get_cluster_colors(n_clusters_initial)

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

file_tsne = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "tsne.npy" if embedding_type == "tsne" else "pca.npy",
)
embedding = np.load(file_tsne)

# prepare for plotly
for n_clusters_ in n_clusters:
    labels = fcluster(linkage, t=n_clusters_, criterion="maxclust")
    df_bursts[f"cluster_{n_clusters_}"] = [f"Cluster {label}" for label in labels]
# df_bursts["cluster"] = [f"Cluster {label}" for label in labels]
df_bursts["embedding_x"] = embedding[:, 0]
df_bursts["embedding_y"] = embedding[:, 1]
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
def create_tsne_plot(df_bursts, n_clusters_):
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


# Create the t-SNE plot
tsne_fig = create_tsne_plot(df_bursts, n_clusters_initial)

# Update the layout to enable cluster toggling


# Define layout of the Dash app
app.layout = html.Div(
    [
        # slider for selecting the number of clusters with label "slide to select number of clusters"
        dcc.Slider(
            id="n-clusters-slider",
            min=n_clusters[0],
            max=n_clusters[-1],
            step=1,
            value=n_clusters_initial,
            tooltip={"placement": "bottom", "always_visible": True},
            vertical=True,
        ),
        # t-SNE plot
        dcc.Graph(id="tsne-plot", figure=tsne_fig, style={"flex": "1"}),
        # Time series plot
        dcc.Graph(id="timeseries-plot", style={"flex": "1"}),
    ],
    style={"display": "flex", "flex-direction": "row", "height": "80vh"},
)
# ])


# Update t-SNE plot based on the selected number of clusters
@app.callback(
    Output("tsne-plot", "figure"),
    [Input("n-clusters-slider", "value")],
)
def update_tsne_plot(n_clusters_):
    print(f"Updating t-SNE plot with {n_clusters_} clusters")
    n_clusters_initial = n_clusters_
    return create_tsne_plot(df_bursts, n_clusters_)


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
                f"cluster_{n_clusters_initial}",
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


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
