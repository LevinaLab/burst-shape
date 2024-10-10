import os

import dash
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.cluster.hierarchy import fcluster

from scripts.test_dash import time_series_data
from src.folders import get_results_folder
from src.persistence import load_burst_matrix

###############################################################################
#                           Load Data and Parameters                          #
###############################################################################
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
n_clusters = 9

# define colors
palette = sns.color_palette(n_colors=n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    for c in cluster_colors
]

# load linkage -> labels
print(f"Loading linkage from {get_results_folder()}")
linkage_file = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "linkage.npy",
)
linkage = np.load(linkage_file)
labels = fcluster(linkage, t=n_clusters, criterion="maxclust")

burst_matrix = load_burst_matrix(burst_extraction_params)
file_idx = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "idx.npy",
)
if os.path.exists(file_idx):
    idx = np.load(file_idx)
    burst_matrix = burst_matrix[idx]

file_tsne = os.path.join(
    get_results_folder(),
    burst_extraction_params,
    clustering_params,
    "tsne.npy",
)
tsne_embedding = np.load(file_tsne)
time_series_data = burst_matrix

###############################################################################
#                       Interactive t-SNE Plot with Dash                      #
###############################################################################
# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    # t-SNE embedding plot
    dcc.Graph(
        id='tsne-plot',
        figure={
            'data': [
                go.Scatter(
                    x=tsne_embedding[:, 0],
                    y=tsne_embedding[:, 1],
                    mode='markers',
                    marker=dict(size=10),
                    customdata=list(range(len(tsne_embedding))),  # Index for callback
                )
            ],
            'layout': go.Layout(
                title='t-SNE Embedding',
                clickmode='event+select',
                xaxis={'title': 'Dimension 1'},
                yaxis={'title': 'Dimension 2'}
            )
        }
    ),

    # Time-series plot
    dcc.Graph(
        id='timeseries-plot',
        figure={
            'data': [go.Scatter(y=[0], mode='lines')],
            'layout': go.Layout(
                title='Time-Series Data',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value'}
            )
        }
    )
])


# Callback to update time-series plot based on clicked point in t-SNE plot
@app.callback(
    Output('timeseries-plot', 'figure'),
    [Input('tsne-plot', 'clickData')]
)
def update_time_series(clickData):
    if clickData is None:
        # Default empty plot if no point is selected
        return {
            'data': [go.Scatter(y=[0], mode='lines')],
            'layout': go.Layout(title='Select a point from the t-SNE plot')
        }

    # Get the index of the clicked point
    point_index = clickData['points'][0]['customdata']

    # Get the corresponding time-series
    selected_timeseries = time_series_data[point_index]

    # Create the time-series plot
    return {
        'data': [
            go.Scatter(
                y=selected_timeseries,
                mode='lines'
            )
        ],
        'layout': go.Layout(
            title=f'Time-Series for Point {point_index}',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Value'}
        )
    }


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
