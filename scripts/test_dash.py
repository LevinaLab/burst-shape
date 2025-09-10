import os

import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

if "DEBUG" in os.environ:
    debug = os.environ["DEBUG"] == "True"
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to DEBUG mode")
    debug = True

# Sample Data (Assume already computed)
tsne_embedding = np.random.rand(10, 2)  # t-SNE embedding of 10 points
time_series_data = np.random.rand(10, 100)  # 10 time-series of length 100

# Create the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    [
        # t-SNE embedding plot
        dcc.Graph(
            id="tsne-plot",
            figure={
                "data": [
                    go.Scatter(
                        x=tsne_embedding[:, 0],
                        y=tsne_embedding[:, 1],
                        mode="markers",
                        marker=dict(size=10),
                        customdata=list(
                            range(len(tsne_embedding))
                        ),  # Index for callback
                    )
                ],
                "layout": go.Layout(
                    title="t-SNE Embedding",
                    clickmode="event+select",
                    xaxis={"title": "Dimension 1"},
                    yaxis={"title": "Dimension 2"},
                ),
            },
        ),
        # Time-series plot
        dcc.Graph(
            id="timeseries-plot",
            figure={
                "data": [go.Scatter(y=[0], mode="lines")],
                "layout": go.Layout(
                    title="Time-Series Data",
                    xaxis={"title": "Time"},
                    yaxis={"title": "Value"},
                ),
            },
        ),
    ]
)


# Callback to update time-series plot based on clicked point in t-SNE plot
@app.callback(Output("timeseries-plot", "figure"), [Input("tsne-plot", "clickData")])
def update_time_series(clickData):
    if clickData is None:
        # Default empty plot if no point is selected
        return {
            "data": [go.Scatter(y=[0], mode="lines")],
            "layout": go.Layout(title="Select a point from the t-SNE plot"),
        }

    # Get the index of the clicked point
    point_index = clickData["points"][0]["customdata"]

    # Get the corresponding time-series
    selected_timeseries = time_series_data[point_index]

    # Create the time-series plot
    return {
        "data": [go.Scatter(y=selected_timeseries, mode="lines")],
        "layout": go.Layout(
            title=f"Time-Series for Point {point_index}",
            xaxis={"title": "Time"},
            yaxis={"title": "Value"},
        ),
    }


# Run the app
if __name__ == "__main__":
    app.run(debug=False, port=8050)
    # app.run(DEBUG=False, port=5000, host="0.0.0.0")
