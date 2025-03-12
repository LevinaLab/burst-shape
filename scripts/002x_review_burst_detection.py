import os

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, Dash
from flask import Flask
from plotly.subplots import make_subplots

from src.persistence import load_df_cultures
from src.persistence.spike_times import (
    get_hommersom_spike_times,
    get_inhibblock_spike_times,
    get_kapucu_spike_times,
)

if "DEBUG" in os.environ:
    debug = os.environ["DEBUG"] == "True"
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to debug mode")
    debug = True


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
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
else:
    dataset = "wagenaar"
print(f"Detected dataset: {dataset}")

match dataset:
    case "wagenaar":
        pivot_index = ["batch", "culture"]
        pivot_columns = "day"
    case "kapucu":
        pivot_index = ["culture_type", "mea_number", "well_id"]
        pivot_columns = "DIV"
    case "hommersom":
        pivot_index = ["batch", "clone"]
        pivot_columns = "well_idx"
    case "inhibblock":
        pivot_index = ["drug_label", "div"]
        pivot_columns = "well_idx"
    case _:
        raise NotImplementedError(f"{dataset} dataset is not implemented.")

df_cultures = load_df_cultures(burst_extraction_params)

# unique culture_type - mea_number - well_id combinations
pivot_table = pd.pivot_table(
    data=df_cultures,
    index=pivot_index,
    columns=pivot_columns,
    values="n_bursts",
    aggfunc="mean",
    fill_value=None,
)

subjects = pivot_table.index.tolist()
subjects = ["-".join([str(s) for s in subject]) for subject in subjects]
days = [f"D{day}" for day in pivot_table.columns.tolist()]
z = pivot_table.to_numpy()

# Custom Colorscale
colorscale = [
    [0, "blue"],  # Empty cells (None or np.nan)
    [1e-5, "blue"],  # Zero values
    [1.1e-5, "yellow"],  # Start of gradient
    [1.0, "red"],  # End of gradient
]
colorscale_alternative = [
    [0, "yellow"],
    [1.0, "red"],
]

colors_bursts = ["red", "blue", "green"]  # Define three alternating colors

# Dash App
server = Flask(__name__)  # Create a Flask app
app = Dash(__name__, server=server)  # Attach Dash to Flask

app.layout = html.Div(
    [
        dcc.Graph(
            id="matrix-plot", config={"displayModeBar": False}, style={"flex": "1"}
        ),  # Takes 2 parts
        # html.Div(id='selected-cell', style={'marginTop': '20px', 'flex': '1'}),  # Takes 1 part
        dcc.Graph(
            id="whole-recording", config={"displayModeBar": False}, style={"flex": "2"}
        ),  # Takes 3 parts
    ],
    style={"display": "flex", "flexDirection": "column", "height": "100vh"},
)


@app.callback(
    [
        Output("matrix-plot", "figure"),
        # Output('selected-cell', 'children'),
        Output("whole-recording", "figure"),
    ],
    [Input("matrix-plot", "clickData")],
)
def update_plot(click_data):
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=days,
            y=subjects,
            colorscale=colorscale if z.min() == 0 else colorscale_alternative,
            showscale=True,  # Show color legend
            hoverongaps=False,  # Avoid hover info for empty cells
        )
    )
    fig.update_layout(
        title="Subject-Day Matrix",
        xaxis_title="Days",
        yaxis_title="Subjects",
        xaxis=dict(tickvals=list(range(len(days))), ticktext=days, side="top"),
        yaxis=dict(tickvals=list(range(len(subjects))), ticktext=subjects),
        plot_bgcolor="white",
    )

    # Handle cell click
    if click_data:
        x = click_data["points"][0]["x"]  # Day label
        div_day = int(x[1:])
        y = click_data["points"][0]["y"]  # Subject label
        index_select = y.split("-")

        # value = z[subjects.index(y)][days.index(x)]
        match dataset:
            case "kapucu":
                selected_text = (
                    f"Selected: day {div_day}, culture_type {index_select[0]}, "
                    f"mea_number {index_select[1]}, well_id {index_select[2]}"
                )
            case "wagenaar":
                index_select = [int(x) for x in index_select]
                selected_text = f"Selected: day {div_day}, batch {index_select[0]}, culture {index_select[1]}"
            case "hommersom":
                selected_text = f"Selected: Batch {index_select[0]}, clone {index_select[1]}, well_idx {div_day}"
            case "inhibblock":
                index_select = (str(index_select[0]), int(index_select[1]))
                selected_text = f"Selected: drug_label {index_select[0]}, div {index_select[1]}, well_idx {div_day}"
            case _:
                raise NotImplementedError(f"{dataset} dataset is not implemented.")

        # Add a black rectangle around the selected cell
        fig.add_shape(
            type="rect",
            x0=days.index(x) - 0.5,  # Start of the cell in the x direction
            x1=days.index(x) + 0.5,  # End of the cell in the x direction
            y0=subjects.index(y) - 0.5,  # Start of the cell in the y direction
            y1=subjects.index(y) + 0.5,  # End of the cell in the y direction
            line=dict(color="black", width=5),  # Black border with width
            fillcolor="rgba(0, 0, 0, 0)",  # Transparent fill
        )

        fig_whole = _create_fig_whole_timeseries(
            df_cultures, index_select, div_day, selected_text
        )
    else:
        selected_text = "Click on a cell to see details."
        fig_whole = go.Figure()
    fig.update_layout(
        title=selected_text,
    )

    return fig, fig_whole


def _create_fig_whole_timeseries(df_cultures, index_select, div_day, selected_text):
    match dataset:
        case "kapucu":
            index = (*index_select, div_day)
            st, gid = get_kapucu_spike_times(
                df_cultures,
                index,
            )
            # st = np.array(st)
            st /= 1000  # convert to seconds
        case "hommersom":
            index = (*index_select, div_day)
            st, gid = get_hommersom_spike_times(
                df_cultures,
                index,
            )
            # st = np.array(st)
            st /= 1000  # convert to seconds
        case "wagenaar":
            index = (*index_select, div_day)
            st, gid = np.loadtxt("../data/extracted/%s-%s-%s.spk.txt" % index).T
        case "inhibblock":
            index = (*index_select, div_day)
            st, gid = get_inhibblock_spike_times(
                df_cultures,
                index,
            )
            st /= 1000  # convert to seconds
        case _:
            raise NotImplementedError(f"{dataset} dataset is not implemented.")

    # trace of firing rate
    bin_size = 0.1  # s
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)  #  / 1000)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])

    fig_whole = make_subplots(rows=2, cols=1, shared_xaxes=True, x_title="Time [s]")
    # line plot of firing rate in black
    fig_whole.add_trace(
        go.Scattergl(
            x=times_all,
            y=firing_rate,
            mode="lines",
            name="Firing rate",
            line=dict(color="black"),
        ),
        1,
        1,
    )

    for row, y_min, y_max in zip([1, 2], [0, min(gid)], [max(firing_rate), max(gid)]):
        x_coords_list, y_coords_list, color_list = [], [], []
        for i, ((start, end), _) in enumerate(
            zip(
                df_cultures.at[index, "burst_start_end"],
                range(df_cultures.at[index, "n_bursts"]),
            )
        ):
            color = colors_bursts[
                i % len(colors_bursts)
            ]  # Cycle through the three colors
            x_coords = [start / 1000, start / 1000, end / 1000, end / 1000]
            y_coords = [y_min, y_max, y_max, y_min]
            fig_whole.add_trace(
                go.Scattergl(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(color=color, width=2),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.5,
                    name=f"Burst {i + 1}",
                    hoverinfo="x+name",
                ),
                row=row,
                col=1,
            )

    fig_whole.add_trace(
        go.Scattergl(
            x=st,
            y=gid,
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=5,
                # color=palette[i_cluster - 1],
                line_width=1,
            ),
            name="Spikes",
        ),
        row=2,
        col=1,
    )
    # update layout
    fig_whole.update_layout(
        title=selected_text,  # + f", {len(burst_starts)} bursts",
        # xaxis_title="Time [ms]",
        yaxis_title="Rate [Hz]",
        yaxis2_title="GID",
        # xaxis=dict(range=[start, end], showgrid=False),
        # yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        showlegend=False,
    )
    fig_whole.update_traces(hoverinfo="skip")
    return fig_whole


if __name__ == "__main__":
    print("Starting the app.")
    if debug is True:
        print("Running locally.")
        app.run(debug=debug, port=8050)
    else:
        print("Running on the internet.")
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(debug=False, port=5000, host="0.0.0.0")
