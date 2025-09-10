"""
Interactive tool for reviewing detected bursts.

This starts a plotly application that displays detected bursts.
It shows an overview of recordings indicating the number of detected bursts as a heatmap.
When clicking on a recording it shows the recording's spikes as raster plot and the firing rate.
The detected bursts are highlighted with rectangles.
"""
import os
import warnings

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from flask import Flask
from plotly.subplots import make_subplots

from src.persistence import load_df_cultures
from src.persistence.spike_times import get_spike_times_in_seconds
from src.settings import get_citation_doi_link, get_dataset_from_burst_extraction_params

# -----------------------------------------------------------------------------
# Get RESAMPLE and DEBUG from environment
# RESAMPLE: use FigureResampler() to reduce traffic, useful for deploying online
# DEBUG:
if "RESAMPLE" in os.environ:
    RESAMPLE = os.environ["RESAMPLE"] == "True"
    print(f"RESAMPLE environment variable present, RESAMPLE set to {RESAMPLE}")
else:
    print("No RESAMPLE environment variable present, RESAMPLE defaulting to False")
    RESAMPLE = False

if RESAMPLE:
    from dash_extensions.enrich import DashProxy, Serverside, ServersideOutputTransform
    from plotly_resampler import ASSETS_FOLDER, FigureResampler
    from plotly_resampler.aggregation import MinMaxLTTB

if "DEBUG" in os.environ:
    DEBUG = os.environ["DEBUG"] == "True"
    print(f"DEBUG environment variable present, DEBUG set to {DEBUG}")
else:
    print("No DEBUG environment variable: defaulting to DEBUG mode")
    DEBUG = True

# -----------------------------------------------------------------------------
# Select dataset
# If DATASET is present in environment choose that, otherwise select manually
if "DATASET" in os.environ:
    match os.environ["DATASET"]:
        case "wagenaar":
            burst_extraction_params = "burst_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
        case "kapucu":
            burst_extraction_params = "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
        case "hommersom":
            burst_extraction_params = "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        case "hommersom_test":
            burst_extraction_params = "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
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
        # "burst_dataset_hommersom_test_minIBI_50_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_hommersom_test_minIBI_50_n_bins_50_normalization_integral_min_length_30_min_firing_rate_1585"
        # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_mossink_maxISIstart_100_maxISIb_100_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    )
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
citation, doi_link = get_citation_doi_link(dataset)
print(f"Detected dataset: {dataset}")

df_cultures = load_df_cultures(burst_extraction_params)

# -----------------------------------------------------------------------------
# DATASET specific settings
# This is the layout for the recordings overview
# pivot_index will be the row index
# pivot columns will be the columns
match dataset:
    case "wagenaar":
        pivot_index = ["batch", "culture"]
        pivot_columns = "day"
        rows_title = "Batch-Culture"
        columns_title = "Days"
    case "kapucu":
        pivot_index = ["culture_type", "mea_number", "well_id"]
        pivot_columns = "DIV"
        rows_title = "Culture Type - MEA - Well"
        columns_title = "DIV"
    case "hommersom_test":
        pivot_index = ["batch", "clone"]
        pivot_columns = "well_idx"
        rows_title = "Batch-Clone"
        columns_title = "Well Index"
    case "inhibblock":
        pivot_index = ["drug_label", "div"]
        pivot_columns = "well_idx"
        rows_title = "Group - Batch"
        columns_title = "Well Index"
    case "mossink":
        pivot_index = ["group", "subject_id"]
        pivot_columns = "well_idx"
        rows_title = "Group - Subject ID"
        columns_title = "Well Index"
    case "hommersom" | "hommersom_binary":
        pivot_index = ["batch"]
        pivot_columns = "well"
        rows_title = "Batch"
        columns_title = "Well"
    case _:
        index_cultures = df_cultures.index.names
        pivot_index = index_cultures[:-1]
        pivot_columns = index_cultures[-1]
        rows_title = f"{index_cultures[:-1]}"
        columns_title = index_cultures[-1]
        warnings.warn(
            f"No layout for the recordings overview defined for dataset={dataset}. "
            f"Defaulting now to pivot_index={pivot_index} and pivot_columns={pivot_columns}."
        )


def _generate_selected_text(index_select, column_select):
    return f"Selected: {rows_title} {index_select}; {columns_title} {column_select}."


# unique culture_type - mea_number - well_id combinations
pivot_table = pd.pivot(
    data=df_cultures.reset_index(),
    index=pivot_index,
    columns=pivot_columns,
    values="n_bursts",
)

DELIM = " | "


def _encode_index(idx_val):
    """Encode single-level or multi-level index value into a human-readable string."""
    if not isinstance(idx_val, tuple):
        idx_val = (idx_val,)
    return DELIM.join(str(v) for v in idx_val)


label_to_index = {_encode_index(tup): tup for tup in pivot_table.index}


def _decode_label_to_index(label):
    idx = label_to_index.get(label, None)
    if isinstance(idx, tuple):
        return idx
    else:
        return (idx,)


def _encode_column(col_val):
    return str(col_val)


label_to_column = {_encode_column(col): col for col in pivot_table.columns}


def _decode_column(label):
    return label_to_column.get(label, None)


subjects = [_encode_index(tup) for tup in pivot_table.index]
days = [_encode_column(col) for col in pivot_table.columns]
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
server = Flask(__name__)

if RESAMPLE:
    # NOTE: Remark how the assets folder is passed to the Dash(proxy) application and how
    #       the lodash script is included as an external script.
    app = DashProxy(
        __name__,
        server=server,
        transforms=[
            ServersideOutputTransform()
        ],  # (backends=[RedisBackend(default_timeout=600)])],
        assets_folder=ASSETS_FOLDER,
        external_scripts=["https://cdn.jsdelivr.net/npm/lodash/lodash.min.js"],
    )
    # app = DashProxy(__name__, server=server, transforms=[ServersideOutputTransform()])
else:
    app = Dash(__name__, server=server)

RECORDING_OVERVIEW_ID = "matrix-plot"
GRAPH_ID = "whole-recording"
OVERVIEW_GRAPH_ID = "whole-recording-overview"
N_BURSTS_SELECT_ID = "n_bursts_selection"

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
        dcc.Graph(
            id=RECORDING_OVERVIEW_ID,
            config={"displayModeBar": False},
            style={"flex": "1"},
        ),
        html.Div(
            children=[
                html.Label("number of burst colors:", style={"marginRight": "8px"}),
                dcc.Dropdown(
                    id=N_BURSTS_SELECT_ID,
                    options=[
                        {"label": f"{i}", "value": i}
                        for i in range(1, len(colors_bursts) + 1)
                    ],
                    value=1,
                    style={"width": "fit-content"},
                    clearable=False,
                ),
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id=GRAPH_ID, config={"displayModeBar": True}, style={"flex": "2"}
                ),
                dcc.Graph(
                    id=OVERVIEW_GRAPH_ID,
                    config={"displayModeBar": False},
                    style={"flex": ".2"},
                ),
                dcc.Loading(dcc.Store(id="store")),
            ],
            style={"display": "flex", "flexDirection": "column", "flex": "2"},
        ),
    ],
    style={"display": "flex", "flexDirection": "column", "height": "100vh"},
)


@app.callback(
    [
        Output(RECORDING_OVERVIEW_ID, "figure"),
        # Output('selected-cell', 'children'),
        Output(GRAPH_ID, "figure"),
        Output(OVERVIEW_GRAPH_ID, "figure"),
        Output("store", "data"),
    ],
    [
        Input(RECORDING_OVERVIEW_ID, "clickData"),
        Input(N_BURSTS_SELECT_ID, "value"),
    ],
)
def update_plot(click_data, n_burst_colors):
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=days,
            y=subjects,
            colorscale=colorscale if z.min() == 0 else colorscale_alternative,
            showscale=True,  # Show color legend
            hoverongaps=False,  # Avoid hover info for empty cells
            colorbar=dict(
                title="Number of Bursts",
                titleside="right",
                titlefont=dict(size=20),
            ),
        )
    )
    fig.update_layout(
        title="Subject-Day Matrix",
        xaxis_title=columns_title,
        yaxis_title=rows_title,
        xaxis=dict(
            side="top",
            titlefont=dict(size=20),
        ),
        yaxis=dict(
            titlefont=dict(size=20),
        ),
        plot_bgcolor="white",
    )

    # Handle cell click
    if click_data:
        x_label = click_data["points"][0]["x"]
        div_day = _decode_column(x_label)
        y_label = click_data["points"][0]["y"]
        index_select = _decode_label_to_index(y_label)
        selected_text = _generate_selected_text(index_select, div_day)

        # Add a black rectangle around the selected cell
        fig.add_shape(
            type="rect",
            x0=days.index(x_label) - 0.5,  # Start of the cell in the x direction
            x1=days.index(x_label) + 0.5,  # End of the cell in the x direction
            y0=subjects.index(y_label) - 0.5,  # Start of the cell in the y direction
            y1=subjects.index(y_label) + 0.5,  # End of the cell in the y direction
            line=dict(color="black", width=5),  # Black border with width
            fillcolor="rgba(0, 0, 0, 0)",  # Transparent fill
        )

        fig_whole, fig_whole_overview = _create_fig_whole_timeseries(
            df_cultures, index_select, div_day, selected_text, n_burst_colors
        )
    else:
        selected_text = "Click on a cell to see details."
        fig_whole = go.Figure()
        fig_whole_overview = go.Figure()
    fig.update_layout(
        title=selected_text,
    )

    if RESAMPLE:
        serverside = Serverside(fig_whole)
    else:
        serverside = None
    return fig, fig_whole, fig_whole_overview, serverside


def _create_fig_whole_timeseries(
    df_cultures, index_select, div_day, selected_text, n_burst_colors
):
    index_df_cultures = (*index_select, div_day)
    st, gid = get_spike_times_in_seconds(df_cultures, index_df_cultures, dataset)

    # trace of firing rate
    bin_size = 0.1  # s
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)  #  / 1000)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])

    _fig_inside = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes="columns",
        horizontal_spacing=0.03,
    )
    if RESAMPLE:
        fig_whole: FigureResampler = FigureResampler(
            _fig_inside,
            create_overview=True,
            overview_row_idxs=[0],
            default_downsampler=MinMaxLTTB(parallel=True),
        )
    else:
        fig_whole = _fig_inside

    # line plot of firing rate in black
    fig_whole.add_trace(
        go.Scattergl(
            mode="lines",
            name="Firing rate",
            line=dict(color="black"),
            **(
                {}
                if RESAMPLE
                else {
                    "x": times_all,
                    "y": firing_rate,
                }
            ),
        ),
        **(
            {}
            if not RESAMPLE
            else {
                "hf_x": times_all,
                "hf_y": firing_rate,
            }
        ),
        row=1,
        col=1,
    )

    for row, y_min, y_max in zip([1, 2], [0, min(gid)], [max(firing_rate), max(gid)]):
        for i, ((start, end), _) in enumerate(
            zip(
                df_cultures.at[index_df_cultures, "burst_start_end"],
                range(df_cultures.at[index_df_cultures, "n_bursts"]),
            )
        ):
            color = colors_bursts[:n_burst_colors][
                i % n_burst_colors
            ]  # Cycle through the three colors
            x_coords = [start / 1000, start / 1000, end / 1000, end / 1000]
            y_coords = [y_min, y_max, y_max, y_min]
            fig_whole.add_trace(
                go.Scattergl(
                    mode="lines",
                    line=dict(color=color, width=2),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.5,
                    name=f"Burst {i + 1}",
                    hoverinfo="x+name",
                    **(
                        {}
                        if RESAMPLE
                        else {
                            "x": x_coords,
                            "y": y_coords,
                        }
                    ),
                ),
                **(
                    {}
                    if not RESAMPLE
                    else {
                        "hf_x": x_coords,
                        "hf_y": y_coords,
                    }
                ),
                row=row,
                col=1,
            )

    fig_whole.add_trace(
        go.Scattergl(
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=5,
                line_width=1,
            ),
            name="Spikes",
            **(
                {}
                if RESAMPLE
                else {
                    "x": st,
                    "y": gid,
                }
            ),
        ),
        **(
            {}
            if not RESAMPLE
            else {
                "hf_x": st,
                "hf_y": gid,
                "max_n_samples": 5000,
            }
        ),
        row=2,
        col=1,
    )
    # update layout
    fig_whole.update_layout(
        title=selected_text,
        xaxis=dict(
            title="Time [s]",
            titlefont=dict(size=20),
        ),
        xaxis2=dict(
            title="Time [s]",
            titlefont=dict(size=20),
        ),
        yaxis=dict(
            title="Rate [Hz]",
            titlefont=dict(size=20),
        ),
        yaxis2=dict(
            title="GID",
            titlefont=dict(size=20),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )
    fig_whole.update_traces(hoverinfo="skip")
    if RESAMPLE:
        fig_whole_overview = fig_whole._create_overview_figure()
    else:
        fig_whole_overview = go.Figure()
    return fig_whole, fig_whole_overview


if RESAMPLE:
    # --- Clientside callbacks used to bidirectionally link the overview and main graph ---
    app.clientside_callback(
        dash.ClientsideFunction(namespace="clientside", function_name="main_to_coarse"),
        dash.Output(
            OVERVIEW_GRAPH_ID, "id", allow_duplicate=True
        ),  # TODO -> look for clean output
        dash.Input(GRAPH_ID, "relayoutData"),
        [dash.State(OVERVIEW_GRAPH_ID, "id"), dash.State(GRAPH_ID, "id")],
        prevent_initial_call=True,
    )
    app.clientside_callback(
        dash.ClientsideFunction(namespace="clientside", function_name="coarse_to_main"),
        dash.Output(GRAPH_ID, "id", allow_duplicate=True),
        dash.Input(OVERVIEW_GRAPH_ID, "selectedData"),
        [dash.State(GRAPH_ID, "id"), dash.State(OVERVIEW_GRAPH_ID, "id")],
        prevent_initial_call=True,
    )

    # --- FigureResampler update callback ---

    # The plotly-resampler callback to update the graph after a relayout event (= zoom/pan)
    # As we use the figure again as output, we need to set: allow_duplicate=True
    @app.callback(
        Output(GRAPH_ID, "figure", allow_duplicate=True),
        Input(GRAPH_ID, "relayoutData"),
        State("store", "data"),  # The server side cached FigureResampler per session
        prevent_initial_call=True,
        # memoize=True,
    )
    def update_fig(relayoutdata: dict, fig: FigureResampler):
        if isinstance(fig, FigureResampler):
            return fig.construct_update_data_patch(relayoutdata)
        else:
            return no_update


if __name__ == "__main__":
    print("Starting the app.")
    if DEBUG is True:
        print("Running locally.")
        app.run(debug=DEBUG, port=8050)
    else:
        print("Running on the internet.")
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
