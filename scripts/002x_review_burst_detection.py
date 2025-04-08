import os

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from flask import Flask
from plotly.subplots import make_subplots

from src.persistence import load_df_cultures
from src.persistence.spike_times import (
    get_hommersom_spike_times,
    get_inhibblock_spike_times,
    get_kapucu_spike_times,
    get_mossink_spike_times,
)

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
    debug = os.environ["DEBUG"] == "True"
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to debug mode")
    debug = True

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
            burst_extraction_params = "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
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
        # "burst_dataset_mossink_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
        "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    )
citation = "the relevant literature"
doi_link = None
if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
    citation = "Kapucu et al. (2022)"
    doi_link = "https://doi.org/10.1038/s41597-022-01242-4"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
    citation = "Hommersom et al. (2024)"
    doi_link = "https://doi.org/10.1101/2024.03.18.585506"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    citation = "Vinogradov et al. (2024)"
    doi_link = "https://doi.org/10.1101/2024.08.21.608974"
elif "mossink" in burst_extraction_params:
    dataset = "mossink"
    citation = "Mossink et al. (2021)"
    doi_link = "https://doi.org/10.17632/bvt5swtc5h.1"
else:
    dataset = "wagenaar"
    citation = "Wagenaar et al. (2006)"
    doi_link = "https://doi.org/10.1186/1471-2202-7-11"
print(f"Detected dataset: {dataset}")

df_cultures = load_df_cultures(burst_extraction_params)

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
    case "mossink":
        pivot_index = ["group", "subject_id"]
        pivot_columns = "well_idx"
    case _:
        raise NotImplementedError(f"{dataset} dataset is not implemented.")

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
if isinstance(subjects[0], tuple):
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

GRAPH_ID = "whole-recording"
OVERVIEW_GRAPH_ID = "whole-recording-overview"

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
            id="matrix-plot", config={"displayModeBar": False}, style={"flex": "1"}
        ),
        # html.Div(id='selected-cell', style={'marginTop': '20px', 'flex': '1'}),  # Takes 1 part
        dcc.Graph(id=GRAPH_ID, config={"displayModeBar": True}, style={"flex": "2"}),
        dcc.Graph(
            id=OVERVIEW_GRAPH_ID, config={"displayModeBar": False}, style={"flex": ".2"}
        ),
        dcc.Loading(dcc.Store(id="store")),
    ],
    style={"display": "flex", "flexDirection": "column", "height": "100vh"},
)


@app.callback(
    [
        Output("matrix-plot", "figure"),
        # Output('selected-cell', 'children'),
        Output(GRAPH_ID, "figure"),
        Output(OVERVIEW_GRAPH_ID, "figure"),
        Output("store", "data"),
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
            case "mossink":
                index_select = (str(index_select[0]), int(index_select[1]))
                selected_text = f"Selected: group {index_select[0]}, subject_id {index_select[1]}, well_idx {div_day}"
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

        fig_whole, fig_whole_overview = _create_fig_whole_timeseries(
            df_cultures, index_select, div_day, selected_text
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
        case "mossink":
            index = (*index_select, div_day)
            st, gid = get_mossink_spike_times(
                df_cultures,
                index,
            )
            st /= 1000
        case _:
            raise NotImplementedError(f"{dataset} dataset is not implemented.")

    # trace of firing rate
    bin_size = 0.1  # s
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)  #  / 1000)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])

    # fig_whole = make_subplots(rows=2, cols=1, shared_xaxes=True, x_title="Time [s]")
    _fig_inside = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes="columns",
        horizontal_spacing=0.03,
        x_title="Time [s]",
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
                # color=palette[i_cluster - 1],
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
    if debug is True:
        print("Running locally.")
        app.run(debug=debug, port=8050)
    else:
        print("Running on the internet.")
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
