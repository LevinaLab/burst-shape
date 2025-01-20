import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from src.persistence import load_df_cultures
from src.persistence.spike_times import get_kapucu_spike_times

burst_extraction_params = (
    "burst_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
)

dataset = "kapucu" if "kapucu" in burst_extraction_params else "wagenaar"
df_cultures = load_df_cultures(burst_extraction_params)

# unique culture_type - mea_number - well_id combinations
pivot_table = pd.pivot_table(
    data=df_cultures,
    index=["culture_type", "mea_number", "well_id"]
    if dataset == "kapucu"
    else ["batch", "culture"],
    columns="DIV" if dataset == "kapucu" else "day",
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

# Dash App
app = dash.Dash(__name__)

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
            colorscale=colorscale,
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
        if dataset == "wagenaar":
            index_select = [int(x) for x in index_select]
        value = z[subjects.index(y)][days.index(x)]
        match dataset:
            case "kapucu":
                selected_text = (
                    f"Selected: day {div_day}, culture_type {index_select[0]}, "
                    f"mea_number {index_select[1]}, well_id {index_select[2]}"
                )
            case "wagenaar":
                selected_text = f"Selected: day {div_day}, batch {index_select[0]}, culture {index_select[1]}"

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
        case "wagenaar":
            index = (*index_select, div_day)
            st, gid = np.loadtxt("../data/extracted/%s-%s-%s.spk.txt" % index).T

    # trace of firing rate
    bin_size = 0.1  # s
    times_all = np.arange(0, st.max() + bin_size, bin_size)
    firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)  #  / 1000)
    times_all = 0.5 * (times_all[1:] + times_all[:-1])

    fig_whole = make_subplots(rows=2, cols=1, shared_xaxes=True, x_title="Time [s]")
    # line plot of firing rate in black
    fig_whole.add_trace(
        go.Scatter(
            x=times_all,
            y=firing_rate,
            mode="lines",
            name="Firing rate",
            line=dict(color="black"),
        ),
        1,
        1,
    )

    color = "red"
    for row, y_max in zip([1, 2], [max(firing_rate), max(gid)]):
        x_coords, y_coords = [], []
        for (start, end), _ in zip(
            df_cultures.at[index, "burst_start_end"],
            range(df_cultures.at[index, "n_bursts"]),
        ):
            x_coords.extend([start / 1000, start / 1000, end / 1000, end / 1000])
            y_coords.extend([0, y_max, y_max, 0])
        fig_whole.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=color,
                opacity=0.5,
                name="Burst",
                hoverinfo="x+name",
                # customdata=list(range(df_cultures.at[index, "n_bursts"])),
            )
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
    app.run_server(debug=True)
