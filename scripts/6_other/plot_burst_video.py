"""Code for plotting short video snippets of burst events."""

import json
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

from src.folders import get_data_human_slice_folder, get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.persistence.spike_times import get_spike_times_in_milliseconds
from src.settings import get_dataset_from_burst_extraction_params

# %% parameters
burst_extraction_params = "burst_dataset_human_slice_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load data
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)


# %% add spatial grid information
# map gid to x and y coordinates on the 16 x 16 grid
# 'A1' -> (0, 0), 'A2' -> (0, 1), ..., 'P16' -> (15, 15)
order_of_characters = "ABCDEFGHJKLMNOPR"  # note "I" and "Q" are missing


def _gid_to_coordinates(gid):
    x_string, y_string = gid[0], gid[1:]
    x = order_of_characters.index(x_string)
    y = int(y_string) - 1
    return x, y


def _gid_to_coordinates_vectorized(gids):
    _coordinates = np.zeros((gids.shape[0], 2), dtype=int)
    for i, gid in enumerate(gids):
        _coordinates[i, :] = _gid_to_coordinates(gid)
    # assert that coordinates are >0 and <16
    assert np.all(_coordinates >= 0) and np.all(
        _coordinates < 16
    ), f"Coordinates should be between 0 and 15, but got {_coordinates}"
    return _coordinates


df_cultures["coordinates"] = df_cultures["gid"].apply(_gid_to_coordinates_vectorized)

"""test_gids = df_cultures.at[df_cultures.index[0], 'gid']
for idx_random in np.random.randint(0, len(test_gids), 10):
    test_gid = test_gids[idx_random]
    x, y = _gid_to_coordinates(test_gid)
    print(f"gid: {test_gid}, coordinates: ({x}, {y})")

print("Testing vectorized version:")
test_gids_vectorized = df_cultures.at[df_cultures.index[0], 'gid'][np.random.randint(0, len(test_gids), 10)]
coordinates_vectorized = _gid_to_coordinates_vectorized(test_gids_vectorized)
for gid, (x, y) in zip(test_gids_vectorized, coordinates_vectorized):
    print(f"gid: {gid}, coordinates: ({x}, {y})")"""

all_gids = []
for i in range(16):
    for j in range(16):
        x_string = order_of_characters[i]
        y_string = str(j + 1)
        gid = f"{x_string}{y_string}"
        all_gids.append(gid)

# plot all gids on a 16 x 16 grid
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
for gid in all_gids:
    x, y = _gid_to_coordinates(gid)
    ax.text(x, y, gid, ha="center", va="center", fontsize=8)
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-0.5, 15.5)
ax.set_aspect("equal")
ax.set_xticks(range(16))
ax.set_yticks(range(16))
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_title("Mapping of GIDs to 16 x 16 grid coordinates")
fig.show()


# %% load layer dictionary and add layer information to the data
def _load_layer_json(idx_culture):
    path_json = os.path.join(
        get_data_human_slice_folder(),
        "raw",
        idx_culture[0],
        idx_culture[1],
        f"LayerDic_Cortex{idx_culture[1][2:]}.json",
    )
    with open(path_json, "r") as f:
        layer_dict = json.load(f)
    return layer_dict


_layer_color_map = {
    "": "lightgray",
    "layer1": "black",
    "layer2_3": "red",
    "layer4": "green",
    "layer5_6": "blue",
    "whitematter": "white",
}

_layer_numeric_map = {
    "": 0,
    "layer1": 1,
    "layer2_3": 2,
    "layer4": 3,
    "layer5_6": 4,
    "whitematter": 5,
}


def _load_layer_array(idx_culture):
    layer_dict = _load_layer_json(idx_culture)
    layer_string = np.zeros((16, 16), dtype="<U11")
    for layer, gids in layer_dict.items():
        for gid in gids:
            x, y = _gid_to_coordinates(gid)
            layer_string[x, y] = layer
    layer_numeric = np.zeros((16, 16), dtype=int)
    for layer, numeric_value in _layer_numeric_map.items():
        layer_numeric[layer_string == layer] = numeric_value
    return layer_string, layer_numeric


def _plot_layer(idx_culture):
    layer_string, layer_numeric = _load_layer_array(idx_culture)

    fig, ax = plt.subplots(constrained_layout=True)
    sns.despine()
    colors = [_layer_color_map[layer] for layer in _layer_numeric_map.keys()]
    labels = list(_layer_numeric_map.keys())
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    sns.heatmap(
        layer_numeric.T,
        ax=ax,
        cbar=False,
        cmap=cmap,
        norm=norm,
        vmin=0,
        vmax=len(colors) - 1,
    )
    ax.set_title("Layer mapping for culture " + idx_culture[1])

    # create legend showing color for each layer (show 'none' for empty label)
    patches = [
        matplotlib.patches.Patch(color=colors[i], label=labels[i] or "none")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    ax.set_xticks(range(16))
    ax.set_xticklabels(order_of_characters)
    ax.set_yticks(range(16))
    ax.set_yticklabels(range(1, 17))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    return fig, ax


fig, ax = _plot_layer(df_cultures.index[0])
fig.show()


# %% function to generate array for video: 16 x 16 x time_bins
def _get_video_array(
    df_cultures, df_bursts, idx_burst, bin_size=100, offsets=(-100, 100)
):
    # st = df_cultures.at[idx_burst[:-1], "times"] * 1000
    st, _ = get_spike_times_in_milliseconds(df_cultures, idx_burst[:-1], dataset)
    coordinates = df_cultures.at[idx_burst[:-1], "coordinates"]

    burst_start, burst_end = (
        df_bursts.at[idx_burst, "start_orig"] + offsets[0],
        df_bursts.at[idx_burst, "end_orig"] + offsets[1],
    )
    filtered_times = (st >= burst_start) & (st <= burst_end)
    st = st[filtered_times]
    assert len(st) > 0
    coordinates = coordinates[filtered_times]

    time_bins = np.arange(burst_start, burst_end + bin_size, bin_size)
    time_indices = np.searchsorted(time_bins, st) - 1

    video_array = np.zeros((16, 16, len(time_bins) - 1), dtype=int)
    for (x, y), t_idx in zip(coordinates, time_indices):
        video_array[x, y, t_idx] += 1

    return video_array, time_bins - df_bursts.at[idx_burst, "start_orig"]


# %% test plot a couple of frames
idx_burst_test = df_bursts.index[1500]
video_array, time_bins = _get_video_array(
    df_cultures, df_bursts, idx_burst_test, bin_size=20, offsets=(-20, 100)
)
for t_idx in range(min(video_array.shape[2], 12)):
    fig, ax = plt.subplots(constrained_layout=True)
    sns.despine()
    sns.heatmap(
        video_array[:, :, t_idx].T, ax=ax, cbar=True, vmin=0, vmax=video_array.max()
    )
    ax.set_title(f"Time bin: {time_bins[t_idx]:.1f} ms")
    fig.show()

# %% generate video with plotly
# idx_burst_test = df_bursts.index[409]
idx_burst_test = ("ID2519CT", "ID2519CT074", "Spont1", 7)
bin_size = 20
offsets = (-20, 20)
transition_speed = 300  # ms

video_array, time_bins = _get_video_array(
    df_cultures,
    df_bursts,
    idx_burst_test,
    bin_size=bin_size,
    offsets=offsets,
)
# Generate labels for the slider (start time of each bin)
bin_labels = [f"{time_bins[i]:.1f}ms" for i in range(len(time_bins) - 1)]


def plot_event(data, labels):
    fig = px.imshow(
        data.transpose(1, 0, 2),  # transpose to have (y, x, time)
        animation_frame=-1,
        labels={"animation_frame": "Time"},
        zmin=0,
        zmax=video_array.max(),
        color_continuous_scale="Viridis",
        binary_string=False,
        title=f"Video for {idx_burst_test}<br>"
        f"start: {df_bursts.at[idx_burst_test, 'start_orig']:.0f}ms, "
        f"bin size: {bin_size} ms",
    )
    fig.update_coloraxes(
        colorbar_title="Spike Count",
        colorbar=dict(
            len=0.7,  # length relative to plotting area (0–1)
            thickness=20,  # width of colorbar in px
            y=0.5,  # vertical position
            yanchor="middle",
            title=dict(side="right"),
        ),
    )

    # Replace frame indices (0, 1, 2...) with actual time strings
    for i, label in enumerate(labels):
        fig.layout.sliders[0].steps[i].label = label

    # axes ticks and tick labels
    fig.update_xaxes(
        tickmode="array",
        tickvals=np.arange(16),
        ticktext=[order_of_characters[i] for i in range(16)],
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=np.arange(16),
        ticktext=[str(i) for i in range(1, 17)],
    )

    # transition speed
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = transition_speed

    fig.show()
    return fig


fig, ax = _plot_layer(idx_burst_test[:-1])
fig.show()
fig.savefig(os.path.join(get_fig_folder(), "burst_layer_mapping.pdf"), dpi=300)


fig = plot_event(video_array, bin_labels)
html_path = os.path.join(get_fig_folder(), "burst_animation.html")
fig.write_html(html_path, include_plotlyjs="cdn", auto_play=False)

# Convert the animation to a GIF using kaleido and imageio
images = []

for i, frame in enumerate(fig.frames):
    # Apply frame updates to all traces
    for trace, frame_trace in zip(fig.data, frame.data):
        trace.update(frame_trace)

    # Update slider state
    if fig.layout.sliders:
        fig.layout.sliders[0].active = i

    images.append(imageio.imread(fig.to_image(format="png", engine="kaleido")))

pil_images = [Image.fromarray(img) for img in images]

pil_images[0].save(
    os.path.join(get_fig_folder(), "burst_animation.gif"),
    save_all=True,
    append_images=pil_images[1:],
    duration=transition_speed,  # milliseconds per frame
    loop=0,
)
