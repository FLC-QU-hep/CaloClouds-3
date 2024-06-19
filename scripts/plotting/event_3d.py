import plotly.express as px
import plotly.graph_objects as go
from pointcloud.data.edm4hep_root_reader import Points, Relationships
import sys
import numpy as np


def get_symmetric_limits(dataframe, x_mean, y_mean, z_mean):
    half_x_range = max(dataframe["X"].max() - x_mean, x_mean - dataframe["X"].min())
    half_y_range = max(dataframe["Y"].max() - y_mean, y_mean - dataframe["Y"].min())
    half_z_range = max(dataframe["Z"].max() - z_mean, z_mean - dataframe["Z"].min())
    return (
        (x_mean - half_x_range, x_mean + half_x_range),
        (y_mean - half_y_range, y_mean + half_y_range),
        (z_mean - half_z_range, z_mean + half_z_range),
    )


def get_hover_data():
    return ["Energy", "Time", "X", "Y", "Z"]


def get_hover_labels(dataframe):
    labels = []
    n_points = len(dataframe)
    for i in range(n_points):
        labels.append(
            f"Name: {dataframe['Name'].array[i]}, \n"
            f"Part: {dataframe['Part'].array[i]}, \n"
            f"Energy: {dataframe['Energy'].array[i]}, \n"
            f"Time: {dataframe['Time'].array[i]}, \n"
            f"(X, Y, Z): ({dataframe['X'].array[i]:.2f}, "
            f"{dataframe['Y'].array[i]:.2f}, {dataframe['Z'].array[i]:.2f})"
        )
    return labels


#def get_color_map(dataframe):
#    unique_names = sorted(dataframe["Name"].unique())
#    color_map = {
#        name: px.colors.qualitative.Plotly[i] for i, name in enumerate(unique_names)
#    }
#    return color_map
#
#
#def get_symbol_map(dataframe):
#    unique_parts = sorted(dataframe["Part"].unique())
#    symbols = ["circle", "square", "diamond", "cross", "x", "circle-open", "square-open", "diamond-open"]
#    symbol_map = {
#        part: symbols[i % len(symbols)] for i, part in enumerate(unique_parts)
#    }
#    return symbol_map
#
#
#def get_scatter_trace(dataframe, use_MC, particle_name, **kwargs):
#    matching = dataframe.loc[dataframe["Name"] == particle_name]
#    if use_MC:
#        matching = matching.loc[matching["Part"] == "MCParticles"]
#    else:
#        matching = matching.loc[matching["Part"] != "MCParticles"]
#    labels = get_hover_labels(matching)
#    colour = get_color_map(matching)[particle_name]
#    symbol_map = get_symbol_map(matching)
#    marker_symbols = [symbol_map[part] for part in matching["Part"]]
#    import ipdb; ipdb.set_trace()
#    trace = go.Scatter3d(
#        mode="markers",
#        x=matching["X"],
#        y=matching["Y"],
#        z=matching["Z"],
#        marker=dict(
#            color=colour,
#            size=matching["EnergySize"],
#    #        symbol=marker_symbols,
#            opacity=0.8,
#        ),
#        legendgroup=particle_name,
#        legendgrouptitle_text=particle_name,
#        text=labels,
#        hoverinfo="text",
#        **kwargs,
#    )
#    return trace


def simple_scatter(dataframe, **kwargs):
    fig = px.scatter_3d(
        dataframe,
        x="X",
        y="Y",
        z="Z",
        color="Name",
        size="EnergySize",
        symbol="Part",
        opacity=0.8,
        custom_data=get_hover_data(),
        **kwargs,
    )
    return fig


def scatter_data(dataframe, x_limits, y_limits, z_limits, **kwargs):
    fig = simple_scatter(dataframe, **kwargs)
    #traces = []
    # particle_names = dataframe["Name"].unique()
    #for name in particle_names:
    #    traces.append(get_scatter_trace(dataframe, True, name, **kwargs))
    #    traces.append(get_scatter_trace(dataframe, False, name, **kwargs))

    #fig = go.Figure(data=traces)

    # make the background black
    # and fix the axis ranges
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(1, 1, 1, 1)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                range=x_limits,
            ),
            yaxis=dict(
                backgroundcolor="rgba(1, 1, 1, 1)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                range=y_limits,
            ),
            zaxis=dict(
                backgroundcolor="rgba(1, 1, 1, 1)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                range=z_limits,
            ),
        )
    )

    # change the hover labels
    fig.update_traces(
        hovertemplate="<br>".join(
            [f"{key}: %{{customdata[{i}]}}" for i, key in enumerate(get_hover_data())]
        )
    )

    return fig


def intrested_in(particle, energy):
    intresting_particles = ["n", "p", "pi0", "pi+", "pi-", "K0L", "K0S", "K-", "K+"]
    intresting_pdgs = [2112, 2212, 111, 211, -211, 130, 310, -321, 321]
    intresting = intresting_particles + intresting_pdgs
    if particle in intresting:
        return energy > 1.5
    return False


def lines(relationships, event_number):
    parents = relationships.parent_MCParticle_idxs(event_number)
    mc_points = relationships.MCParticles(event_number)
    pdg = mc_points["PDG"]
    energy = mc_points["Energy"]
    total_points = len(mc_points["X"])
    unchecked = np.ones(total_points, dtype=bool)
    lines = []

    def add_to_line(point_idx):
        lines[-1][0].append(mc_points["X"][point_idx])
        lines[-1][1].append(mc_points["Y"][point_idx])
        lines[-1][2].append(mc_points["Z"][point_idx])

    def break_line():
        lines.append([[], [], []])

    # start from the far end of the parent list and work back
    next_idx = total_points - 1
    break_line()
    add_to_line(next_idx)
    while np.any(unchecked):
        if not unchecked[next_idx]:
            next_idx = np.where(unchecked)[0][0]
            add_to_line(next_idx)
        unchecked[next_idx] = False
        these_parents = parents[next_idx]
        if not len(these_parents):
            break_line()
            continue
        keep_all = intrested_in(pdg[next_idx], energy[next_idx])
        for parent in these_parents[:-1]:
            if keep_all or intrested_in(pdg[parent], energy[parent]):
                #print(f"Adding {pdg[parent]}, {energy[parent]} keep_all={keep_all}")
                add_to_line(parent)
                break_line()
        if keep_all or intrested_in(pdg[these_parents[-1]], energy[these_parents[-1]]):
            #print(f"Adding {pdg[these_parents[-1]]}, {energy[these_parents[-1]]} keep_all={keep_all}")
            add_to_line(these_parents[-1])
        next_idx = these_parents[-1]
    return lines


def add_lines(fig, relationships, event_number):
    list_lines = lines(relationships, event_number)
    if len(list_lines) == 0:
        x_lines, y_lines, z_lines = [], [], []
    else:
        x_lines, y_lines, z_lines = list_lines[0]
        for line in list_lines[1:]:
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)
            x_lines += line[0]
            y_lines += line[1]
            z_lines += line[2]

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color="white", width=2),
            hoverinfo="skip",
            name="Relationships",
        )
    )
    return fig


def get_event_choice(last_event=-1):
    try:
        choice = input(
            "Enter the event number, 'n' for next event," " or anything else to exit: "
        ).strip()
        if choice.lower() == "n":
            return last_event + 1
        return int(choice)
    except ValueError:
        return "stop"


def interactive_plot(file_names, **kwargs):
    points = Points(file_names)
    relationships = Relationships(file_names)
    print(f"Number of events: {len(points)}")
    choice = -1
    score_table = get_score_table(file_names)
    print_top_scores(score_table)
    print_bottom_scores(score_table)
    while (choice := get_event_choice(choice)) != "stop":
        if choice < 0:
            print("Invalid event number.")
            choice = -1
            continue
        if choice >= len(points):
            print("Event index out of range.")
            choice = -1
            continue
        print(f"Score is {score_table[choice]}")
        data = points[choice]
        x_mean, y_mean, z_mean = points.get_ecal_center(choice)
        x_limits, y_limits, z_limits = get_symmetric_limits(
            data, x_mean, y_mean, z_mean
        )
        fig = scatter_data(
            data, x_limits, y_limits, z_limits, symbol_map=points.symbol_map, **kwargs
        )
        add_lines(fig, relationships, choice)
        fig.show()


def get_score_table(
    file_names,
    scores_file="../point-cloud-diffusion-logs/anomalies/scores_per_file.npz",
):
    scores = np.load(scores_file)
    scores_list = []
    for name in file_names:
        name_base = name.split("/")[-1].split(".")[0].replace("-", "_")
        if "_part" in name:
            key = name_base + "_all_steps"
            scores_list.append(scores[key][:, 1])
        elif "_event" in name:
            pre_event, event_number = name_base.split("_event")
            if len(event_number) == 4:
                part_number = event_number[0]
            else:
                part_number = "0"
            event_number = int(event_number) - 1000 * int(part_number)
            key = pre_event + "_part" + part_number + "_all_steps"
            scores_list.append(scores[key][[event_number], 1])
    scores = np.concatenate(scores_list)
    return scores


def print_top_scores(score_table, n=10):
    top_scores = np.argsort(score_table)[::-1][:n]
    for i, score in enumerate(top_scores):
        print(f"Event {score} has a score of {score_table[score]}")
    return top_scores


def print_bottom_scores(score_table, n=10):
    bottom_scores = np.argsort(score_table)[:n]
    for i, score in enumerate(bottom_scores):
        print(f"Event {score} has a score of {score_table[score]}")
    return bottom_scores


if __name__ == "__main__":
    if len(sys.argv) > 1:
        interactive_plot(sys.argv[1:])
