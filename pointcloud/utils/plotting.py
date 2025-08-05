from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic

from ..utils import metrics



nice_hex = [
    ["#00E5E3", "#8DD7BF", "#FF96C5", "#FF5768", "#FFBF65"],
    ["#FC6238", "#FFD872", "#F2D4CC", "#E77577", "#6C88C4"],
    ["#C05780", "#FF828B", "#E7C582", "#00B0BA", "#0065A2"],
    ["#00CDAC", "#FF6F68", "#FFDACC", "#FF60A8", "#CFF800"],
    ["#FF5C77", "#4DD091", "#FFEC59", "#FFA23A", "#74737A"],
]


class PltConfigs:
    def __init__(self):
        # legend font
        self.font = font_manager.FontProperties(
            family="serif",
            size=23,
            # size=20
        )
        self.text_size = 20

        # radial profile
        self.bins_r = 35
        self.origin = (0, 40)
        # self.origin = (3.754597092*10, -3.611833191*10)

        # occupancy
        self.occup_bins = np.linspace(150, 1419, 50)
        self.plot_text_occupancy = False
        self.occ_indent = 20

        # e_sum
        self.e_sum_bins = np.linspace(20.01, 2200, 50)
        self.plot_text_e = False
        self.plot_legend_e = True
        self.e_indent = 20

        # hits
        self.hit_bins = np.logspace(np.log10(0.01000001), np.log10(200), 50)
        # self.hit_bins = np.logspace(np.log10(0.01), np.log10(100), 70)
        self.ylim_hits = (1, 1 * 1e7)
        # self.ylim_hits = (10, 8*1e5)

        # CoG
        self.bins_cog = 30
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.cog_ranges = [
            (-0.3 - 1.5, -0.3 + 1.5),
            (1875, 1940),
            (39.8 - 1.5, 39.8 + 1.5),
        ]

        # xyz featas
        self.bins_feats = 50
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.feats_ranges = [(-200, 200), (1811, 2011), (-160, 240)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]

        # all
        self.threshold = 0.1  # MeV / half a MIP
        # self.color_lines = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        self.color_lines = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
        # self.color_lines = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

        self.include_artifacts = (
            False  # include_artifacts = True -> keeps points that hit dead material
        )

        # percentile edges for occupancy based metrics
        self.layer_edges = [0, 8, 11, 13, 15, 16, 18, 19, 21, 24, 29]
        self.radial_edges = [
            0,
            6.558,
            9.849,
            12.96,
            17.028,
            23.434,
            33.609,
            40.119,
            48.491,
            68.808,
            300,
        ]

        # error band
        self.alpha = 0.3


plt_config = PltConfigs()


def get_cog(cloud, thr=plt_config.threshold):  # expects shape [events, points, 4]
    """
    Get Center of Gravity
    """
    cloud = cloud.copy()
    # set all pionts with energy below threshold to zero
    mask = cloud[..., -1] < thr
    cloud[mask] = 0

    x, y, z, e = (
        cloud[..., 0],
        cloud[..., 1],
        cloud[..., 2],
        cloud[..., 3],
    )

    # mask out events with zero or negative e sum
    mask = e.sum(axis=1) > 0
    x, y, z, e = x[mask], y[mask], z[mask], e[mask]

    x_cog = np.sum((x * e), axis=1) / e.sum(axis=1)
    y_cog = np.sum((y * e), axis=1) / e.sum(axis=1)
    z_cog = np.sum((z * e), axis=1) / e.sum(axis=1)
    return x_cog, y_cog, z_cog


def get_features(
    plt_config, map_layers, half_cell_size_global, events, global_shower_axis_char
):
    """
    High level features used to evaluate events.

    Parameters
    ----------
    plt_config : PltConfigs
        plotting configurations
    map_layers : list
        list of dictionaries, each containing the grid of cells for a layer
    half_cell_size_global : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    events : list
        list of 2D arrays, each containing the energy deposited
        in each cell of the layer with the arangement specified by map_layers
        in the style of a histogram
    global_shower_axis_char : char
        "x", "y" or "z" specifying the direction of the local z axis
        that runs through the layers, in the global coordinate system
        of the detector

    Returns
    -------
    features : dict
        dictionary containing the following features:
        e_radial : array
            radial profile of the energy deposition
        e_sum : array
            total energy deposited in the detector
        hits : array
            energy deposited in each cell
        occ : array
            number of cells hit
        e_layers_distibution : array
            energy deposited in each layer
        e_layers : array
            average energy deposited in each layer
        e_layers_std : array
            standard deviation of the energy deposited in each layer
        occ_layers : array
            number of cells hit in each layer
        e_radial_lists : list
            list of radial profiles of the energy deposition for each layer
        hits_noThreshold : array
            energy deposited in each cell without threshold
        binned_layer_e : array
            binned energy deposited in each layer
        binned_radial_e : array
            binned radial profile of the energy deposition

    """
    incident_point = plt_config.origin

    occ_list = []  # occupancy
    hits_list = []  # energy per cell
    e_sum_list = []  # energy per shower
    e_radial = []  # radial profile
    e_layers_list = []  # energy per layer
    occ_layers_list = []  # occupancy per layer
    e_radidal_lists = []  # radial profile per layer
    hits_noThreshold_list = []  # energy per cell without threshold

    for layers in tqdm(events):
        occ = 0
        e_sum = 0
        e_layers = []
        occ_layers = []
        e_radial_layers = []
        for i, layer in enumerate(layers):
            # layer = layer*1000 # energy rescale to MeV
            layer = layer.copy()  # for following inplace operations
            layer_noThreshold = layer.copy()
            layer[layer < plt_config.threshold] = 0

            hit_mask = layer > 0  # shape i.e. 82,81
            layer_hits = layer[hit_mask]

            occ += hit_mask.sum()
            e_sum += layer.sum()

            hits_list.append(layer_hits)
            hits_noThreshold_list.append(layer_noThreshold[layer_noThreshold > 0])
            e_layers.append(layer.sum())

            occ_layers.append(hit_mask.sum())

            # get radial profile #######################

            x_hit_idx, y_hit_idx = np.where(hit_mask)

            if not global_shower_axis_char == "y":
                raise NotImplementedError(
                    "We don't have a muon map system for different"
                    " global shower directions than y"
                )
            # due to rotations, the x and z have been swapped.
            local_x = "z"
            local_y = "x"

            x_cell_coord = (
                map_layers[i][f"{local_x}edges"][:-1][x_hit_idx] + half_cell_size_global
            )
            y_cell_coord = (
                map_layers[i][f"{local_y}edges"][:-1][y_hit_idx] + half_cell_size_global
            )
            e_cell = layer[x_hit_idx, y_hit_idx]
            dist_to_origin = np.sqrt(
                (x_cell_coord - incident_point[0]) ** 2
                + (y_cell_coord - incident_point[1]) ** 2
            )
            e_radial.append([dist_to_origin, e_cell])
            e_radial_layers.append([dist_to_origin, e_cell])
            ############################################

        e_layers_list.append(e_layers)
        occ_layers_list.append(occ_layers)

        occ_list.append(occ)
        e_sum_list.append(e_sum)

        e_radial_layers = np.concatenate(e_radial_layers, axis=1)
        e_radidal_lists.append(e_radial_layers)

    features = {}

    # out shape: [2, flattend hits]
    features["e_radial"] = np.concatenate(e_radial, axis=1)
    features["e_sum"] = np.array(e_sum_list)
    # hit energies
    features["hits"] = np.concatenate(hits_list)
    features["occ"] = np.array(occ_list)
    # distibution of energy per layer
    features["e_layers_distibution"] = np.array(e_layers_list)
    # average energy per layer
    features["e_layers"] = np.array(e_layers_list).mean(axis=0)
    # std of energy per layer
    features["e_layers_std"] = np.array(e_layers_list).std(axis=0)
    features["occ_layers"] = np.array(occ_layers_list)  # .sum(axis=0)/len(events)
    # nested list: e_rad_lst[ EVENTS ][DIST,E ] [ POINTS ]
    features["e_radial_lists"] = e_radidal_lists
    # hit energies without threshold
    features["hits_noThreshold"] = np.concatenate(hits_noThreshold_list)

    # add binned layer and radial energy metrics
    # shape: [bin_centeres, events]
    features["binned_layer_e"] = metrics.binned_layer_energy(
        features["e_layers_distibution"], bin_edges=plt_config.layer_edges
    )
    # shape: [bin_centeres, events]
    features["binned_radial_e"] = metrics.binned_radial_energy(
        features["e_radial_lists"], bin_edges=plt_config.radial_edges
    )

    return features


def plt_radial(
    e_radial,
    e_radial_list,
    labels,
    plt_config=plt_config,
    title=r"\textbf{full spectrum}",
    events=40_000,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)

    ## for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(e_radial_list)):
        axs[0].plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    axs[0].legend(prop=plt_config.font, loc="upper right")
    ########################################################

    # mean and std as binned statistic
    mean, edges, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="mean"
    )
    std, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="std"
    )
    count, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="count"
    )

    mean_shower = mean * count / events  # mean shower energy per bin
    std_shower = std / np.sqrt(
        count
    )  # std of individual event measures --> std of mean
    std_shower = std_shower * count / events  # std of shower energy per bin

    # data histogram
    axs[0].stairs(mean_shower, edges=edges, color="lightgrey", fill=True)
    # h = axs[0].hist(e_radial[0], bins=plt_config.bins_r, weights=e_radial[1]/events, color='lightgrey', rasterized=True)
    # h = axs[0].hist(e_radial[0], bins=plt_config.bins_r, weights=e_radial[1]/events, color='dimgrey', histtype='step', lw=2)

    # uncertainty band
    axs[0].stairs(
        mean_shower + std_shower,
        edges=edges,
        baseline=mean_shower - std_shower,
        color="dimgrey",
        lw=2,
        hatch="///",
    )

    for i, e_radial_ in enumerate(e_radial_list):
        # h1 = axs[0].hist(e_radial_[0], bins=edges, weights=e_radial_[1]/events, histtype='step', linestyle='-', lw=3, color=plt_config.color_lines[i])

        # mean and std as binned statistic
        mean, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="mean"
        )
        std, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="std"
        )
        count, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="count"
        )
        mean_shower_gen = mean * count / events  # mean shower energy per bin
        std_shower_gen = std / np.sqrt(
            count
        )  # std of individual event measures --> std of mean
        std_shower_gen = std_shower_gen * count / events  # std of shower energy per bin

        # gen histogram
        axs[0].stairs(
            mean_shower_gen,
            edges=edges,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
        )
        # uncertainty band
        axs[0].stairs(
            mean_shower_gen + std_shower_gen,
            edges=edges,
            baseline=mean_shower_gen - std_shower_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

        # ratio plot on the bottom
        lims_min = 0.5
        lims_max = 1.9

        # plot ratio
        ratio = (mean_shower_gen) / (mean_shower)
        axs[1].stairs(ratio, edges=edges, color=plt_config.color_lines[i], lw=3)

        # plot error band
        ratio_err = ratio * np.sqrt(
            ((std_shower) / (mean_shower)) ** 2
            + ((std_shower_gen) / (mean_shower_gen)) ** 2
        )
        axs[1].stairs(
            ratio + ratio_err,
            edges=edges,
            baseline=ratio - ratio_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle="-", lw=1, color="k")

    axs[0].set_ylim(1.1e-4, 5e4)

    axs[0].set_yscale("log")

    # ax2 = ax1.twiny()
    # ax2.set_xticks( ax1.get_xticks() )
    # ax2.set_xbound(ax1.get_xbound())
    # ax2.set_xticklabels([int(x / (half_cell_size_global*2)) for x in ax1.get_xticks()])
    # ax2.set_xlabel("radius [cells]")

    plt.xlabel("radius [mm]")
    axs[0].set_ylabel("mean energy [MeV]")
    axs[1].set_ylabel("ratio to G4")

    # ax = plt.gca()
    # plt.text(0.70, 1.02, 'full spectrum', fontsize=plt_config.font.get_size(), family='serif', transform=ax.transAxes)

    plt.subplots_adjust(hspace=0.1)
    # plt.tight_layout()

    plt.savefig("radial.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def plt_spinal(
    e_layers,
    e_layers_list,
    e_layers_std_real,
    e_layers_std_list,
    labels,
    plt_config=plt_config,
    title=r"\textbf{full spectrum}",
    events=40_000,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)

    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) - 10, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(e_layers_list)):
        axs[0].plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    ########################################################

    # pos = np.arange(1, len(e_layers) + 1)
    bins = np.arange(0.5, len(e_layers) + 1.5)

    # h_data = axs[0].hist(
    #     pos, bins=bins, weights=e_layers, color="lightgrey", rasterized=True
    # )
    axs[0].stairs(e_layers, edges=bins, color="lightgrey", fill=True)
    # axs[0].hist(
    #     pos, bins=bins, weights=e_layers, color="dimgrey", histtype="step", lw=2
    # )

    # uncertainty band
    err_data = e_layers_std_real / np.sqrt(
        events
    )  # std of individual measures --> std of mean
    axs[0].stairs(
        e_layers + err_data,
        edges=bins,
        baseline=e_layers - err_data,
        color="dimgrey",
        lw=2,
        hatch="///",
    )

    for i, (e_layers_, e_layers_std_) in enumerate(
        zip(e_layers_list, e_layers_std_list)
    ):
        # axs[0].hist(
        #     pos,
        #     bins=bins,
        #     weights=e_layers_,
        #     histtype="step",
        #     linestyle="-",
        #     lw=3,
        #     color=plt_config.color_lines[i],
        # )
        axs[0].stairs(
            e_layers_, edges=bins, linestyle="-", lw=3, color=plt_config.color_lines[i]
        )
        # uncertainty band
        err_gen = e_layers_std_ / np.sqrt(events)
        axs[0].stairs(
            e_layers_ + err_gen,
            edges=bins,
            baseline=e_layers_ - err_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

        # ratio plot on the bottom
        lims_min = 0.8
        lims_max = 1.05

        # plot ratio
        ratio = (e_layers_) / (e_layers)
        axs[1].stairs(ratio, edges=bins, color=plt_config.color_lines[i], lw=3)

        # plot error band
        ratio_err = ratio * np.sqrt(
            ((err_gen) / (e_layers_)) ** 2 + ((err_data) / (e_layers)) ** 2
        )
        axs[1].stairs(
            ratio + ratio_err,
            edges=bins,
            baseline=ratio - ratio_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle="-", lw=1, color="k")

    axs[0].set_yscale("log")
    axs[0].set_ylim(1.1e-1, 2e2)
    axs[0].set_xlim(0, 31)
    plt.xlabel("layers")
    axs[0].set_ylabel("mean energy [MeV]")
    axs[1].set_ylabel("ratio to G4")

    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    # plt.legend(prop=plt_config.font, loc='best')
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig("spinal.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def plt_occupancy(occ, occ_list, labels, plt_config=plt_config):
    plt.figure(figsize=(7, 7))

    # for legend ##########################################
    plt.hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(occ_list)):
        plt.plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    ########################################################

    h = plt.hist(occ, bins=plt_config.occup_bins, color="lightgrey", rasterized=True)
    h = plt.hist(
        occ, bins=plt_config.occup_bins, color="dimgrey", histtype="step", lw=2
    )

    for i, occ_ in enumerate(occ_list):
        plt.hist(
            occ_,
            bins=h[1],
            histtype="step",
            linestyle="-",
            lw=2.5,
            color=plt_config.color_lines[i],
        )

    plt.xlim(
        plt_config.occup_bins.min() - plt_config.occ_indent,
        plt_config.occup_bins.max() + plt_config.occ_indent,
    )
    plt.xlabel("number of hits")
    plt.ylabel("# showers")

    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    # plt.legend(prop=plt_config.font, loc='best')
    if plt_config.plot_text_occupancy:
        plt.text(315, 540, "10 GeV", fontsize=plt_config.font.get_size() + 2)
        plt.text(870, 215, "50 GeV", fontsize=plt_config.font.get_size() + 2)
        plt.text(1230, 170, "90 GeV", fontsize=plt_config.font.get_size() + 2)

    plt.tight_layout()
    plt.savefig("occ.pdf", dpi=100)
    plt.show()


def plt_hit_e(
    hits, hits_list, labels, plt_config=plt_config, title=r"\textbf{full spectrum}"
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)

    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(hits_list)):
        axs[0].plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    # plt.legend(prop=plt_config.font, loc="best")
    # plt.title(
    #     r"\textbf{validation set, 50 GeV}",
    #     fontsize=plt_config.font.get_size(),
    #     loc="right",
    # )
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################

    h_data = axs[0].hist(
        hits, bins=plt_config.hit_bins, color="lightgrey", rasterized=True
    )
    # h = axs[0].hist(
    #     hits, bins=plt_config.hit_bins, histtype="step", color="dimgrey", lw=2
    # )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )

    for i, hits_ in enumerate(hits_list):
        h_gen = axs[0].hist(
            hits_,
            bins=h_data[1],
            histtype="step",
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
        )

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

        # ratio plot on the bottom
        lims_min = 0.5
        lims_max = 1.9
        # eps = 1e-8 # to avoid division by zero

        # plot ratio
        ratio = (h_gen[0]) / (h_data[0])
        axs[1].stairs(ratio, edges=h_data[1], color=plt_config.color_lines[i], lw=3)

        # plot error band
        ratio_err = ratio * np.sqrt(
            ((h_data_err) / (h_data[0])) ** 2 + ((h_gen_err) / (h_gen[0])) ** 2
        )
        axs[1].stairs(
            ratio + ratio_err,
            edges=h_data[1],
            baseline=ratio - ratio_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    axs[1].set_ylim(lims_min, lims_max)

    # horizontal line at 1
    axs[1].axhline(1, linestyle="-", lw=1, color="k")

    axs[0].axvspan(
        h_data[1].min(),
        plt_config.threshold,
        facecolor="gray",
        alpha=0.5,
        hatch="/",
        edgecolor="k",
    )
    axs[1].axvspan(
        h_data[1].min(),
        plt_config.threshold,
        facecolor="gray",
        alpha=0.5,
        hatch="/",
        edgecolor="k",
    )
    axs[0].set_xlim(h_data[1].min(), h_data[1].max())
    # axs[0].set_xlim(h_data[1].min(), 3e2)
    axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])

    axs[0].set_yscale("log")
    axs[0].set_xscale("log")

    plt.xlabel("visible cell energy [MeV]")
    axs[0].set_ylabel("# cells")
    axs[1].set_ylabel("ratio to G4")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig("hits.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def plt_esum(e_sum, e_sum_list, labels, plt_config=plt_config):
    plt.figure(figsize=(7, 7))

    h = plt.hist(
        np.array(e_sum), bins=plt_config.e_sum_bins, color="lightgrey", rasterized=True
    )
    h = plt.hist(
        np.array(e_sum),
        bins=plt_config.e_sum_bins,
        histtype="step",
        color="dimgrey",
        lw=2,
    )

    # for legend ##########################################
    plt.hist(
        np.zeros(10), label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(e_sum_list)):
        plt.plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    ########################################################

    for i, e_sum_ in enumerate(e_sum_list):
        plt.hist(
            np.array(e_sum_),
            bins=h[1],
            histtype="step",
            linestyle="-",
            lw=2.5,
            color=plt_config.color_lines[i],
        )

    plt.xlim(
        plt_config.e_sum_bins.min() - plt_config.e_indent,
        plt_config.e_sum_bins.max() + plt_config.e_indent,
    )
    plt.xlabel("energy sum [MeV]")
    plt.ylabel("# showers")

    if plt_config.plot_text_e:
        plt.text(300, 740, "10 GeV", fontsize=plt_config.font.get_size() + 2)
        plt.text(1170, 250, "50 GeV", fontsize=plt_config.font.get_size() + 2)
        plt.text(1930, 160, "90 GeV", fontsize=plt_config.font.get_size() + 2)
        plt.ylim(0, 799)

    # if plt_config.plot_legend_e:
    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    # plt.legend(prop=plt_config.font, loc='best')

    plt.tight_layout()
    plt.savefig("e_sum.pdf", dpi=100)
    plt.show()


def plt_cog(
    cog, cog_list, labels, plt_config=plt_config, title=r"\textbf{full spectrum}"
):
    lables = ["X", "Y", "Z"]
    # plt.figure(figsize=(21, 9))
    fig, axs = plt.subplots(2, 3, figsize=(25, 9), height_ratios=[3, 1], sharex="col")

    for k, j in enumerate([0, 2, 1]):
        # plt.subplot(1, 3, k+1)

        axs[0, k].set_xlim(plt_config.cog_ranges[j])

        # real data
        h_data = axs[0, k].hist(
            np.array(cog[j]),
            bins=plt_config.bins_cog,
            color="lightgrey",
            range=plt_config.cog_ranges[j],
            rasterized=True,
        )
        # h = axs[0, k].hist(
        #     np.array(cog[j]), bins=h[1], color="dimgrey", histtype="step", lw=2
        # )

        # uncertainty band
        h_data_err = np.sqrt(h_data[0])
        axs[0, k].stairs(
            h_data[0] + h_data_err,
            edges=h_data[1],
            baseline=h_data[0] - h_data_err,
            color="dimgrey",
            lw=2,
            hatch="///",
        )

        # for legend ##############################################
        if k == k:
            #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
            axs[0, k].hist(
                np.zeros(1) - 10,
                label=labels[0],
                color="lightgrey",
                edgecolor="dimgrey",
                lw=2,
            )
            for i in range(len(cog_list)):
                axs[0, k].plot(
                    0,
                    0,
                    linestyle="-",
                    lw=3,
                    color=plt_config.color_lines[i],
                    label=labels[i + 1],
                )
        ###########################################################

        # generated data
        for i, cog_ in enumerate(cog_list):
            h_gen = axs[0, k].hist(
                np.array(cog_[j]),
                bins=h_data[1],
                histtype="step",
                linestyle="-",
                lw=3,
                color=plt_config.color_lines[i],
                range=plt_config.cog_ranges[j],
            )

            # uncertainty band in histogram
            h_gen_err = np.sqrt(h_gen[0])
            axs[0, k].stairs(
                h_gen[0] + h_gen_err,
                edges=h_gen[1],
                baseline=h_gen[0] - h_gen_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )

            # ratio plot on the bottom
            lims_min = [0.5, 0.5, 0.5]
            lims_max = [1.5, 1.5, 1.5]
            # eps = 1e-8  # to avoid division by zero

            # plot ratio
            ratio = (h_gen[0]) / (h_data[0])
            axs[1, k].stairs(
                ratio, edges=h_data[1], color=plt_config.color_lines[i], lw=2
            )

            # plot error band
            ratio_err = ratio * np.sqrt(
                ((h_data_err) / (h_data[0])) ** 2 + ((h_gen_err) / (h_gen[0])) ** 2
            )
            axs[1, k].stairs(
                ratio + ratio_err,
                edges=h_data[1],
                baseline=ratio - ratio_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )

        # horizontal line at 1
        axs[1, k].axhline(1, linestyle="-", lw=1, color="k")

        # for legend ##############################################
        if k == 2:
            # plt.legend(prop=plt_config.font, loc=(0.37, 0.76))
            axs[0, k].legend(prop=plt_config.font, loc="best")

        # ax = plt.gca()
        axs[0, k].set_title(title, fontsize=plt_config.font.get_size(), loc="right")

        ###########################################################

        axs[0, k].set_ylim(0, max(h_data[0]) + max(h_data[0]) * 0.2)
        if k == 2:
            axs[0, k].set_ylim(0, max(h_data[0]) + max(h_data[0]) * 0.75)
        axs[1, k].set_ylim(lims_min[k], lims_max[k])
        # axs[1, k].set_yscale('log')

        axs[1, k].set_xlabel(f"center of gravity {lables[j]} [mm]")
        axs[0, k].set_ylabel("# showers")
        axs[1, k].set_ylabel("ratio to G4")

    # plt.tight_layout()
    plt.subplots_adjust(left=0, hspace=0.1, wspace=0.25)
    plt.savefig("cog.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def plt_feats(
    events: np.array,
    events_list: list,
    labels: list,
    plt_config: PltConfigs = plt_config,
    title: str = r"\textbf{full spectrum}",
    scale: str | None = None,
    density: bool = False,
):
    """
    Plot the features of the showers. (? TODO better description)

    Parameters
    ----------
    events : np.array (n_events, max_points, 4)
        The showersfrom a ground truth. The third dimension is (x, y, z, e).
    events_list : list of np.array (n_events, max_points, 4)
        The showers for the different models. The third dimension is (x, y, z, e).
    labels : list
        Names of the models.
    plt_config : PltConfigs, optional
        The configuration for the plot.
    title : str, optional
        The title of the plot.
    scale : str, optional
        The scale of the y-axis. Default is None, can be 'log'.
    density : bool, optional
        If True, the y-axis is the density. Default is False.
    """
    lables = ["X", "Y", "Z"]
    plt.figure(figsize=(21, 7))

    for k, j in enumerate([0, 2, 1]):
        plt.subplot(1, 3, k + 1)

        plt.xlim(plt_config.feats_ranges[j])

        h = plt.hist(
            np.array(events[:, :, j][events[:, :, 3] != 0.0].flatten()),
            bins=plt_config.bins_feats,
            color="lightgrey",
            range=plt_config.feats_ranges[j],
            rasterized=True,
            density=density,
        )
        # h = plt.hist(
        #     np.array(events[:, j, :][events[:, 3, :] != 0.0].flatten()),
        #     bins=h[1],
        #     color="dimgrey",
        #     histtype="step",
        #     lw=2,
        #     density=density,
        # )

        # for legend ##############################################
        if k == k:
            #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
            plt.hist(
                np.ones(10) * (-300),
                label=labels[0],
                color="lightgrey",
                edgecolor="dimgrey",
                lw=2,
                density=density,
            )
            for i in range(len(events_list)):
                plt.plot(
                    0,
                    0,
                    linestyle="-",
                    lw=3,
                    color=plt_config.color_lines[i],
                    label=labels[i + 1],
                )
        ###########################################################

        for i, events_ in enumerate(events_list):
            plt.hist(
                np.array(events_[:, :, j][events_[:, :, 3] != 0.0].flatten()),
                bins=h[1],
                histtype="step",
                linestyle="-",
                lw=3,
                color=plt_config.color_lines[i],
                range=plt_config.feats_ranges[j],
                density=density,
            )

        # for legend ##############################################
        if k == 2:
            # plt.legend(prop=plt_config.font, loc=(0.37, 0.76))
            plt.legend(prop=plt_config.font, loc="best")

        plt.title(title, fontsize=plt_config.font.get_size(), loc="right")

        ###########################################################

        if density:
            plt.ylim(1e-6, max(h[0]) + max(h[0]) * 0.5)
        else:
            plt.ylim(1, max(h[0]) + max(h[0]) * 0.5)

        if scale == "log":
            plt.yscale("log")

        plt.xlabel(f"feature {lables[j]}")
        plt.ylabel("# points")

    plt.tight_layout()
    # plt.savefig('cog.pdf', dpi=100)
    plt.show()


def plt_occupancy_singleE(occ_list, occ_list_list, labels, plt_config=plt_config):
    fig, axs = plt.subplots(1, 1, figsize=(9, 9), sharex=False)

    # for legend ##########################################
    axs.hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(occ_list_list[0])):
        axs.plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    ########################################################

    for j, (occ, occ_list) in enumerate(
        zip(occ_list, occ_list_list)
    ):  # loop over energyies
        h = axs.hist(
            occ, bins=plt_config.occup_bins, color="lightgrey", rasterized=True
        )
        # h = axs.hist(
        #     occ, bins=plt_config.occup_bins, color="dimgrey", histtype="step", lw=2
        # )

        # uncertainty
        h_err = np.sqrt(h[0])
        axs.stairs(
            h[0] + h_err,
            edges=h[1],
            baseline=h[0] - h_err,
            color="dimgrey",
            lw=2,
            hatch="///",
        )

        for i, occ_ in enumerate(occ_list):  # loop over models
            h1 = axs.hist(
                occ_,
                bins=h[1],
                histtype="step",
                linestyle="-",
                lw=2.5,
                color=plt_config.color_lines[i],
            )

            # uncertainty band in histogram
            h1_err = np.sqrt(h1[0])
            axs.stairs(
                h1[0] + h1_err,
                edges=h1[1],
                baseline=h1[0] - h1_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )

            # # ratio plot on the bottom
            # lims_min = 0.5
            # lims_max = 1.7
            # eps = 1e-5
            # x_nhits_range = [
            #     plt_config.occup_bins.min() - plt_config.occ_indent,
            #     400,
            #     950,
            #     plt_config.occup_bins.max() + plt_config.occ_indent,
            # ]

            # centers = np.array((h[1][:-1] + h[1][1:])/2)
            # x_mask = (centers >= x_nhits_range[j]) & (centers < x_nhits_range[j+1])

            # centers = centers[x_mask]
            # ratios = np.clip(
            #     np.array((h1[0] + eps) / (h[0] + eps)), lims_min, lims_max
            # )[x_mask]
            # mask = (ratios > lims_min) & (
            #     ratios < lims_max
            # )  # mask ratios within plotting y range
            #  # only connect dots with adjecent points
            # starts = np.argwhere(np.insert(mask[:-1], 0, False) < mask)[:, 0]
            # ends = np.argwhere(np.append(mask[1:], False) < mask)[:, 0] + 1
            # indexes = np.stack((starts, ends)).T
            # for idxs in indexes:
            #     sub_mask = np.zeros(len(mask), dtype=bool)
            #     sub_mask[idxs[0] : idxs[1]] = True
            #     axs[1].plot(
            #         centers[sub_mask],
            #         ratios[sub_mask],
            #         linestyle=":",
            #         lw=2,
            #         marker="o",
            #         color=plt_config.color_lines[i],
            #     )
            # #  remaining points either above or below plotting y range
            # mask = ratios == lims_min
            # axs[1].plot(
            #     centers[mask],
            #     ratios[mask],
            #     linestyle="",
            #     lw=2,
            #     marker="v",
            #     color=plt_config.color_lines[i],
            #     clip_on=False,
            # )
            # mask = (ratios == lims_max)
            # axs[1].plot(
            #     centers[mask],
            #     ratios[mask],
            #     linestyle="",
            #     lw=2,
            #     marker="^",
            #     color=plt_config.color_lines[i],
            #     clip_on=False,
            # )

    # horizontal line at 1
    # axs[1].axhline(1, linestyle='-', lw=1, color='k')
    # axs[1].axvline(x_nhits_range[1], linestyle='-', lw=2, color='k')
    # axs[1].axvline(x_nhits_range[2], linestyle='-', lw=2, color='k')

    axs.set_xlim(
        plt_config.occup_bins.min() - plt_config.occ_indent,
        plt_config.occup_bins.max() + plt_config.occ_indent,
    )
    # axs[1].set_ylim(lims_min, lims_max)
    plt.xlabel("number of hits")
    axs.set_ylabel("# showers")
    # axs[1].set_ylabel('ratio to G4')

    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    axs.legend(prop=plt_config.font, loc="best")
    # axs[0].text(315, 540, '10 GeV', fontsize=plt_config.font.get_size() + 2)
    # axs[0].text(870, 215, '50 GeV', fontsize=plt_config.font.get_size() + 2)
    # axs[0].text(1230, 170, '90 GeV', fontsize=plt_config.font.get_size() + 2)

    # y = 1.75
    # plt.text(190, y, '10 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(650, y, '50 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(1100, y, '90 GeV', fontsize=plt_config.font.get_size() + 2)

    plt.text(350, 600, "10 GeV", fontsize=plt_config.font.get_size() + 2)
    plt.text(750, 475, "50 GeV", fontsize=plt_config.font.get_size() + 2)
    plt.text(1150, 350, "90 GeV", fontsize=plt_config.font.get_size() + 2)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("occ_singleE.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def plt_esum_singleE(e_sum_list, e_sum_list_list, labels, plt_config=plt_config):
    fig, axs = plt.subplots(1, 1, figsize=(9, 9), sharex=False)

    # for legend ##########################################
    axs.hist(
        np.zeros(10), label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(e_sum_list)):
        axs.plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    ########################################################

    for j, (e_sum, e_sum_list) in enumerate(
        zip(e_sum_list, e_sum_list_list)
    ):  # loop over energyies
        h = axs.hist(
            np.array(e_sum),
            bins=plt_config.e_sum_bins,
            color="lightgrey",
            rasterized=True,
        )
        # h = axs.hist(
        #     np.array(e_sum),
        #     bins=plt_config.e_sum_bins,
        #     histtype="step",
        #     color="dimgrey",
        #     lw=2,
        # )

        # uncertainty
        h_err = np.sqrt(h[0])
        axs.stairs(
            h[0] + h_err,
            edges=h[1],
            baseline=h[0] - h_err,
            color="dimgrey",
            lw=2,
            hatch="///",
        )

        for i, e_sum_ in enumerate(e_sum_list):
            h1 = axs.hist(
                np.array(e_sum_),
                bins=h[1],
                histtype="step",
                linestyle="-",
                lw=2.5,
                color=plt_config.color_lines[i],
            )

            # uncertainty band in histogram
            h1_err = np.sqrt(h1[0])
            axs.stairs(
                h1[0] + h1_err,
                edges=h1[1],
                baseline=h1[0] - h1_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )

            # ratio plot on the bottom
            # lims_min = 0.5
            # lims_max = 2.0
            # eps = 1e-5
            # x_nhits_range = [
            #     plt_config.e_sum_bins.min() - plt_config.e_indent,
            #     500,
            #     1300,
            #     plt_config.e_sum_bins.max() + plt_config.e_indent,
            # ]

            # centers = np.array((h[1][:-1] + h[1][1:]) / 2)
            # x_mask = (centers >= x_nhits_range[j]) & (centers < x_nhits_range[j + 1])

            # centers = centers[x_mask]
            # ratios = np.clip(
            #     np.array((h1[0] + eps) / (h[0] + eps)), lims_min, lims_max
            # )[x_mask]
            # mask = (ratios > lims_min) & (
            #     ratios < lims_max
            # )  # mask ratios within plotting y range
            # # only connect dots with adjecent points
            # starts = np.argwhere(np.insert(mask[:-1], 0, False) < mask)[:, 0]
            # ends = np.argwhere(np.append(mask[1:], False) < mask)[:, 0] + 1
            # indexes = np.stack((starts, ends)).T
            # for idxs in indexes:
            #     sub_mask = np.zeros(len(mask), dtype=bool)
            #     sub_mask[idxs[0] : idxs[1]] = True
            #     axs[1].plot(
            #         centers[sub_mask],
            #         ratios[sub_mask],
            #         linestyle=":",
            #         lw=2,
            #         marker="o",
            #         color=plt_config.color_lines[i],
            #     )
            # # remaining points either above or below plotting y range
            # mask = ratios == lims_min
            # axs[1].plot(
            #     centers[mask],
            #     ratios[mask],
            #     linestyle="",
            #     lw=2,
            #     marker="v",
            #     color=plt_config.color_lines[i],
            #     clip_on=False,
            # )
            # mask = ratios == lims_max
            # axs[1].plot(
            #     centers[mask],
            #     ratios[mask],
            #     linestyle="",
            #     lw=2,
            #     marker="^",
            #     color=plt_config.color_lines[i],
            #     clip_on=False,
            # )

    # horizontal line at 1
    # axs[1].axhline(1, linestyle='-', lw=1, color='k')
    # axs[1].axvline(x_nhits_range[1], linestyle='-', lw=2, color='k')
    # axs[1].axvline(x_nhits_range[2], linestyle='-', lw=2, color='k')

    axs.set_xlim(
        plt_config.e_sum_bins.min() - plt_config.e_indent,
        plt_config.e_sum_bins.max() + plt_config.e_indent,
    )
    # axs[1].set_ylim(lims_min, lims_max)
    plt.xlabel("energy sum [MeV]")
    axs.set_ylabel("# showers")
    # axs[1].set_ylabel('ratio to G4')

    # y = 2.05
    # plt.text(150, y, '10 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(750, y, '50 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(1700, y, '90 GeV', fontsize=plt_config.font.get_size() + 2)

    # plt.text(320, 5, '10 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(880, 4.1, '50 GeV', fontsize=plt_config.font.get_size() + 2)
    # plt.text(1750, 3.4, '90 GeV', fontsize=plt_config.font.get_size() + 2)

    plt.text(320, 900, "10 GeV", fontsize=plt_config.font.get_size() + 2)
    plt.text(1000, 650, "50 GeV", fontsize=plt_config.font.get_size() + 2)
    plt.text(1800, 400, "90 GeV", fontsize=plt_config.font.get_size() + 2)

    # if plt_config.plot_legend_e:
    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    axs.legend(prop=plt_config.font, loc="best")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig("e_sum_singleE.pdf", dpi=100, bbox_inches="tight")
    plt.show()


def get_plots(
    plt_config,
    map_layers,
    half_cell_size_global,
    events,
    events_list: list,
    global_shower_axis_char: str,
    labels: list = ["1", "2", "3"],
    title=r"\textbf{full spectrum}",
):
    dict_real = get_features(
        plt_config, map_layers, half_cell_size_global, events, global_shower_axis_char
    )
    e_radial_real = dict_real["e_radial"]
    occ_real = dict_real["occ"]
    e_sum_real = dict_real["e_sum"]
    hits_real = dict_real["hits_noThreshold"]
    e_layers_real = dict_real["e_layers"]

    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = [], [], [], [], []

    for i in range(len(events_list)):
        dict_fake = get_features(
            plt_config,
            map_layers,
            half_cell_size_global,
            events_list[i],
            global_shower_axis_char,
        )

        e_radial_list.append(dict_fake["e_radial"])
        occ_list.append(dict_fake["occ"])
        e_sum_list.append(dict_fake["e_sum"])
        hits_list.append(dict_fake["hits_noThreshold"])
        e_layers_list.append(dict_fake["e_layers"])

    plt_radial(e_radial_real, e_radial_list, labels=labels, title=title)
    plt_spinal(e_layers_real, e_layers_list, labels=labels, title=title)
    plt_hit_e(hits_real, hits_list, labels=labels, title=title)
    plt_occupancy(occ_real, occ_list, labels=labels)
    plt_esum(e_sum_real, e_sum_list, labels=labels)


def get_observables_for_plotting(
    plt_config,
    map_layers,
    half_cell_size_global,
    events,
    events_list: list,
    global_shower_axis_char: str,
):
    dict_real = get_features(
        plt_config, map_layers, half_cell_size_global, events, global_shower_axis_char
    )
    e_radial_real = dict_real["e_radial"]
    occ_real = dict_real["occ"]
    e_sum_real = dict_real["e_sum"]
    hits_real = dict_real["hits_noThreshold"]
    e_layers_real = dict_real["e_layers"]
    e_layers_std_real = dict_real["e_layers_std"]

    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list, e_layers_std_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(len(events_list)):
        dict_fake = get_features(
            plt_config,
            map_layers,
            half_cell_size_global,
            events_list[i],
            global_shower_axis_char,
        )

        e_radial_list.append(dict_fake["e_radial"])
        occ_list.append(dict_fake["occ"])
        e_sum_list.append(dict_fake["e_sum"])
        hits_list.append(dict_fake["hits_noThreshold"])
        e_layers_list.append(dict_fake["e_layers"])
        e_layers_std_list.append(dict_fake["e_layers_std"])

    fakes_list = [
        e_radial_list,
        occ_list,
        e_sum_list,
        hits_list,
        e_layers_list,
        e_layers_std_list,
    ]

    real_list = [
        e_radial_real,
        occ_real,
        e_sum_real,
        hits_real,
        e_layers_real,
        e_layers_std_real,
    ]

    return real_list, fakes_list


def get_plots_from_observables(
    real_list: list,
    fakes_list: list,
    labels: list = ["1", "2", "3"],
    title=r"\textbf{full spectrum}",
    events=40_000,
):
    if len(real_list) != len(fakes_list):
        (
            e_radial_real,
            occ_real,
            e_sum_real,
            hits_real,
            e_layers_real,
            occ_layer_real,
            e_layers_distibution_real,
            e_radial_lists_real,
            hits_noThreshold_list_real,
        ) = real_list
    else:
        (
            e_radial_real,
            occ_real,
            e_sum_real,
            hits_real,
            e_layers_real,
            e_layers_std_real,
        ) = real_list

    (
        e_radial_list,
        occ_list,
        e_sum_list,
        hits_list,
        e_layers_list,
        e_layers_std_list,
    ) = fakes_list

    plt_radial(e_radial_real, e_radial_list, labels=labels, title=title, events=events)
    plt_spinal(
        e_layers_real,
        e_layers_list,
        e_layers_std_real,
        e_layers_std_list,
        labels=labels,
        title=title,
    )
    if len(real_list) != len(fakes_list):
        plt_hit_e(hits_noThreshold_list_real, hits_list, labels=labels, title=title)
    else:
        plt_hit_e(hits_real, hits_list, labels=labels, title=title)
    plt_occupancy(occ_real, occ_list, labels=labels)
    plt_esum(e_sum_real, e_sum_list, labels=labels)


def get_plots_from_observables_singleE(
    real_list_list: list, fakes_list_list: list, labels: list = ["1", "2", "3"]
):
    occ_real_list, occ_fake_list_list = [], []
    e_sum_real_list, e_sum_fake_list_list = [], []
    for i in range(len(real_list_list)):  # observables for certain single energy
        e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real = real_list_list[
            i
        ]
        occ_real_list.append(occ_real)
        e_sum_real_list.append(e_sum_real)

        e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = fakes_list_list[
            i
        ]
        occ_fake_list_list.append(occ_list)
        e_sum_fake_list_list.append(e_sum_list)

    plt_occupancy_singleE(occ_real_list, occ_fake_list_list, labels=labels)
    plt_esum_singleE(e_sum_real_list, e_sum_fake_list_list, labels=labels)


def plot_line_with_devation(
    ax,
    colour,
    xs,
    ys,
    ys_up_displacement,
    ys_down_displacement=None,
    clip_to_zero=False,
    **line_kwargs,
):
    """
    Plot a line with a shaded region around it.
    The shaded region usually represents the standard deviation of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    colour : str
        matplotlib colour to use for the line and shaded region.
    xs : array-like, 1D
        The x values to plot.
    ys : array-like, 1D
        The y values for the central line.
    ys_up_displacement : array-like, 1D
        The y values for the upper edge of the shaded region.
    ys_down_displacement : array-like, 1D, optional
        The y values for the lower edge of the shaded region.
        If not given, the same distance as for the upper edge is used.
    clip_to_zero : bool, optional
        If True, no line goes below 0.
    **line_kwargs
        Additional keyword arguments to pass to the `ax.plot` function
        of the central line.

    """
    if ys_down_displacement is None:
        ys_down_displacement = ys_up_displacement
    ys_low = ys - ys_down_displacement
    ys_high = ys + ys_up_displacement
    if clip_to_zero:
        ys = np.maximum(ys, 0)
        ys_low = np.maximum(ys_low, 0)
        ys_high = np.maximum(ys_high, 0)
    ax.plot(xs, ys, color=colour, **line_kwargs)
    ax.fill_between(xs, ys_low, ys_high, color=colour, alpha=0.2)


def plot_hist_with_devation(
        ax, colour, bins, counts, errors_up, errors_down=None,
        clip_to_zero=False, **hist_kwargs):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    reduced_args = hist_kwargs.copy()
    for key in ['color', 'histtype', 'weights']:
        if key in reduced_args:
            del reduced_args[key]
    ax.hist(
        bin_centers, bins=bins, color=colour, weights=counts,
        histtype='step', **reduced_args)
    
    if errors_down is None:
        errors_down = errors_up

    lower = counts - errors_down
    upper = counts + errors_up

    lower = np.repeat(lower, 2)
    upper = np.repeat(upper, 2)

    if clip_to_zero:
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)

    bin_corners = np.repeat(bins, 2)[1:-1]

    ax.fill_between(bin_corners, lower, upper, color=colour, alpha=0.2)


def heatmap(
    arr,
    x_spec,
    y_spec,
    x_label=None,
    y_label=None,
    cbar_label=None,
    title=None,
    ax=None,
    **pcolour_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    n_x_points, n_y_points = arr.shape
    if len(x_spec) == 2:
        # they are limits, transform to even range
        x_spec = np.linspace(*x_spec, n_x_points + 1)
    if len(y_spec) == 2:
        y_spec = np.linspace(*y_spec, n_y_points + 1)
    cmesh = ax.pcolormesh(x_spec, y_spec, arr.T, **pcolour_kwargs)
    cbar = fig.colorbar(cmesh)
    cbar.set_label(cbar_label)
    return fig, ax, cbar, cmesh


def heatmap_tile(arr, z, X_meshgrid, Y_meshgrid, ax, cmap=plt.cm.cool, alpha=0.5):
    """
    Plot a heatmap as a flat slice in a 3D plot

    Parameters
    ----------
    arr : 2D array
        The data to plot.
    z : float
        The z high to put it at
    Xmin : float
        The minimum x value.
    Ymin : float
        The minimum y value.
    Xstep_size : float
        The step size in x.
    Ystep_size : float
        The step size in y.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use.
    alpha : float, optional
        The alpha value for the heatmap.

    """

    Z_meshgrid = np.full_like(X_meshgrid, z)
    face_colors = cmap(arr)
    face_colors[:, :, -1] = alpha
    ax.plot_surface(
        X_meshgrid,
        Y_meshgrid,
        Z_meshgrid,
        facecolors=face_colors,
        linewidth=0,
    )


def heatmap_stack(
    arrs,
    zs,
    Xmin=0,
    Xmax=None,
    Ymin=0,
    Ymax=None,
    ax=None,
    cmap=plt.cm.cool,
    alpha=0.5,
):
    """
    Plot a series of heatmaps as flat slices in a 3D plot

    Parameters
    ----------
    arrs : 3D array
        The data to plot in a grid of (x, y).
    zs : float
        The z high to put each slice at
    Xmin : float, optional
        The minimum x value.
        Default is 0.
    Xmax : float, optional
        The maximum x value.
        Default is length of the second dimension of arrs.
    Ymin : float, optional
        The minimum y value.
        Default is 0.
    Ymax : float, optional
        The maximum y value.
        Default is length of the third dimension of arrs.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use.
    alpha : float, optional
        The alpha value for the heatmap.

    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)

    if Xmax is None:
        Xmax = arrs.shape[1]
    if Ymax is None:
        Ymax = arrs.shape[2]

    X_tile_centers = np.linspace(Xmin, Xmax, arrs.shape[1])
    Y_tile_centers = np.linspace(Ymin, Ymax, arrs.shape[2])
    X_meshgrid, Y_meshgrid = np.meshgrid(X_tile_centers, Y_tile_centers, indexing="ij")

    for z, arr, alph in zip(zs, arrs, alpha):
        heatmap_tile(arr, z, X_meshgrid, Y_meshgrid, ax, cmap, alph)

    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    z_padding = (max(zs) - min(zs)) * 0.1
    ax.set_zlim(min(zs) - z_padding, max(zs) + z_padding)
    ax.get_figure().canvas.draw()


def plot_event(
    event_n,
    cond_features,
    event,
    energy_scale=1,
    xy_scale=1,
    xlim=(-150, 150),
    ylim=(-150, 150),
    ax=None,
):
    """
    Plot a event in 3D

    Parameters
    ----------
    event_n : int
        The event number, for the title.
    cond_features : float
        The conditioning features of the event, also for the title.
    event : np.array (n_points, 4)
        The event to plot. The columns are (x, y, z, e).
    energy_scale : float, optional
        Scale factor to apply to energy before plotting.
        Default is 1.
    xz_scale : float, optional
        Scale factor to apply to x and z before plotting.
        Default is 1.
    xlim : tuple, optional
        The limits for the x axis.
        Default is (-150, 150).
    ylim : tuple, optional
        The limits for the y axis.
        Default is (-150, 150).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
        If not given, a new figure is created.
        Default is None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The 3D axes the event was plotted on.

    """
    energy = event[:, 3]
    mask = energy > 0
    n_points = sum(mask)
    energy = energy[mask] * energy_scale
    xs = event[mask, 0] * xy_scale
    ys = event[mask, 1] * xy_scale
    zs = event[mask, 2]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=energy, s=energy, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # write a title
    observed_energy = energy.sum()
    formated_conditioning = ", ".join(f"{f:.2f}" for f in cond_features)
    ax.set_title(
        f"Evt: {event_n}, $n_{{pts}}$: {n_points}, conditioning: "
        + formated_conditioning +
        f"$E_{{vis}}$: {observed_energy:.2f}"
    )
    return ax


def plot_model(incident_energy, model):
    """
    Using the standrd plotting function to plot the model output

    Parameters
    ----------
    incident_energy : float
        The incident energy to generate the event with.
    model : Model
        The model to generate the event with.
        Must have an `inference` method that
        takes an incident energy and returns an event.
        The event should have shape (n_points, 4) where the columns are (x, y, z, e).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The 3D axes the event was plotted on.

    """
    xs, ys, zs, es = model.inference(incident_energy)
    event = np.empty((len(xs), 4))
    event[:, 0] = xs
    event[:, 1] = ys
    event[:, 2] = zs
    event[:, 3] = es
    return plot_event("Modeled", incident_energy, event, energy_scale=1.0, x_scale=150)


def blank_axes(ax):
    """
    Remove the ticks and labels, and lines from the axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to blank.

    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_frame_on(False)


def project_if_needed(element, element_type, sequence_len):
    is_elem = isinstance(element, element_type)
    if not is_elem:
        assert (
            len(element) == sequence_len
        ), f"Expected length {sequence_len}, but see {len(element)}; {element}"
        return element
    return [element] * sequence_len


class RatioPlots:
    def __init__(
        self,
        x_labels,
        truth_data,
        truth_label="g4",
        truth_color=(0.5, 0.5, 0.5, 0.5),
        max_cols=3,
        n_bins=50,
        x_ranges=None,
        logx=False,
        logy=False,
    ):
        # set up plot axes
        self.n_features = len(x_labels)
        self.n_cols = min(self.n_features, max_cols)
        self.n_rows = int(self.n_features / self.n_cols)
        height_ratios = [3, 1] * self.n_rows
        self.fig, self.axes = plt.subplots(
            2 * self.n_rows,
            self.n_cols,
            figsize=(20, 5 * self.n_rows),
            gridspec_kw={"height_ratios": height_ratios},
        )
        if self.n_cols == 1:
            self.axes = self.axes[:, None]
        # ylabels to the left
        for row in range(self.n_rows):
            ax = self.axes[row * 2, 0]
            ax.set_ylabel("Counts")

        # calculate bin locations
        n_bins = project_if_needed(n_bins, int, self.n_features)
        self.logx = project_if_needed(logx, bool, self.n_features)
        self.logy = project_if_needed(logy, bool, self.n_features)
        if x_ranges is None:
            x_ranges = [(np.min(d), np.max(d)) for d in truth_data]
        elif not hasattr(x_ranges[0], "__iter__"):
            x_ranges = [x_ranges] * self.n_features

        self.bins = []
        for i, (x_min, x_max) in enumerate(x_ranges):
            if self.logx[i] and x_max > 0:
                if x_min <= 0:
                    x_min = 0.0001
                here = np.logspace(np.log10(x_min), np.log10(x_max), n_bins[i])
            else:
                here = np.linspace(x_min, x_max, n_bins[i])
            self.bins.append(here)

        # plot truth hists, and record their counts
        truth_hist_kwargs = dict(
            label=truth_label, histtype="stepfilled", color=truth_color
        )
        self.truth_counts = []

        for i, label in enumerate(x_labels):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            main_ax = self.axes[row * 2, col]
            main_ax.set_xlabel(label)
            n, _, _ = main_ax.hist(
                truth_data[i], bins=self.bins[i], **truth_hist_kwargs
            )
            self.truth_counts.append(n)
        del truth_data

        # prep for plotting ratios
        self.ratio_min_maxes = [[1.0, 1.0] for _ in range(self.n_features)]
        self.bin_centers = [0.5 * (b[1:] + b[:-1]) for b in self.bins]
        for i, (x_min, x_max) in enumerate(x_ranges):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.hlines(1, x_min, x_max, color=truth_color)

    def add_comparison(self, data, label, colour):
        model_hist_kwargs = dict(label=label, histtype="step", color=colour)
        model_ratio_kwargs = dict(label=label, c=colour)
        for i in range(self.n_features):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            main_ax = self.axes[row * 2, col]
            n, _, _ = main_ax.hist(data[i], bins=self.bins[i], **model_hist_kwargs)
            ratio = n / self.truth_counts[i]
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.plot(self.bin_centers[i], ratio, **model_ratio_kwargs)
            self.ratio_min_maxes[i][0] = min(
                self.ratio_min_maxes[i][0], np.nanmin(ratio)
            )
            self.ratio_min_maxes[i][1] = max(
                self.ratio_min_maxes[i][1], np.nanmax(ratio)
            )

    def finalise(self):
        self.axes[-2, -1].legend()
        for i, (y_min, y_max) in enumerate(self.ratio_min_maxes):
            clipped_min = min(y_min, 0.0)
            clipped_max = min(y_max, 2.0)
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.set_ylim(clipped_min, clipped_max)
            ratio_ax.set_xlim(self.bins[i][0], self.bins[i][-1])
            main_ax = self.axes[row * 2, col]
            main_ax.set_xlim(self.bins[i][0], self.bins[i][-1])
        self.fig.tight_layout()

