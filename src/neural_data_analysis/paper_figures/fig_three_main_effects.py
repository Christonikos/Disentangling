#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
"""
---------------------------------------------------------
## FIRST ORDER EFFECTS 
---------------------------------------------------------

This script generates figures for the first order effects in neural data analysis.

This corresponds to the fig.3 of the paper.

The script analyzes the main effects of Violation, Congruency, and Transition
in neural data. It processes input arguments to configure the analysis, collects
scores from subjects, and generates plots to visualize the effects.

Functions:
----------
- gn_parsing_object: Determines the distractor type based on grammatical number.
- response_parsing_object: Determines the response type based on response category.
- length_parsing_object: Determines the length of the epoch based on cropping.
- baseline_parsing_object: Determines whether baseline correction is applied.
- ssp_parsing_object: Determines whether SSP cleaning is applied.
- grid_search_parsing_object: Determines whether grid search is applied.
- collect_scores: Collects scores from all subjects for a given construction and effect.
- make_figs_path: Creates the path for saving figures.
- diagonal_cluster_test: Performs a permutation cluster test on diagonals.
"""
# ============================================================================
# modules
import sys

sys.path.append("../")

# Standard library imports
import argparse

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from mne.stats import permutation_cluster_1samp_test
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local application imports
import config as c
from repos import func_repo as f


fdr = sm.stats.fdrcorrection
# =============================================================================
# ARGPARSE INPUTS
# =============================================================================
# Parsing INPUTS
parser = argparse.ArgumentParser(description=" ")
parser.add_argument(
    "-eoi",
    "--events_of_interest",
    nargs="+",
    default=["first_word_onset"],
    help="Select events to epoch.",
)
parser.add_argument(
    "-rt",
    "--response_type",
    default="all",
    help="Parse epochs based on reponse type.\
                        Options: correct, false, all",
)
parser.add_argument(
    "-dn",
    "--distractor_number",
    default="all",
    help="Parse epochs based on the \
                        grammatical number of the distractor.\
                        Options: all, sing, plur. \
                        This is the grammatical number of the First Noun.",
)
parser.add_argument(
    "-sensor",
    "--sensor_type",
    default="all",
    help="Select the sensor type to use for decoding \
                        Options: all, meg, mag, grad, eeg. \
                        all: Magnetometers, Gradiometeres and EEG.\
                        meg: Magnetometers and Gradiometeres.\
                        mag: Magnetometers.\
                        grad:Gradiometers.",
)
parser.add_argument(
    "-data",
    "--data_type",
    default="preprocessed",
    help="Select the data type to use for decoding \
                        Options: raw, preprocessed. \
                        raw: Maxwell filtered and EEG interpolated.\
                        preprocessed: raw + IQR clipped + Smoothed + Detrended.",
)
parser.add_argument(
    "-baseline",
    "--baseline",
    default="yes",
    help="Select whether to apply baseline",
)
parser.add_argument(
    "-crop",
    "--crop",
    default="no",
    help="Select whether to crop around the target onset",
)
parser.add_argument(
    "-reject", "--reject", default="no", help="Select whether to reject epochs"
)
parser.add_argument(
    "-ssp",
    "--ssp",
    default="yes",
    help="Select whether to load SSP cleaned epochs",
)
parser.add_argument("-tmin", "--tmin", default=-0.5, help="Cropping tmin")
parser.add_argument("-tmax", "--tmax", default=1.5, help="Cropping tmax")
parser.add_argument(
    "-str", "--structure", default="pp", help="Options: pp, obj, sem"
)
parser.add_argument(
    "-roi",
    "--roi",
    default="whole_coverage",
    help="Options: whole_coverage, left_frontal, right_frontal, left_parietal\
                        right_parietal, left_occipital, right_occipital",
)
parser.add_argument(
    "-grid",
    "--grid",
    default="no",
    help="Options: yes/no Whether to see results generated with grid search or not",
)
parser.add_argument("-smooth", "--smooth", default="yes")
args = parser.parse_args()

root_dir = "first_order_effects"


# =============================================================================
# Select distractor grammatical number
# =============================================================================
def gn_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the grammatical number of the distractor from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: The type of distractor based on the grammatical number.
    """
    if args.distractor_number == "all":
        distractor_type = "both_distractors"
    elif args.distractor_number == "sing":
        distractor_type = "singular_distractor"
    elif args.distractor_number == "plur":
        distractor_type = "plural_distractor"

    return distractor_type


# =============================================================================
# Select RESPONSE TYPE
# =============================================================================
def response_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the response type from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: The type of response based on the response type.
    """
    if args.response_type == "all":
        response_type = "all_responses"
    elif args.response_type == "correct":
        response_type = "correct_responses"
    elif args.response_type == "false":
        response_type = "false_responses"

    return response_type


# =============================================================================
# Select LENGTH
# =============================================================================
def length_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the length of the epoch based on cropping from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: The length of the epoch based on cropping.
    """
    if args.crop == "yes":
        length = "cropped_around_target"
    else:
        length = "whole_sentence"

    return length


# ============================================================================
# Select BASELINE
# =============================================================================
def baseline_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the baseline correction option from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: 'with_baseline' if baseline correction is applied, otherwise 'without_baseline'.
    """
    if args.baseline == "yes":
        baseline = "with_baseline"
    else:
        baseline = "without_baseline"

    return baseline


# =============================================================================
# Select SSP
# =============================================================================
def ssp_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the SSP (Signal Space Projection) option from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: 'with_ssp' if SSP is applied, otherwise 'without_ssp'.
    """
    if args.ssp == "yes":
        ssp = "with_ssp"
    elif args.ssp == "no":
        ssp = "without_ssp"
    return ssp


# =============================================================================
# Select GRID SEARCH
# =============================================================================
def grid_search_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse the grid search option from the arguments.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.

    Returns:
    str: 'with_grid_search' if grid search is applied, otherwise 'without_grid_search'.
    """
    if args.grid == "yes":
        grid = "with_grid_search"
    elif args.grid == "no":
        grid = "without_grid_search"
    return grid


# =============================================================================
# Collect scores from all subjects per construction, effect and SOA
# =============================================================================
def collect_scores(path, construction, effect):
    """
    Parameters
    ----------
    path : object (c.path)
    soa : Options [125, 250, 325, 500] (c.soas) #comes from another project. No multiple SOAS in
    this project.
    construction : Options [pp, obj] (c.constructions)
    effect : Options (c.first_order_effects or c.second_order_effects)

    Returns
    -------
    all_diagonals : List that contains the diag of the GAT. len(list)=#subjects
    all_scores : List that contains the GAT. len(list)=#subjects
    error : The SEM of the diagonals

    """

    all_diagonals, all_scores = ([] for i in range(0, 2))
    for subject in c.subjects_list:
        path2scores = c.join(
            c.project_path,
            "Output",
            args.roi,
            grid_search_parsing_object(args),
            root_dir,
            "Decoding",
            construction,
            effect,
            args.sensor_type,
            args.data_type,
            subject,
            length_parsing_object(args),
            ssp_parsing_object(args),
            baseline_parsing_object(args),
            response_parsing_object(args),
            gn_parsing_object(args),
        )

        scores = np.load(
            c.join(
                path2scores,
                [file for file in c.see(path2scores) if "scores" in file][0],
            )
        )
        all_diagonals.append(np.diag(scores))
        all_scores.append(scores)

    # returns the SEM of the diagonals across subjects
    error = stats.sem(all_scores, axis=0)

    # return the times vector
    path_in = c.join(
        c.project_path,
        "Output",
        args.roi,
        grid_search_parsing_object(args),
        root_dir,
        "Decoding",
        construction,
        effect,
        args.sensor_type,
        args.data_type,
        "mean_scores",
        length_parsing_object(args),
        ssp_parsing_object(args),
        baseline_parsing_object(args),
        response_parsing_object(args),
        gn_parsing_object(args),
    )

    times = np.load(
        c.join(
            path_in, [file for file in c.see(path_in) if "times" in file][0]
        )
    )

    return all_diagonals, all_scores, error, times


# =============================================================================
# Create the figures path
# =============================================================================
def make_figs_path(construction: str) -> str:
    """
    Creates the path for saving figures of decoding results.

    Parameters
    ----------
    construction : str
        The construction type (e.g., 'pp', 'obj').

    Returns
    -------
    str
        The full path to the figure file.
    """
    # General path for decoding results
    path2figs = c.join(
        c.figures_path,
        "decoding_results",
        args.roi,
        "first_order_effects",
        construction,
        effect,
        args.sensor_type,
        args.data_type,
        "mean_scores",
        length_parsing_object(args),
        ssp_parsing_object(args),
        baseline_parsing_object(args),
        response_parsing_object(args),
        gn_parsing_object(args),
    )

    # Make path if it does not exist
    if not c.exists(path2figs):
        c.make(path2figs)

    # Full path to the figure file
    fname = c.join(path2figs, "first_order_effects.pdf")

    return fname


# =============================================================================
# Permutation cluster test
# =============================================================================


def diagonal_cluster_test(all_diagonals: list) -> tuple:
    """
    Perform a permutation cluster test on the provided diagonals.

    Parameters
    ----------
    all_diagonals : list
        A list of arrays where each array represents the diagonal scores for a subject.

    Returns
    -------
    tuple
        A tuple containing:
        - cluster_p_values: The p-values for each identified cluster.
        - clusters: The clusters identified in the permutation test.

    Notes
    -----
    - The function first converts the input list to a numpy array.
    - A cluster threshold is set using a t-distribution with a p-value threshold of 0.01.
    - The permutation cluster test is performed with 1000 permutations.
    - The function currently does not apply FDR correction to the cluster p-values.
    """

    # Convert the list of diagonals to a numpy array
    all_diagonals = np.array(all_diagonals)

    # ~~~~~~~~~~~~~~~~~~~
    ## Set threshold
    # ~~~~~~~~~~~~~~~~~~~
    # Set the p-value threshold for cluster formation
    p_threshold = 0.01

    # Number of subjects in the study
    n_subjects = len(c.subjects_list)

    # Calculate the t-distribution threshold for the given p-value and number of subjects
    thres = -stats.distributions.t.ppf(p_threshold / 2.0, n_subjects - 1)

    # Set threshold to None to use default behavior in permutation_cluster_1samp_test
    thres = None

    # ~~~~~~~~~~~~~~~~~~~
    ## Compute statistic
    # ~~~~~~~~~~~~~~~~~~~
    # Perform the permutation cluster test
    _, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        all_diagonals - 0.5,
        n_permutations=1000,
        threshold=thres,
        tail=1,
        n_jobs=-1,
        verbose=False,
        seed=42,
        out_type="mask",
    )

    # Correct with FDR (currently commented out)
    # _, cluster_p_values = fdr(cluster_p_values)

    return cluster_p_values, clusters


# %%
# =============================================================================
# FIGURE CONSTRUCTION
# =============================================================================
ticks = [
    r"$w_{1}$",
    r"$w_{2}$",
    r"$w_{3}$",
    r"$w_{4}$",
    r"$w_{5}$",
    r"$w_{6}$",
    r"$w_{7}$",
    "",
    "",
    "",
]
fig = plt.figure(dpi=100, facecolor="w", edgecolor="w")
fig.set_size_inches(12, 4)
labelling = ["A.", "C.", "B."]
titles = [
    r"$\mathcal{Violation}$",
    r"$\mathcal{Congruency}$",
    r"$\mathcal{Transition}$",
]
colors = ["darkblue", "darkgreen", "darkred"]


for idx, effect in enumerate(c.first_order_effects):
    if effect == "main_effect_of_violation" and args.crop == "yes":
        args.grid = "yes"
    else:
        args.grid = "no"

    diagonals, errors, mean_diagonals = [{} for i in range(0, 3)]
    for construction in c.constructions:
        # collect the scores
        diagonals[construction], _, errors[construction], times = (
            collect_scores(c.path, construction, effect)
        )
        mean_diagonals[construction] = np.mean(diagonals[construction], axis=0)

    if args.smooth == "yes":
        width_sec = 0.1  # Gaussian-kernal width in [sec]
        for construction in c.constructions:
            smoothed = gaussian_filter1d(
                mean_diagonals[construction], width_sec * 100
            )
            mean_diagonals[construction] = smoothed
            smoothed_errors = gaussian_filter1d(
                errors[construction], width_sec * 100
            )
            errors[construction] = smoothed_errors
    # times
    # tmin=-0.5
    # tmax =1.5
    # times=np.linspace(-0.5, 1.5, len(times))
    # times=times+0.016

    if effect == "main_effect_of_violation":
        peaks = {}
        peaks["pp_syntax"] = round(np.max(mean_diagonals["pp_syntax"]), 2)
        peaks["objrc_syntax"] = round(
            np.max(mean_diagonals["objrc_syntax"]), 2
        )
        peaks["pp_semantics"] = round(
            np.max(mean_diagonals["pp_semantics"]), 2
        )

        ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax0.plot(
            times,
            mean_diagonals["pp_syntax"],
            color=colors[0],
            label=r"$\mathcal{PP-Number}$",
        )
        ax0.fill_between(
            times,
            mean_diagonals["pp_syntax"] - np.diag(errors["pp_syntax"]),
            mean_diagonals["pp_syntax"] + np.diag(errors["pp_syntax"]),
            color=colors[0],
            alpha=0.12,
            label="_nolegend_",
        )
        ax0.plot(
            times,
            mean_diagonals["objrc_syntax"],
            color=colors[1],
            label=r"$\mathcal{ObjRC-Number}$",
        )
        ax0.fill_between(
            times,
            mean_diagonals["objrc_syntax"] - np.diag(errors["objrc_syntax"]),
            mean_diagonals["objrc_syntax"] + np.diag(errors["objrc_syntax"]),
            color=colors[1],
            alpha=0.12,
            label="_nolegend_",
        )
        ax0.plot(
            times,
            mean_diagonals["pp_semantics"],
            color=colors[2],
            label=r"$\mathcal{PP-Animacy}$",
        )
        ax0.fill_between(
            times,
            mean_diagonals["pp_semantics"] - np.diag(errors["pp_semantics"]),
            mean_diagonals["pp_semantics"] + np.diag(errors["pp_semantics"]),
            color=colors[2],
            alpha=0.12,
            label="_nolegend_",
        )
        # plt.text(-0.127, 1.14, labelling[idx], fontweight="bold",
        #         transform=ax0.transAxes, size=14)
        # plt.title(titles[idx])
        plt.axvline(2.5, color="k", alpha=0.3, label=r"$\mathcal{Target}$")

        # ax0.legend(loc='upper center', bbox_to_anchor=(0.45, -0.5),
        #   fancybox=True, shadow=False, ncol=4)
        positionsing = [0.7, 0.75, 0.65]

        (
            diagonals,
            errors,
            mean_diagonals,
            cluster_p_values,
            clusters,
            max_auc,
        ) = [{} for i in range(0, 6)]
        for construction in c.constructions:
            # collect the scores
            diagonals[construction], _, errors[construction], times = (
                collect_scores(c.path, construction, effect)
            )
            mean_diagonals[construction] = np.mean(
                diagonals[construction], axis=0
            )
            max_auc[construction] = np.round(
                np.max(mean_diagonals[construction]), 2
            )
            # GET THE SIGNIFICANCE CLUSTERS
            (
                cluster_p_values[construction],
                clusters[construction],
            ) = diagonal_cluster_test(diagonals[construction])
        # times=np.linspace(-0.5, 1.5, len(times))
        ## PLOT SIGNIFICANCE

        starting_times = []

        for idx, construction in enumerate(c.constructions):
            significance_time = []
            for i_clust, cluster in enumerate(clusters[construction]):
                if cluster_p_values[construction][i_clust] < 0.05:
                    significance_time.append(times[cluster])

                if significance_time:
                    starting_times.append(significance_time[0][0])
                    if len(significance_time) == 1:
                        plt.axvspan(
                            significance_time[0][0],
                            significance_time[0][-1],
                            positionsing[idx] + 0.15,
                            positionsing[idx] + 0.18,
                            color=colors[idx],
                            alpha=0.3,
                        )
                    else:
                        for i in range(0, len(significance_time)):
                            plt.axvspan(
                                significance_time[i][0],
                                significance_time[i][-1],
                                positionsing[idx] + 0.15,
                                positionsing[idx] + 0.18,
                                color=colors[idx],
                                alpha=0.3,
                            )

        onset_times = np.unique(np.array(starting_times) - 2.5)
        plt.title("              ")

    elif effect == "main_effect_of_transition":
        ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
        ax1.plot(times, mean_diagonals["pp_syntax"], color=colors[0])
        ax1.fill_between(
            times,
            mean_diagonals["pp_syntax"] - np.diag(errors["pp_syntax"]),
            mean_diagonals["pp_syntax"] + np.diag(errors["pp_syntax"]),
            color=colors[0],
            alpha=0.12,
            label="_nolegend_",
        )
        ax1.plot(times, mean_diagonals["objrc_syntax"], color=colors[1])
        ax1.fill_between(
            times,
            mean_diagonals["objrc_syntax"] - np.diag(errors["objrc_syntax"]),
            mean_diagonals["objrc_syntax"] + np.diag(errors["objrc_syntax"]),
            color=colors[1],
            alpha=0.12,
            label="_nolegend_",
        )
        ax1.plot(times, mean_diagonals["pp_semantics"], color=colors[2])
        ax1.fill_between(
            times,
            mean_diagonals["pp_semantics"] - np.diag(errors["pp_semantics"]),
            mean_diagonals["pp_semantics"] + np.diag(errors["pp_semantics"]),
            color=colors[2],
            alpha=0.12,
            label="_nolegend_",
        )
        # plt.text(-0.3, 1.14, labelling[idx], fontweight="bold",
        #         transform=ax1.transAxes, size=14)
        plt.title("              ")
        # plt.tight_layout()

    elif effect == "main_effect_of_congruency":
        ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
        ax2.plot(times, mean_diagonals["pp_syntax"], color=colors[0])
        ax2.fill_between(
            times,
            mean_diagonals["pp_syntax"] - np.diag(errors["pp_syntax"]),
            mean_diagonals["pp_syntax"] + np.diag(errors["pp_syntax"]),
            color=colors[0],
            alpha=0.12,
            label="_nolegend_",
        )
        ax2.plot(times, mean_diagonals["objrc_syntax"], color=colors[1])
        ax2.fill_between(
            times,
            mean_diagonals["objrc_syntax"] - np.diag(errors["objrc_syntax"]),
            mean_diagonals["objrc_syntax"] + np.diag(errors["objrc_syntax"]),
            color=colors[1],
            alpha=0.12,
            label="_nolegend_",
        )
        ax2.plot(times, mean_diagonals["pp_semantics"], color=colors[2])
        ax2.fill_between(
            times,
            mean_diagonals["pp_semantics"] - np.diag(errors["pp_semantics"]),
            mean_diagonals["pp_semantics"] + np.diag(errors["pp_semantics"]),
            color=colors[2],
            alpha=0.12,
            label="_nolegend_",
        )
        # ax2.axes.get_yaxis().set_visible(False)
        # plt.text(-0.3, 1.14, labelling[idx], fontweight="bold",
        #         transform=ax2.transAxes, size=14)
        plt.title("              ")
        # plt.tight_layout()

    tick_values = np.arange(0, 5, 0.5)
    plt.xticks(tick_values, ticks)

    plt.ylabel("              ", size=12)
    plt.xlabel("              ", size=12)
    plt.ylim(0.4, 0.71)
    plt.axhline(0.5, linestyle="--", color="black", label="_nolegend_")
    plt.axvline(2.5, color="k", alpha=0.3, label="target")
    sns.despine(offset=10, trim=False)
    plt.tight_layout()


# plt.tight_layout()
fig.savefig(
    "first_order_effects_human_data.png",
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=1200,
)
plt.show()


# %% Make a legend here

# fig=plt.figure(dpi=400, facecolor='w', edgecolor='w')
# fig.set_size_inches(8,5)
# plt.plot(0,0, color='darkblue', linestyle='-', label='PP-Number')
# plt.plot(0,0, color='darkgreen', linestyle='-', label='Objrc-Number')
# plt.plot(0,0, color='darkred', linestyle='-', label='PP-Animacy')
# plt.plot(0,0, color='dimgrey', linestyle='-', alpha=0.3, label='Target onset')
# plt.plot(0,0, color='black', linestyle='--', label='Chance')
# plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.5),
#   fancybox=True, shadow=False, ncol=5)
# plt.axis('off')
# plt.tight_layout()
# fig.savefig('empty_legend.png',bbox_inches='tight', pad_inches=0.2, dpi=400)

# plt.show()
