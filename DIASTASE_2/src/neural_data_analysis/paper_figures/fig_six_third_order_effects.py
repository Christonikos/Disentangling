#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
"""
---------------------------------------------------------
## THIRD ORDER EFFECTS 
---------------------------------------------------------

This script generates figures for the third order effects in neural data analysis.

This corresponds to the fig.6 of the paper.


The script analyzes the interactions between different conditions in neural data.
It processes input arguments to configure the analysis, collects scores from subjects,
and generates plots to visualize the third order effects.

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
    default="sing",
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
    default="yes",
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
    default="yes",
    help="Options: yes/no Whether to see results generated with grid search or not",
)
parser.add_argument("-smooth", "--smooth", default="yes")
args = parser.parse_args()

root_dir = "third_order_effects"
effect = "number_effects"


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
def collect_scores(path, construction, effect, condition):
    """
    Parameters
    ----------
    path : object (c.path)
    soa : Options [125, 250, 325, 500] (c.soas)
    construction : Options [pp, obj] (c.constructions)
    effect : Options (c.first_order_effects or c.second_order_effects)

    Returns
    -------
    all_diagonals : List that contains the diag of the GAT. len(list)=#subjects
    all_scores : List that contains the GAT. len(list)=#subjects
    error : The SEM of the diagonals

    """

    all_diagonals = []
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
            condition,
        )

        diag = np.load(
            c.join(
                path2scores,
                [file for file in c.see(path2scores) if "scores" in file][0],
            )
        )
        if np.isnan(diag).any():
            continue
        all_diagonals.append(diag)

    # returns the SEM of the diagonals across subjects
    error = stats.sem(all_diagonals, axis=0)

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
        "P01",
        length_parsing_object(args),
        ssp_parsing_object(args),
        baseline_parsing_object(args),
        response_parsing_object(args),
        gn_parsing_object(args),
        "all",
    )

    times = np.load(
        c.join(
            path_in, [file for file in c.see(path_in) if "times" in file][0]
        )
    )

    return all_diagonals, error, times


# =============================================================================
# Create the figures path
# =============================================================================
def make_figs_path(construction: str) -> str:
    """
    Create the path for saving figures based on the provided construction.

    This function generates a path for decoding results and ensures the path exists.
    It then returns the full path to the PDF file where the second order effects
    will be saved.

    Parameters:
    construction (str): The type of construction for which the figures are being generated.

    Returns:
    str: The full path to the PDF file where the second order effects will be saved.
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
    # Full path to the PDF file
    fname = c.join(path2figs, "second_order_effects.pdf")

    return fname


# =============================================================================
# Return Title and color per Construction
# =============================================================================
def color_and_title(construction: str) -> tuple[str, str]:
    """
    Return the title and color associated with a given construction.

    Parameters:
    construction (str): The type of construction. Options are "pp_syntax", "objrc_syntax", and "pp_semantics".

    Returns:
    tuple[str, str]: A tuple containing the title and color associated with the construction.
    """
    if construction == "pp_syntax":
        title = r"$\mathcal{PP-Number}$"
        color = "darkblue"
    elif construction == "objrc_syntax":
        color = "darkgreen"
        title = r"$\mathcal{ObjRC-Number}$"
    elif construction == "pp_semantics":
        color = "darkred"
        title = r"$\mathcal{PP-Animacy}$"
    return title, color


# =============================================================================
# Permutation cluster test
# =============================================================================


def diagonal_cluster_test(
    all_diagonals: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Perform a permutation cluster test on the provided diagonals.

    This function computes the cluster-level statistics for the given diagonals
    using a permutation test. It returns the p-values for each cluster and the
    clusters themselves.

    Parameters:
    all_diagonals (np.ndarray): A 2D array where each row represents a subject's
                                diagonal data.

    Returns:
    tuple[np.ndarray, list[np.ndarray]]: A tuple containing:
        - cluster_p_values (np.ndarray): An array of p-values for each identified cluster.
        - clusters (list[np.ndarray]): A list of arrays, each representing a cluster mask.
    """
    # ~~~~~~~~~~~~~~~~~~~
    ## Set threshold
    # ~~~~~~~~~~~~~~~~~~~
    # Set cluster threshold
    p_threshold = 0.01
    n_subjects = len(c.subjects_list)
    thres = -stats.distributions.t.ppf(p_threshold / 2.0, n_subjects - 1)
    thres = None  # No threshold applied

    # ~~~~~~~~~~~~~~~~~~~
    ## Compute statistic
    # ~~~~~~~~~~~~~~~~~~~
    fvals, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        all_diagonals - 0,
        n_permutations=1000,
        threshold=thres,
        tail=0,
        n_jobs=-1,
        verbose=False,
        seed=42,
        out_type="mask",
    )
    # Correct with FDR (commented out)
    # _, cluster_p_values = fdr(cluster_p_values)

    return cluster_p_values, clusters


# %%
labelling = ["Congruent", "Incongruent", "Target"]
colors = ["darkblue", "darkgreen", "darkred"]


fig = plt.figure(dpi=100, facecolor="w", edgecolor="w")
fig.set_size_inches(12, 4)

difference = {}
timing = {}

lines = []
line_labels = [r"$congruent$", r"$incongruent$", r"$p<0.05$"]

if args.distractor_number == "sing":
    sup_title = r"$Singular \quad attractor$"
else:
    sup_title = r"$Plural \quad attractor$"

for idx, construction in enumerate(c.constructions):
    timing[construction] = []
    l = plt.subplot(1, 3, idx + 1)
    lines.append(l)
    ##########################
    # collect the scores
    ##########################
    diagonals, errors = ({} for i in range(0, 2))
    ## CONGRUENT
    diagonals["congruent"], errors["congruent"], times = collect_scores(
        c.path, construction, effect, "congruent"
    )
    ## INCONGRUENT
    diagonals["incongruent"], errors["incongruent"], times = collect_scores(
        c.path, construction, effect, "incongruent"
    )

    title, color = color_and_title(construction)

    difference[construction] = np.array(diagonals["congruent"]) - np.array(
        diagonals["incongruent"]
    )

    ## times used for the cropping (should probably soft-code later)
    tmin = -0.5
    tmax = 1.5
    times = np.linspace(-0.5, 1.5, len(times))
    times = times + 0.016

    ##################################
    # DIAGONAL
    ##################################
    ## CONGRUENT
    mean_diagon_congruent = np.mean(diagonals["congruent"], axis=0)
    if args.smooth == "yes":
        width_sec = 0.05  # Gaussian-kernal width in [sec]
        mean_diagon_congruent = gaussian_filter1d(
            mean_diagon_congruent, width_sec * 100
        )
    plt.plot(
        times,
        mean_diagon_congruent,
        color=color,
        label="_nolegend_",
        linestyle="-",
    )
    plt.fill_between(
        times,
        mean_diagon_congruent - errors["congruent"],
        mean_diagon_congruent + errors["congruent"],
        color=color,
        alpha=0.12,
        label="_nolegend_",
    )

    ## INCONGRUENT
    mean_diagon_incongruent = np.mean(diagonals["incongruent"], axis=0)
    if args.smooth == "yes":
        width_sec = 0.05  # Gaussian-kernal width in [sec]
        mean_diagon_incongruent = gaussian_filter1d(
            mean_diagon_incongruent, width_sec * 100
        )
    plt.plot(
        times,
        mean_diagon_incongruent,
        color=color,
        label="_nolegend_",
        linestyle="--",
    )
    plt.fill_between(
        times,
        mean_diagon_incongruent - errors["incongruent"],
        mean_diagon_incongruent + errors["incongruent"],
        color="red",
        alpha=0.12,
        label="_nolegend_",
    )

    pvals, clusters = diagonal_cluster_test(difference[construction])
    test = []
    for i_clust, cluster in enumerate(clusters):
        if pvals[i_clust] < 0.05:
            # plt.plot(times[cluster], np.ones_like(times[cluster])*(0.7),
            # color=colors[idx], label='_nolegend_', linestyle=':')
            test.append(times[cluster])
    if test:
        plt.axvspan(
            test[0][0], test[0][-1], 0.95, 1, color=colors[idx], alpha=0.5
        )
        timing[construction].append([test[0][0], test[0][-1]])

    # plt.ylabel('AUC', size=12);
    # plt.xlabel('time (s)', size=12);
    plt.ylim(0.33, 0.73)
    plt.axhline(0.5, linestyle="--", color="black", label="_nolegend_")

    plt.axvline(0, color="k", alpha=0.3, label="_nolegend_")

    sns.despine(offset=10, trim=False)
    plt.title("                 ")
    # plt.tight_layout()

plt.plot(
    0,
    0,
    alpha=0.7,
    linestyle="-",
    color="grey",
)
plt.plot(
    0,
    0,
    alpha=0.7,
    linestyle="--",
    color="grey",
)
plt.plot(
    0,
    0,
    alpha=0.7,
    linestyle=":",
    color="grey",
)

fig.savefig(
    f"third_order_effects_{args.distractor_number}.png",
    bbox_inches="tight",
    pad_inches=0.2,
    dpi=1200,
)
plt.show()
