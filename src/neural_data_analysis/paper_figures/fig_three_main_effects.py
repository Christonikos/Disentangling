#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Data Analysis - First Order Effects (Figure 3)
==================================================

This script generates Figure 3 of the paper, analyzing and visualizing first-order effects
in neural data processing during language comprehension tasks. It examines three main effects:
1. Violation
2. Congruency 
3. Transition

Key Features:
------------
- Processes MEG/EEG data for different experimental conditions
- Handles multiple data types (raw, preprocessed)
- Supports various sensor configurations (MEG, EEG, combined)
- Implements statistical analysis with cluster-based permutation tests
- Generates publication-ready figure with error bars and significance markers (SEM)

Main Parameters:
--------------
- events_of_interest: Which events to analyze (e.g., "first_word_onset")
- response_type: Filter by response accuracy ("correct", "false", "all")
- distractor_number: Grammatical number filtering ("sing", "plur", "all")
- sensor_type: Sensor selection ("meg", "eeg", "mag", "grad", "all")
- data_type: Processing level ("raw", "preprocessed")
- baseline: Whether to apply baseline correction
- crop: Whether to crop epochs around target onset
- roi: Region of interest for analysis

Usage:
------
python fig_three_main_effects.py [-h] [-eoi EVENTS] [-rt RESPONSE_TYPE] 
                                [-dn DISTRACTOR_NUMBER] [-sensor SENSOR_TYPE]
                                [-data DATA_TYPE] [-baseline BASELINE]
                                [-crop CROP] ...

Example:
-------
python fig_three_main_effects.py -eoi first_word_onset -rt correct -sensor meg 
                                -data preprocessed -baseline yes



Notes:
------
- The script assumes a specific directory structure for input/output
- Statistical significance is assessed using cluster-based permutation tests
- Smoothing can be applied to the results for visualization

Author: Christos-Nikolaos Zacharopoulos

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
    Parse and validate the grammatical number settings for distractor analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing distractor_number setting

    Returns
    -------
    str
        The parsed distractor type:
        - "both_distractors": When analyzing all grammatical numbers
        - "singular_distractor": When analyzing only singular distractors
        - "plural_distractor": When analyzing only plural distractors

    Notes
    -----
    This function is used to filter experimental data based on the grammatical 
    number of distractors in linguistic stimuli.
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
    Parse and validate response type settings for analysis filtering.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing response_type setting

    Returns
    -------
    str
        The parsed response type:
        - "all_responses": Include all participant responses
        - "correct_responses": Include only correct responses
        - "false_responses": Include only incorrect responses

    Notes
    -----
    This function is used to filter experimental data based on participant 
    response accuracy.
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
    Parse and validate epoch length settings for temporal analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing crop setting

    Returns
    -------
    str
        The parsed length type:
        - "cropped_around_target": Analysis window centered on target stimulus
        - "whole_sentence": Full sentence analysis window

    Notes
    -----
    This function determines the temporal window used for analysis, either 
    focusing on the target stimulus or analyzing the entire sentence.
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
    Parse and validate baseline correction settings.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing baseline setting

    Returns
    -------
    str
        The parsed baseline setting:
        - "with_baseline": Apply baseline correction
        - "without_baseline": Skip baseline correction

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
    Parse and validate Signal Space Projection (SSP) settings.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing ssp setting

    Returns
    -------
    str
        The parsed SSP setting:
        - "with_ssp": Apply SSP noise reduction
        - "without_ssp": Skip SSP processing

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
    Parse and validate grid search settings for parameter optimization.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing grid setting

    Returns
    -------
    str
        The parsed grid search setting:
        - "with_grid_search": Use grid search for parameter optimization
        - "without_grid_search": Use default parameters


    """
    if args.grid == "yes":
        grid = "with_grid_search"
    elif args.grid == "no":
        grid = "without_grid_search"
    return grid


# =============================================================================
# Collect scores from all subjects per construction, effect and SOA
# =============================================================================
def collect_scores(path: str, construction: str, effect: str) -> tuple:
    """
    Aggregate and process decoding scores across all subjects.

    Parameters
    ----------
    path : str
        Base path to the data directory
    construction : str
        Type of linguistic construction ('pp', 'obj', 'sem')
    effect : str
        The effect being analyzed (from c.first_order_effects)

    Returns
    -------
    tuple
        Contains:
        - all_diagonals : list
            Diagonal scores for each subject [n_subjects × n_timepoints]
        - all_scores : list
            Full GAT matrices for each subject [n_subjects × n_timepoints × n_timepoints]
        - error : ndarray
            Standard error of the mean across subjects
        - times : ndarray
            Time points vector for the analysis

    Notes
    -----
    - Loads and processes pre-computed numpy arrays
    - Handles different experimental conditions through parsing objects
    - Organizes results based on specified directory structure
    - Can handle multiple data types (raw, preprocessed)
    - Supports various sensor configurations
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
    Generate standardized file paths for saving figure outputs.

    Parameters
    ----------
    construction : str
        The linguistic construction type being analyzed (e.g., 'pp', 'obj')

    Returns
    -------
    str
        Complete file path for saving the figure, including all relevant 
        parameters in the directory structure

    Notes
    -----
    - Creates nested directory structure if it doesn't exist
    - Incorporates multiple experimental parameters in path:
        - ROI
        - Effect type
        - Sensor type
        - Data processing level
        - Response filtering
        - Baseline correction status
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
    Perform cluster-based permutation testing on diagonal scores.

    Parameters
    ----------
    all_diagonals : list
        List of arrays containing diagonal scores for each subject
        Shape: [n_subjects × n_timepoints]

    Returns
    -------
    tuple
        Contains:
        - cluster_p_values : array
            P-values for each identified cluster
        - clusters : list
            List of boolean masks identifying significant clusters

    Notes
    -----
    - Uses MNE's implementation of cluster-based permutation testing
    - Default parameters:
        - 1000 permutations
        - One-tailed test (tail=1)
        - Threshold determined by t-distribution (p < 0.01)

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
