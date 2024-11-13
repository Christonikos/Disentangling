#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
"""
Neural Data Analysis - Second Order Effects (Figure 4)
===================================================

This script generates Figure 4 of the paper, analyzing and visualizing second-order 
(interaction) effects in neural data during language processing tasks.

Key Analyses:
------------
1. Congruency Effects: Analyzes how grammatical congruency interacts with:
   - PP-Number: Prepositional Phrase number agreement
   - ObjRC-Number: Object Relative Clause number agreement
   - PP-Animacy: SAnimacy in Prepositional Phrases

Features:
---------
- Processes MEG/EEG data across multiple experimental conditions
- Implements cluster-based statistical testing
- Generates publication-ready figure with:
  * Separate plots for each construction type
  * Error bars showing standard error of the mean
  * Significance markers for cluster-based statistics


Main Parameters:
--------------
events_of_interest : list
    Which events to analyze (default: ["first_word_onset"])
response_type : str
    Filter by response accuracy ("correct", "false", "all")
distractor_number : str
    Grammatical number filtering ("sing", "plur", "all")
sensor_type : str
    Sensor selection ("meg", "eeg", "mag", "grad", "all")
data_type : str
    Processing level ("raw", "preprocessed")
roi : str
    Region of interest for analysis

Usage:
------
python fig_four_second_order_effects.py [-h] [-eoi EVENTS] [-rt RESPONSE_TYPE] 
                                      [-dn DISTRACTOR_NUMBER] [-sensor SENSOR_TYPE]
                                      [-data DATA_TYPE] ...

Example:
-------
python fig_four_second_order_effects.py -eoi first_word_onset -rt correct -sensor meg 
                                      -data preprocessed -baseline yes

Dependencies:
------------
Standard Library:
    - sys
    - argparse
External:
    - numpy: Numerical computations
    - matplotlib: Plotting
    - seaborn: Enhanced plotting
    - scipy: Statistical functions
    - mne: MEG/EEG analysis
    - statsmodels: Statistical modeling
Local:
    - config: Configuration settings
    - func_repo: Utility functions

Notes:
------
- The script assumes a specific directory structure for input/output
- Statistical significance is assessed using cluster-based permutation tests
- Results can be smoothed using Gaussian filtering for visualization
- Timing information is extracted and saved for further analysis

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
    default="no",
    help="Options: yes/no Whether to see results generated with grid search or not",
)
parser.add_argument("-smooth", "--smooth", default="yes")
args = parser.parse_args()

root_dir = "second_order_effects"
effect = "congruency_effects"


# =============================================================================
# Select distractor grammatical number
# =============================================================================
def gn_parsing_object(args: argparse.Namespace) -> str:
    """
    Parse and validate grammatical number settings for distractor analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing distractor_number setting.
        Valid options are:
        - "all": Include all grammatical numbers
        - "sing": Include only singular distractors
        - "plur": Include only plural distractors

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
    number of distractors in linguistic stimuli. It's crucial for analyzing 
    number agreement effects.
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
        Command line arguments containing response_type setting.
        Valid options are:
        - "all": Include all responses
        - "correct": Include only correct responses
        - "false": Include only incorrect responses

    Returns
    -------
    str
        The parsed response type:
        - "all_responses": Include all participant responses
        - "correct_responses": Include only correct responses
        - "false_responses": Include only incorrect responses

    Notes
    -----
    This function enables analysis of neural responses based on behavioral 
    performance, allowing investigation of error-related processing.
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
        Command line arguments containing crop setting.
        Valid options are:
        - "yes": Crop epochs around target
        - "no": Use full sentence epochs

    Returns
    -------
    str
        The parsed length type:
        - "cropped_around_target": Analysis window centered on target stimulus
        - "whole_sentence": Full sentence analysis window

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
        Command line arguments containing baseline setting.
        Valid options are:
        - "yes": Apply baseline correction
        - "no": Skip baseline correction

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
        Command line arguments containing ssp setting.
        Valid options are:
        - "yes": Apply SSP
        - "no": Skip SSP

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
        Command line arguments containing grid setting.
        Valid options are:
        - "yes": Use grid search
        - "no": Use default parameters

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
def collect_scores(path: str, construction: str, effect: str, condition: str) -> tuple:
    """
    Aggregate decoding scores across subjects for specific experimental conditions.

    Parameters
    ----------
    path : str
        Base path to the data directory
    construction : str
        Type of linguistic construction ('pp', 'obj', 'sem')
    effect : str
        The effect being analyzed (from second_order_effects)
    condition : str
        Specific condition to analyze ('congruent' or 'incongruent')

    Returns
    -------
    tuple
        - all_diagonals : list
            Diagonal scores for each subject [n_subjects × n_timepoints]
        - error : ndarray
            Standard error of the mean across subjects
        - times : ndarray
            Time points vector for the analysis

    Notes
    -----
    - Loads pre-computed scores from numpy arrays
    - Handles different experimental conditions through parsing objects
    - Automatically detects and loads appropriate time vectors
    - Compatible with various sensor configurations and preprocessing options
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
        all_diagonals.append(diag)

    # Calculate standard error of the mean across subjects
    error = stats.sem(all_diagonals, axis=0)

    # Load time vector from first subject (same for all)
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
    Generate standardized file paths for saving figures.

    Parameters
    ----------
    construction : str
        The linguistic construction type being analyzed

    Returns
    -------
    str
        Complete file path for saving the figure

    Notes
    -----
    - Creates nested directory structure if it doesn't exist
    - Incorporates experimental parameters in path:
        * ROI
        * Effect type
        * Sensor type
        * Data processing level
        * Response filtering
        * Baseline correction status

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
    # Create path if it doesn't exist
    if not c.exists(path2figs):
        c.make(path2figs)
    fname = c.join(path2figs, "second_order_effects.pdf")

    return fname


# =============================================================================
# Return Title and color per Construction
# =============================================================================
def color_and_title(construction: str) -> tuple[str, str]:
    """
    Determine the visualization properties for each construction type.

    Parameters
    ----------
    construction : str
        The type of construction being analyzed. Options:
        - "pp_syntax": Prepositional Phrase Number agreement
        - "objrc_syntax": Object Relative Clause Number agreement
        - "pp_semantics": Prepositional Phrase Animacy

    Returns
    -------
    tuple[str, str]
        - title: Formatted title string with mathematical notation
        - color: Color code for plotting (darkblue, darkgreen, or darkred)


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


def diagonal_cluster_test(all_diagonals: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Perform cluster-based permutation testing on temporal data.

    Parameters
    ----------
    all_diagonals : np.ndarray
        Subject-wise diagonal scores, shape: (n_subjects, n_timepoints)

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        - cluster_p_values: P-values for each identified cluster
        - clusters: List of boolean masks identifying significant clusters

    Statistical Details
    ------------------
    - Uses MNE's cluster-based permutation test
    - Parameters:
        * Permutations: 1000
        * Threshold: p < 0.01 (two-tailed)
        * Test statistic: one-sample t-test
    - FDR correction available but currently disabled

    Notes
    -----
    The function identifies temporally contiguous clusters of significant 
    differences from chance level (0.5) while controlling for multiple 
    comparisons.
    """
    # Set cluster threshold
    p_threshold = 0.01
    n_subjects = len(c.subjects_list)
    thres = -stats.distributions.t.ppf(p_threshold / 2.0, n_subjects - 1)
    thres = None

    # Compute statistic
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
    # FDR correction is available but currently disabled
    # _, cluster_p_values = fdr(cluster_p_values)

    return cluster_p_values, clusters


# %%
labelling = ["Congruent", "Incongruent", "Target"]
colors = ["darkblue", "darkgreen", "darkred"]


fig = plt.figure(dpi=100, facecolor="w", edgecolor="w")
fig.set_size_inches(12, 4)

timing = {}

difference = {}
lines = []
line_labels = [r"$congruent$", r"$incongruent$", r"$p<0.05$"]
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
            plt.plot(
                times[cluster],
                np.ones_like(times[cluster]) * (0.7),
                color=colors[idx],
                label="_nolegend_",
                linestyle=":",
            )
            test.append(times[cluster])
    if test:
        plt.axvspan(
            test[0][0], test[0][-1], 0.95, 1, color=colors[idx], alpha=0.5
        )
        timing[construction].append([test[0][0], test[0][-1]])

    plt.ylabel("                  ", size=12)
    plt.xlabel("                  ", size=12)
    plt.ylim(0.43, 0.7)
    plt.axhline(0.5, linestyle="--", color="black", label="_nolegend_")

    plt.axvline(0, color="k", alpha=0.3, label="_nolegend_")

    sns.despine(offset=10, trim=False)
    plt.title("                  ")
    plt.tight_layout()

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
    "second_order_effects.png", bbox_inches="tight", pad_inches=0.2, dpi=1200
)
plt.show()


# %% Create a standalone legend
# fig=plt.figure(dpi=400, facecolor='w', edgecolor='w')
# fig.set_size_inches(7,5)
# plt.plot(0,0, color='darkblue', linestyle='-', label='PP-Number')
# plt.plot(0,0, color='darkgreen', linestyle='-', label='ObjRC-Number')
# plt.plot(0,0, color='darkred', linestyle='-', label='PP-Animacy')
# # plt.plot(0,0, color='dimgrey', linestyle='-', label='Congruent')
# # plt.plot(0,0, color='dimgrey', linestyle='--', label='Incongruent')
# # plt.plot(0,0, color='dimgrey', linestyle=':', label='p<0.05')
# plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.5),
# fancybox=True, shadow=False, ncol=5)
# plt.axis('off')
# plt.tight_layout()
# fig.savefig('congruency_legend.png',bbox_inches='tight', pad_inches=0.2, dpi=400)

# plt.show()
