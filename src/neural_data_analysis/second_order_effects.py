#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""
Second Order Effects Decoding Analysis

This script performs decoding analysis for second-order effects across three linguistic constructions:

    1. Prepositional Phrases/Grammatical Number manipulation (PP - Number)
    2. Object Relative Clauses/Grammatical Number manipulation (ObjRC - Number)
    3. Prepositional Phrases/Animacy manipulation (PP - Animacy)

The analysis trains classifiers on violation vs. non-violation data and tests them in two conditions:
1. Violation vs. non-violation in incongruent cases
2. Violation vs. non-violation in congruent cases

Key Features:
- Supports multiple sensor types (MEG, EEG, combined)
- Handles different preprocessing options (raw, preprocessed data)
- Implements cross-validated classification using Logistic Regression
- Computes AUC scores for performance evaluation
- Saves results per subject and condition

Main Parameters:
    -eoi/--events_of_interest : list
        Events to epoch (default: ["first_word_onset"])
    -rt/--response_type : str
        Response type filtering ("correct", "false", "all")
    -dn/--distractor_number : str
        Grammatical number filtering ("all", "sing", "plur")
    -sensor/--sensor_type : str
        Sensor type selection ("all", "meg", "mag", "grad", "eeg")
    -data/--data_type : str
        Data preprocessing level ("raw", "preprocessed")
    -baseline/--baseline : str
        Whether to apply baseline correction ("yes", "no")
    -crop/--crop : str
        Whether to crop around target onset ("yes", "no")
    -reject/--reject : str
        Whether to reject epochs ("yes", "no")
    -ssp/--ssp : str
        Whether to load SSP cleaned epochs ("yes", "no")
    -tmin/--tmin : float
        Start time for epoch cropping (default: -0.5)
    -tmax/--tmax : float
        End time for epoch cropping (default: 1.5)

Output Structure:
    Results are saved in directories organized by:
    - Construction type
    - Effect type
    - Sensor type
    - Data type
    - Subject
    - Response type
    - Distractor number
    - Condition (congruent/incongruent)

The script saves three types of scores per subject:
1. Overall classification scores
2. Scores for incongruent conditions
3. Scores for congruent conditions

Dependencies:
    - numpy
    - mne
    - scikit-learn
    - scipy
    - custom config and function repositories

Author: Christos-Nikolaos Zacharopoulos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

# Standard library imports
import os
import sys
import argparse
from typing import Tuple, Dict, Iterator, Any, List

# Scientific computing imports
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

# MEG/EEG analysis imports
import mne
from mne.decoding import GeneralizingEstimator

# Machine learning imports
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Local imports
sys.path.append("../../")
import config as c
from repos import func_repo as f

# Global configuration
n_jobs = -1  # Use all available CPU cores


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
    default="correct",
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
    default="raw",
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
parser.add_argument("-tmin", "--tmin", default=-0.5, help="Cropping tmin")
parser.add_argument("-tmax", "--tmax", default=1.5, help="Cropping tmax")
args = parser.parse_args()


# =============================================================================
# Global variables and attributes
# =============================================================================
root_dir = "second_order_effects"
# get path object
path = c.path


print(vars(args))


def run_decoding(construction, effect, path, root_dir, args, condition):
    """
    General decoding script to be used based on the effect and the configuration
    that we would like to test.

    Parameters
    ----------
    construction : STR
        e.g: pp_syntax.
    effect : STR
        e.g: violation_main_effect.
    standard : STR
        e.g: 'GSLS/pp'.
    deviant : STR
        e.g: 'GDLD/pp'.
    path : CLASS
    root_dir : STR.
    args : Namespace

    condition: Optional and used only in the case of the second order effects
    condition can be congruent or incongruent.

    Returns
    -------
    Performs preprocessing of the epochs and parsing based on the response type
    and the grammatical type of the distractor.

    Trains the classifier and stores the AUC scores per subject on a path specified
    by the above parameters.

    """

    # =========================================================================
    # General purpose functions
    # =========================================================================

    def load_epochs(subject: str, c, eoi: str, args) -> mne.Epochs:
        """
        Load and preprocess epochs for a given subject based on specified parameters.

        Parameters
        ----------
        subject : str
            The identifier for the subject whose data is to be loaded.
        c : module
            Configuration module containing path and utility functions.
        eoi : str
            Event of interest, used to filter or specify the epochs to load.
        args : Namespace
            Command-line arguments or configuration settings that specify data processing options.

        Returns
        -------
        mne.Epochs
            The loaded and preprocessed epochs object.

        Notes
        -----
        - The function determines the path to the epoch files based on whether SSP is applied.
        - It loads the last file in the directory, as MNE splits data and the last run contains all epochs.
        - The function filters the epochs based on the specified sensor type.
        """

        # Define paths based on whether SSP is applied
        if args.ssp == "yes":
            path = os.path.join(
                c.data_path, subject, "Epochs", args.data_type, "SSP"
            )
        else:
            path = os.path.join(c.data_path, subject, "Epochs", args.data_type)

        # Collect all files in the specified path
        files = []
        for element in c.see(path):
            file = c.join(path, element)
            if os.path.isfile(file):
                files.append(file)

        # Load the last file, which contains all epochs
        file = files[-1]

        print(40 * "--")
        print(f"Loading: {args.data_type} data.")
        print("Input: ", file)

        # Load epochs with projection applied
        epochs = mne.read_epochs(file, preload=True, proj=True)

        # Filter epochs based on sensor type
        if args.sensor_type == "all":
            epochs = epochs.copy().pick_types(meg=True, eeg=True, misc=False)
        elif args.sensor_type == "meg":
            epochs = epochs.copy().pick_types(meg=True, eeg=False, misc=False)
        elif args.sensor_type == "mag":
            epochs = epochs.copy().pick_types("mag", misc=False)
        elif args.sensor_type == "grad":
            epochs = epochs.copy().pick_types("grad", misc=False)
        elif args.sensor_type == "eeg":
            epochs = epochs.copy().pick_types(meg=False, eeg=True, misc=False)

        print(40 * "--")

        return epochs

    def parse_based_on_response_type(
        epochs: mne.Epochs, args: argparse.Namespace
    ) -> mne.Epochs:
        """
        Filter epochs based on the response type specified in the arguments.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data.
        args : argparse.Namespace
            Parsed command line arguments containing the response type.

        Returns
        -------
        mne.Epochs
            The filtered epochs object based on the specified response type.
        """
        if args.response_type == "correct":
            epochs = epochs[epochs.metadata.response == "correct"]
        elif args.response_type == "false":
            epochs = epochs[epochs.metadata.response == "false"]
        # If response_type is "all", no filtering is applied.

        return epochs

    # epochs-preprocessing
    def epochs_preprocessing(
        epochs: mne.Epochs, standard: Tuple[str, str]
    ) -> Tuple[mne.Epochs, np.ndarray]:
        """
        Preprocess the MEG/EEG epochs by resampling, cropping, applying baseline correction,
        and filtering based on response type and distractor grammatical number.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data to be preprocessed.
        standard : Tuple[str, str]
            A tuple representing the standard conditions for the analysis.

        Returns
        -------
        Tuple[mne.Epochs, np.ndarray]
            A tuple containing the preprocessed epochs object and the array of time points.
        """

        # Resample epochs to 100Hz
        print("Resampling epochs to 100Hz")
        epochs.resample(100, n_jobs=n_jobs)

        # Crop epochs around the target onset if specified
        if args.crop == "yes":
            print("Cropping epochs around the target onset (-0.5[s], +1.2[s])")
            epochs.crop(2.5 - np.abs(args.tmin), 2.5 + np.abs(args.tmax))

        # Apply baseline correction if specified
        if args.baseline == "yes":
            print("Applying baseline correction")
            target_onset = 2.5
            epochs.apply_baseline((target_onset - 0.5, target_onset))

        # Filter epochs based on response type
        print("Filtering epochs based on response type.")
        print(f"Response type: {args.response_type}")
        epochs = parse_based_on_response_type(epochs, args)

        # Extract time points from epochs
        times = epochs.times

        return epochs, times

    def parse_for_training(
        construction: str,
    ) -> dict[str, tuple[str, str] | str]:
        """
        Determine the standard and deviant conditions for training a classifier
        based on the specified linguistic construction.

        Parameters
        ----------
        construction : str
            The type of linguistic construction to analyze. Options include:
            - "pp_syntax": Prepositional Phrases with syntactic manipulation.
            - "pp_semantics": Prepositional Phrases with semantic manipulation.
            - "objrc_syntax": Object Relative Clauses with syntactic manipulation.

        Returns
        -------
        dict[str, tuple[str, str] | str]
            A dictionary containing the conditions for standard, deviant,
            congruent, and incongruent cases. The keys are:
            - "standard": Tuple of standard conditions.
            - "deviant": Tuple of deviant conditions.
            - "congruent_standard": Standard condition for congruent cases.
            - "congruent_deviant": Deviant condition for congruent cases.
            - "incongruent_standard": Standard condition for incongruent cases.
            - "incongruent_deviant": Deviant condition for incongruent cases.

        Raises
        ------
        ValueError
            If the construction type is not recognized.
        """
        if construction == "pp_syntax":
            standard = ("GSLS/synt/PP", "GSLD/synt/PP")
            deviant = ("GDLS/synt/PP", "GDLD/synt/PP")

            congruent_standard = "GSLS/synt/PP"
            congruent_deviant = "GDLD/synt/PP"

            incongruent_standard = "GSLD/synt/PP"
            incongruent_deviant = "GDLS/synt/PP"

        elif construction == "pp_semantics":
            standard = ("GSLS/sem/PP", "GSLD/sem/PP")
            deviant = ("GDLS/sem/PP", "GDLD/sem/PP")

            congruent_standard = "GSLS/sem/PP"
            congruent_deviant = "GDLD/sem/PP"

            incongruent_standard = "GSLD/sem/PP"
            incongruent_deviant = "GDLS/sem/PP"

        elif construction == "objrc_syntax":
            standard = ("GSLS/synt/objRC", "GDLS/synt/objRC")
            deviant = ("GSLD/synt/objRC", "GDLD/synt/objRC")

            congruent_standard = "GSLS/synt/objRC"
            congruent_deviant = "GDLD/synt/objRC"

            incongruent_standard = "GDLS/synt/objRC"
            incongruent_deviant = "GSLD/synt/objRC"

        else:
            raise ValueError(f"Unrecognized construction: {construction}")

        conditions = {
            "standard": standard,
            "deviant": deviant,
            "congruent_standard": congruent_standard,
            "congruent_deviant": congruent_deviant,
            "incongruent_standard": incongruent_standard,
            "incongruent_deviant": incongruent_deviant,
        }

        return conditions

    def parse_conditions(epochs: mne.Epochs, conditions: dict) -> dict:
        """
        Parse the given epochs into different conditions based on the provided condition labels.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data to be parsed.
        conditions : dict
            A dictionary containing condition labels as keys and their corresponding
            epoch selection criteria as values. Expected keys are:
            - "standard"
            - "deviant"
            - "congruent_standard"
            - "congruent_deviant"
            - "incongruent_standard"
            - "incongruent_deviant"

        Returns
        -------
        dict
            A dictionary with the same keys as the input conditions, where each key maps
            to the corresponding parsed epochs object.
        """

        parsed_epochs = {}

        # Parse epochs based on conditions
        parsed_epochs["standard"] = epochs[conditions["standard"]]
        parsed_epochs["deviant"] = epochs[conditions["deviant"]]
        parsed_epochs["congruent_standard"] = epochs[
            conditions["congruent_standard"]
        ]
        parsed_epochs["congruent_deviant"] = epochs[
            conditions["congruent_deviant"]
        ]
        parsed_epochs["incongruent_standard"] = epochs[
            conditions["incongruent_standard"]
        ]
        parsed_epochs["incongruent_deviant"] = epochs[
            conditions["incongruent_deviant"]
        ]

        return parsed_epochs

    def get_common_indices(parsed_epochs: dict) -> dict:
        """
        Identify common indices between standard and congruent/incongruent conditions
        for both standard and deviant epochs.

        Parameters
        ----------
        parsed_epochs : dict
            A dictionary containing parsed epochs with keys:
            - "standard"
            - "deviant"
            - "congruent_standard"
            - "congruent_deviant"
            - "incongruent_standard"
            - "incongruent_deviant"

        Returns
        -------
        dict
            A dictionary with keys:
            - "congruent_standard"
            - "congruent_deviant"
            - "incongruent_standard"
            - "incongruent_deviant"
            Each key maps to an array of indices representing the intersection of
            metadata indices between the standard/deviant and their corresponding
            congruent/incongruent conditions.
        """

        indices = {}

        # =========================================================================
        #  CONGRUENT CASE
        # =========================================================================
        _, congruent_standard, _ = np.intersect1d(
            parsed_epochs["standard"].metadata.index,
            parsed_epochs["congruent_standard"].metadata.index,
            return_indices=True,
        )

        _, congruent_deviant, _ = np.intersect1d(
            parsed_epochs["deviant"].metadata.index,
            parsed_epochs["congruent_deviant"].metadata.index,
            return_indices=True,
        )

        # =========================================================================
        #  INCONGRUENT CASE
        # =========================================================================

        _, incongruent_standard, _ = np.intersect1d(
            parsed_epochs["standard"].metadata.index,
            parsed_epochs["incongruent_standard"].metadata.index,
            return_indices=True,
        )

        _, incongruent_deviant, _ = np.intersect1d(
            parsed_epochs["deviant"].metadata.index,
            parsed_epochs["incongruent_deviant"].metadata.index,
            return_indices=True,
        )

        indices["congruent_standard"] = congruent_standard
        indices["congruent_deviant"] = congruent_deviant
        indices["incongruent_standard"] = incongruent_standard
        indices["incongruent_deviant"] = incongruent_deviant

        # Sanity checks to ensure the sum of congruent and incongruent indices matches the total
        assert len(indices["congruent_standard"]) + len(
            indices["incongruent_standard"]
        ) == len(
            parsed_epochs["standard"]
        ), "Unequal lengths for standard epochs"

        assert len(indices["congruent_deviant"]) + len(
            indices["incongruent_deviant"]
        ) == len(
            parsed_epochs["deviant"]
        ), "Unequal lengths for deviant epochs"

        return indices

    def make_sklearn_compatible(
        parsed_epochs: Dict[str, mne.Epochs]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert parsed epochs into a format compatible with scikit-learn.

        This function concatenates the data from standard and deviant epochs,
        and creates corresponding labels for classification.

        Parameters
        ----------
        parsed_epochs : Dict[str, mne.Epochs]
            A dictionary containing the parsed epochs with keys "standard" and "deviant".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the feature matrix X and the label vector y.
            - X : np.ndarray
                The concatenated data from standard and deviant epochs.
            - y : np.ndarray
                The labels for the epochs, where 1 represents standard and 0 represents deviant.
        """
        # Keep the seed constant (42)
        np.random.seed(c.random_state)

        # Make the data compatible with sklearn
        X = np.concatenate(
            (
                parsed_epochs["standard"].get_data(),
                parsed_epochs["deviant"].get_data(),
            )
        )

        y = np.concatenate(
            (
                np.ones(parsed_epochs["standard"].get_data().shape[0]),
                np.zeros(parsed_epochs["deviant"].get_data().shape[0]),
            )
        )

        return X, y

    def preprocess_decoding_input(X: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to the input data prior to decoding.

        This function uses a Savitzky-Golay filter to smooth the input data,
        which can help improve the performance of the classifier.

        # The Savitzky-Golay filter parameters are:
        # - window_length: 11 (The length of the filter window, must be a positive odd integer)
        # - polyorder: 1 (The order of the polynomial used to fit the samples, must be less than window_length)
        # At a sampling rate of 100Hz, a window_length of 11 corresponds to a time window of 110ms.
        # This means the filter smooths the data over a period of 110ms, which is suitable for capturing
        # short-term trends in the data without overly distorting the signal.
        Parameters
        ----------
        X : np.ndarray
            The input feature matrix to be smoothed.

        Returns
        -------
        np.ndarray
            The smoothed feature matrix.
        """
        # Apply smoothing prior to decoding

        X = savgol_filter(X, 11, 1)

        return X

    def build_classifier() -> GeneralizingEstimator:
        """
        Build and return a classifier pipeline with standard scaling and logistic regression.

        This function creates a machine learning pipeline that includes standard scaling
        and logistic regression with balanced class weights. The logistic regression
        classifier is wrapped in a GeneralizingEstimator to perform temporal generalization
        decoding.

        Returns
        -------
        GeneralizingEstimator
            A generalizing estimator with a logistic regression classifier.
        """
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                random_state=c.random_state,
                max_iter=1000,
                solver="lbfgs",
            ),
        )

        classifier = GeneralizingEstimator(
            clf, scoring="roc_auc", n_jobs=n_jobs, verbose=True
        )

        return classifier

    def split_in_folds(
        X: np.ndarray, y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Split the data into stratified folds for cross-validation.

        This function splits the input data into stratified folds, ensuring that each fold
        has approximately the same distribution of classes. The number of folds is determined
        based on the number of class members, with a minimum of 2 and a maximum of 5 folds.

        Parameters
        ----------
        X : np.ndarray
            The input feature matrix.
        y : np.ndarray
            The target labels.

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            An iterator that provides the training and testing indices for each fold.
        """
        n_class_members = int(np.round(y.shape[0] / 2))
        if n_class_members < 5:
            n_folds = n_class_members
        else:
            n_folds = 5

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=c.random_state
        )

        folds = skf.split(X, y)

        print(40 * "--")
        print(f"#of FOLDS: {n_folds}")
        print(40 * "--")

        return folds

    def create_dirs(
        subject: str, args: argparse.Namespace, condition: str
    ) -> str:
        """
        Create directories for storing output based on various parameters.

        This function constructs a directory path based on the subject, response type,
        distractor grammatical number, epoch length, SSP, baseline, and condition.
        It ensures that the directory exists, creating it if necessary.

        Parameters
        ----------
        subject : str
            The identifier for the subject. If empty, 'mean_scores' is used.
        args : argparse.Namespace
            Parsed command line arguments containing various settings such as response type,
            distractor number, crop, SSP, and baseline.
        condition : str
            The condition for which the directory is being created. Can be 'congruent',
            'incongruent', or None.

        Returns
        -------
        str
            The path to the output directory.
        """

        # Update based on response type
        if args.response_type == "correct":
            response_type = "correct_responses"
        elif args.response_type == "all":
            response_type = "all_responses"
        elif args.response_type == "false":
            response_type = "false_responses"

        # Update based on the grammatical number of the distractor
        if args.distractor_number == "sing":
            distractor_number = "singular_distractor"
        elif args.distractor_number == "plur":
            distractor_number = "plural_distractor"
        else:
            distractor_number = "both_distractors"

        # Epochs length
        if args.crop == "yes":
            length = "cropped_around_target"
        elif args.crop == "no":
            length = "whole_sentence"

        # SSP
        if args.ssp == "yes":
            ssp = "with_ssp"
        elif args.ssp == "no":
            ssp = "without_ssp"

        # Baseline
        if args.baseline == "yes":
            baseline = "with_baseline"
        elif args.baseline == "no":
            baseline = "without_baseline"

        if condition:
            # Split at the subject level
            if subject:
                path2output = c.join(
                    path.to_output(),
                    "whole_coverage",
                    root_dir,
                    "Decoding",
                    construction,
                    effect,
                    args.sensor_type,
                    args.data_type,
                    subject,
                    length,
                    ssp,
                    baseline,
                    response_type,
                    distractor_number,
                    condition,
                )
            else:
                path2output = c.join(
                    path.to_output(),
                    "whole_coverage",
                    root_dir,
                    "Decoding",
                    construction,
                    effect,
                    args.sensor_type,
                    args.data_type,
                    "mean_scores",
                    length,
                    ssp,
                    baseline,
                    response_type,
                    distractor_number,
                    condition,
                )
            if not c.exists(path2output):
                c.make(path2output)
        else:
            # Split at the subject level
            if subject:
                path2output = c.join(
                    path.to_output(),
                    "whole_coverage",
                    root_dir,
                    "Decoding",
                    construction,
                    effect,
                    args.sensor_type,
                    args.data_type,
                    subject,
                    length,
                    ssp,
                    baseline,
                    response_type,
                    distractor_number,
                )
            else:
                path2output = c.join(
                    path.to_output(),
                    "whole_coverage",
                    root_dir,
                    "Decoding",
                    construction,
                    effect,
                    args.sensor_type,
                    args.data_type,
                    "mean_scores",
                    length,
                    ssp,
                    baseline,
                    response_type,
                    distractor_number,
                )
            if not c.exists(path2output):
                c.make(path2output)

        return path2output

    def save_score_per_subject(
        scores_all: np.ndarray,
        scores_incongruent: np.ndarray,
        scores_congruent: np.ndarray,
        times: np.ndarray,
        subject: str,
        condition: str,
    ) -> None:
        """
        Save classification scores and time points for a given subject and condition.

        Parameters
        ----------
        scores_all : np.ndarray
            Array containing the overall classification scores for the subject.
        scores_incongruent : np.ndarray
            Array containing the classification scores for incongruent conditions.
        scores_congruent : np.ndarray
            Array containing the classification scores for congruent conditions.
        times : np.ndarray
            Array of time points corresponding to the scores.
        subject : str
            Identifier for the subject whose scores are being saved.
        condition : str
            The condition under which the scores are being saved (e.g., "all", "incongruent", "congruent").

        Returns
        -------
        None
        """

        print(
            "",
            40 * "--",
            "\n",
            f"Saving scores for subject: {subject}",
            "\n",
            40 * "--",
        )

        # Make paths
        scores_all_path = create_dirs(subject, args, "all")
        scores_incongruent_path = create_dirs(subject, args, "incongruent")
        scores_congruent_path = create_dirs(subject, args, "congruent")

        # Make filenames
        fname_scores_all = c.join(
            scores_all_path, construction + "_all_scores.npy"
        )
        fname_scores_incongruent = c.join(
            scores_incongruent_path, construction + "_incongruent_scores.npy"
        )
        fname_scores_congruent = c.join(
            scores_congruent_path, construction + "_congruent_scores.npy"
        )

        # Save files
        np.save(fname_scores_all, scores_all)
        np.save(fname_scores_incongruent, scores_incongruent)
        np.save(fname_scores_congruent, scores_congruent)

        # Times saved in the 'all scores case' path
        fname_times = c.join(scores_all_path, construction + "_times.npy")
        np.save(fname_times, times)

    def false_responses_checker(
        standard: Tuple[str, str], epochs: mne.Epochs
    ) -> bool:
        """
        Check if all required conditions are available in the epochs metadata.

        Parameters
        ----------
        standard : Tuple[str, str]
            A tuple representing the standard conditions for the analysis.
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data.

        Returns
        -------
        bool
            True if all required conditions are available, False otherwise.
        """
        structure = standard[0].split("/")[1]
        available_conditions = (
            epochs[structure].metadata.condition.unique().tolist()
        )
        required_conditions = [i.split("/")[0] for i in list(standard)]
        result = all(
            elem in available_conditions for elem in required_conditions
        )

        return result

    def train_and_test_on_different_data(
        X: np.ndarray,
        y: np.ndarray,
        folds: list,
        indices: dict,
        classifier,
        offset: int,
    ) -> tuple:
        """
        Train and test a classifier on violation vs non-violation data for each syntactic structure.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix where each row corresponds to a sample and each column to a feature.
        y : np.ndarray
            The target vector where each element corresponds to the class label of a sample.
        folds : list
            A list of tuples, each containing train and test indices for cross-validation.
        indices : dict
            A dictionary containing indices for different conditions (e.g., incongruent, congruent).
        classifier :
            A machine learning classifier object with fit and score methods.
        offset : int
            An offset value used to adjust indices for deviant conditions.

        Returns
        -------
        tuple
            A tuple containing mean scores across folds for all, incongruent, and congruent conditions.
        """
        # Save scores per fold
        scores_incongruent, scores_congruent, scores_all = (
            [] for _ in range(3)
        )
        for idx, (train_index, test_index) in enumerate(folds):
            # Train model
            print("Training")
            classifier.fit(X[train_index], y[train_index])

            sub_incongruent = np.concatenate(
                [
                    np.intersect1d(
                        test_index, indices["incongruent_standard"]
                    ),
                    np.intersect1d(
                        test_index, indices["incongruent_deviant"] + offset
                    ),
                ]
            )

            sub_congruent = np.concatenate(
                [
                    np.intersect1d(test_index, indices["congruent_standard"]),
                    np.intersect1d(
                        test_index, indices["congruent_deviant"] + offset
                    ),
                ]
            )

            # Test on subjects
            print(f"Testing on INCONGRUENT, fold: {idx+1}")
            score_incongruent = classifier.score(
                X[sub_incongruent], y[sub_incongruent]
            )
            print(f"Testing on CONGRUENT, fold: {idx+1}")
            score_congruent = classifier.score(
                X[sub_congruent], y[sub_congruent]
            )
            print(f"Testing on ALL, fold: {idx+1}")
            score_all = classifier.score(X[test_index], y[test_index])

            # All violation vs all non-violation
            scores_all.append(np.diag(score_all))
            # Only incongruent
            scores_incongruent.append(np.diag(score_incongruent))
            # Only congruent
            scores_congruent.append(np.diag(score_congruent))

        # Return the mean across folds
        scores_incongruent = np.mean(scores_incongruent, axis=0)
        scores_congruent = np.mean(scores_congruent, axis=0)
        scores_all = np.mean(scores_all, axis=0)

        return scores_all, scores_incongruent, scores_congruent

    def loop_and_classify(conditions: Dict[str, Any], condition: str) -> None:
        """
        Loop through subjects, load and preprocess data, train classifiers, and save scores.

        Parameters
        ----------
        conditions : Dict[str, Any]
            A dictionary containing conditions for parsing epochs.
        condition : str
            The condition type, either 'congruent' or 'incongruent'.

        Returns
        -------
        None
        """
        (
            all_subjects_all_scores,
            all_subjects_congruent_scores,
            all_subjects_incongruent_scores,
        ) = ([] for i in range(0, 3))

        # Loop through subjects and load data
        for subject in c.subjects_list:
            print(40 * "**")
            print(f"{construction, effect, subject}")
            print(f"Response type: {args.response_type}")
            print(f"Distractor number: {args.distractor_number}")
            print(40 * "**")

            # Load the epochs
            epochs = load_epochs(subject, c, args.events_of_interest, args)
            # Preprocess epochs
            epochs, times = epochs_preprocessing(epochs)
            parsed_epochs = parse_conditions(epochs, conditions)
            # Get the common indices
            indices = get_common_indices(parsed_epochs)
            # Make the data compatible with sklearn
            X, y = make_sklearn_compatible(parsed_epochs)
            # Define offset
            offset = np.where(np.diff(y) == -1)[0][0] + 1
            # Preprocess decoding input
            X = preprocess_decoding_input(X)
            # Train classifier
            classifier = build_classifier()
            # Split in folds
            folds = split_in_folds(X, y)

            # Train and test a classifier per condition (ALL, CONGRUENT, INCONGRUENT) per subject
            (
                scores_all,
                scores_incongruent,
                scores_congruent,
            ) = train_and_test_on_different_data(
                X, y, folds, indices, classifier, offset
            )

            # Save scores per subject for post-processing
            save_score_per_subject(
                scores_all,
                scores_incongruent,
                scores_congruent,
                times,
                subject,
                condition,
            )

    def save_mean_AUC(
        scores: np.ndarray,
        construction: str,
        times: np.ndarray,
        condition: str,
    ) -> None:
        """
        Save the mean AUC scores and the standard error of the mean (SEM) across subjects.

        Parameters
        ----------
        scores : np.ndarray
            Array containing the AUC scores for all subjects.
        construction : str
            The construction type (e.g., 'pp_syntax').
        times : np.ndarray
            Array of time points corresponding to the scores.
        condition : str
            The condition under which the scores are being saved (e.g., 'all', 'incongruent', 'congruent').

        Returns
        -------
        None
        """
        # Get the path
        path2output = create_dirs([], args, condition)

        fname_mean = c.join(
            path2output, construction + "_grand_average_AUC.npy"
        )
        fname_error = c.join(
            path2output, construction + "_grand_average_error.npy"
        )

        fname_times = c.join(path2output, construction + "_times.npy")

        # Take the mean across subjects
        mean_score = np.mean(scores, axis=0)
        # Take the error across subjects
        error = stats.sem(scores, axis=0)
        # Save files
        np.save(fname_mean, mean_score)
        np.save(fname_error, error)
        np.save(fname_times, times)

    # =========================================================================
    #    RUN
    # =========================================================================
    conditions = parse_for_training(construction)
    loop_and_classify(
        conditions
    )  # takes place for all subjects simultaneously


# =============================================================================
# WRAP UP AND COMPILE
# =============================================================================


def wrap_up_and_compile(
    effect: str,
    constructions: List[str],
    path: Any,
    root_dir: str,
    args: argparse.Namespace,
) -> None:
    """
    Run the decoding process for each construction and congruency condition.

    Parameters
    ----------
    effect : str
        The effect type to be analyzed (e.g., 'congruency_effects').
    constructions : List[str]
        A list of construction types to be analyzed.
    path : Any
        The path object or module used for file operations.
    root_dir : str
        The root directory where the output will be saved.
    args : argparse.Namespace
        Parsed command line arguments containing various settings and parameters.

    Returns
    -------
    None
    """
    for congruency in ["congruent", "incongruent"]:
        for construction in constructions:
            print(f"{congruency, construction}")
            run_decoding(
                construction, effect, path, root_dir, args, congruency
            )


# Call the function with appropriate arguments
wrap_up_and_compile(
    "congruency_effects", c.constructions, path, root_dir, args
)
