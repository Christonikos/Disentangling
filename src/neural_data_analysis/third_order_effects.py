#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""Neural decoding analysis for grammatical number effects in language processing.

This script performs time-resolved decoding analyses to investigate third-order 
effects in grammatical number agreement processing across different linguistic 
constructions. It uses MEG/EEG data to classify violation vs non-violation trials
while considering the role of attractor number and agreement congruency.

The analysis pipeline:
1. Loads preprocessed MEG/EEG epochs for each subject
2. Applies additional preprocessing (resampling, baseline correction)
3. Trains classifiers on violation vs non-violation trials
4. Tests generalization across different conditions:
   - Violation vs non-violation with singular attractors
   - Violation vs non-violation with plural attractors
   - Congruent agreement conditions
   - Incongruent agreement conditions
5. Saves classification scores (AUC) for each subject and condition

The script analyzes three linguistic constructions:
- Prepositional Phrases (PP) - Syntax
- Prepositional Phrases (PP) - Semantics  
- Object Relative Clauses (ObjRC) - Syntax

Usage
-----
python third_order_effects.py [-h] [-eoi EVENTS] [-rt RESPONSE_TYPE] 
                            [-dn DISTRACTOR_NUMBER] [-sensor SENSOR_TYPE]
                            [-data DATA_TYPE] [-baseline BASELINE] [-crop CROP]
                            [-reject REJECT] [-ssp SSP] [-tmin TMIN] [-tmax TMAX]
                            [-grid GRID]

Key Parameters
-------------
response_type : {'correct', 'false', 'all'}
    Type of behavioral responses to include
distractor_number : {'singular', 'plural', 'all'} 
    Grammatical number of the attractor noun
sensor_type : {'all', 'meg', 'mag', 'grad', 'eeg'}
    Type of sensors to use in the analysis
data_type : {'raw', 'preprocessed'}
    Level of data preprocessing to use

Output
------
Saves per-subject and grand average classification scores in:
{output_dir}/whole_coverage/Decoding/{construction}/{effect}/...

Notes
-----
- Uses scikit-learn for classification (LogisticRegression with StandardScaler)
- Implements time-generalization analysis using MNE-Python
- Handles both MEG and EEG data types
- Supports different preprocessing options (SSP, baseline correction, etc.)

Author: Christos-Nikolaos Zacharopoulos
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================

# Standard library imports
import sys
import os
import argparse
from typing import Tuple, Dict, Iterator, Any, List

# Third-party imports
import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from mne.decoding import GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.signal import savgol_filter
from scipy import stats

# Local application imports
import config as c
from repos import func_repo as f

# Add project path
sys.path.append("../../")

n_jobs = -1


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
    default="singular",
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
parser.add_argument("-grid", "--grid", default="no", help="Grid search ON/OFF")
args = parser.parse_args()


# =============================================================================
# Global variables and attributes
# =============================================================================
root_dir = "third_order_effects"
# get path object
path = c.path


def run_decoding(construction, effect, path, root_dir, args, condition):
    """Run decoding analysis for grammatical number effects in language processing.

    This function performs decoding analysis to classify violation vs non-violation trials
    across different linguistic constructions (PP, ObjRC), conditions
    (congruent/incongruent) and feature of intereset (singular/plural, animate/inanimate)

    Parameters
    ----------
    construction : str
        The linguistic construction to analyze ('pp_syntax', 'pp_semantics', or 'objrc_syntax')
    effect : str
        The effect being analyzed (e.g., 'violation_main_effect')
    path : Path
        Path object containing directory structure information
    root_dir : str
        Root directory for saving results
    args : argparse.Namespace
        Command line arguments controlling analysis parameters
    condition : str
        Analysis condition ('congruent' or 'incongruent')

    Notes
    -----
    The function:
    1. Loads MEG/EEG epochs data
    2. Preprocesses and parses epochs based on response type and distractor number
    3. Trains classifiers on violation vs non-violation trials
    4. Tests generalization across time points
    5. Saves AUC scores per subject and condition
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

        # Filter epochs based on distractor grammatical number
        print("Filtering epochs based on grammatical number.")
        print(f"Distractor number: {args.distractor_number}")
        epochs = parse_based_on_distractor_number(epochs, args, standard)

        # Extract time points from epochs
        times = epochs.times

        return epochs, times

    def parse_for_training(
        construction: str, args: argparse.Namespace
    ) -> dict:
        """
        Determine the standard and deviant conditions for training a classifier
        based on the specified construction type.

        Parameters
        ----------
        construction : str
            The type of construction to be analyzed (e.g., 'pp_syntax', 'pp_semantics', 'objrc_syntax').
        args : argparse.Namespace
            Parsed command line arguments containing various settings and parameters.

        Returns
        -------
        dict
            A dictionary containing the standard and deviant conditions for both
            congruent and incongruent scenarios, keyed by 'standard', 'deviant',
            'congruent_standard', 'congruent_deviant', 'incongruent_standard', and
            'incongruent_deviant'.
        """
        if construction == "pp_syntax":
            standard = "GSLS/synt/PP", "GSLD/synt/PP"
            deviant = "GDLS/synt/PP", "GDLD/synt/PP"
            congruent_standard = "GSLS/synt/PP"
            congruent_deviant = "GDLD/synt/PP"
            incongruent_standard = "GSLD/synt/PP"
            incongruent_deviant = "GDLS/synt/PP"

        elif construction == "pp_semantics":
            standard = "GSLS/sem/PP", "GSLD/sem/PP"
            deviant = "GDLS/sem/PP", "GDLD/sem/PP"
            congruent_standard = "GSLS/sem/PP"
            congruent_deviant = "GDLD/sem/PP"
            incongruent_standard = "GSLD/sem/PP"
            incongruent_deviant = "GDLS/sem/PP"

        elif construction == "objrc_syntax":
            standard = "GSLS/synt/objRC", "GDLS/synt/objRC"
            deviant = "GSLD/synt/objRC", "GDLD/synt/objRC"
            congruent_standard = "GSLS/synt/objRC"
            congruent_deviant = "GDLD/synt/objRC"
            incongruent_standard = "GDLS/synt/objRC"
            incongruent_deviant = "GSLD/synt/objRC"

        conditions = {
            "standard": standard,
            "deviant": deviant,
            "congruent_standard": congruent_standard,
            "congruent_deviant": congruent_deviant,
            "incongruent_standard": incongruent_standard,
            "incongruent_deviant": incongruent_deviant,
        }

        return conditions

    def parse_conditions(
        epochs: mne.Epochs, conditions: Dict[str, str], construction: str
    ) -> Dict[str, mne.Epochs]:
        """
        Parse and filter epochs based on specified conditions and construction type.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data.
        conditions : dict
            A dictionary containing condition labels as keys and their corresponding
            epoch selection criteria as values.
        construction : str
            The type of construction to be analyzed (e.g., 'pp_syntax', 'pp_semantics', 'objrc_syntax').

        Returns
        -------
        dict
            A dictionary containing parsed epochs for each condition, including 'standard',
            'deviant', 'congruent_standard', 'congruent_deviant', 'incongruent_standard',
            and 'incongruent_deviant'.
        """
        parsed_epochs = {}

        # Parse epochs based on conditions
        standard = epochs[conditions["standard"]]
        deviant = epochs[conditions["deviant"]]

        congruent_standard = epochs[conditions["congruent_standard"]]
        congruent_deviant = epochs[conditions["congruent_deviant"]]

        incongruent_standard = epochs[conditions["incongruent_standard"]]
        incongruent_deviant = epochs[conditions["incongruent_deviant"]]

        if construction == "pp_syntax":
            if args.distractor_number == "singular":
                congruent_standard = congruent_standard[
                    congruent_standard.metadata.G_number == "plur"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.G_number == "plur"
                ]
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.G_number == "plur"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.G_number == "plur"
                ]
            elif args.distractor_number == "plural":
                congruent_standard = congruent_standard[
                    congruent_standard.metadata.G_number == "sing"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.G_number == "sing"
                ]
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.G_number == "sing"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.G_number == "sing"
                ]

        elif construction == "objrc_syntax":
            if args.distractor_number == "singular":
                congruent_standard = congruent_standard[
                    congruent_standard.metadata.G_number == "sing"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.G_number == "sing"
                ]
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.G_number == "sing"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.G_number == "sing"
                ]
            elif args.distractor_number == "plural":
                congruent_standard = congruent_standard[
                    congruent_standard.metadata.G_number == "plur"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.G_number == "plur"
                ]
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.G_number == "plur"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.G_number == "plur"
                ]

        elif construction == "pp_semantics":
            if args.distractor_number == "singular":
                # Match singular to inanimate attractor
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "anim"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "anim"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "anim"
                ]
            elif args.distractor_number == "plural":
                # Match plural to animate attractor
                congruent_deviant = congruent_deviant[
                    congruent_deviant.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "inanim"
                ]
                incongruent_standard = incongruent_standard[
                    incongruent_standard.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "inanim"
                ]
                incongruent_deviant = incongruent_deviant[
                    incongruent_deviant.metadata.pair_index.str.split(
                        "-", n=1, expand=True
                    )[0]
                    == "inanim"
                ]

        parsed_epochs["standard"] = standard
        parsed_epochs["deviant"] = deviant
        parsed_epochs["congruent_standard"] = congruent_standard
        parsed_epochs["congruent_deviant"] = congruent_deviant
        parsed_epochs["incongruent_standard"] = incongruent_standard
        parsed_epochs["incongruent_deviant"] = incongruent_deviant

        return parsed_epochs

    def get_common_indices(parsed_epochs: dict) -> dict:
        """
        Compute the common indices between standard and deviant epochs for both congruent and incongruent cases.

        Parameters
        ----------
        parsed_epochs : dict
            A dictionary containing parsed epochs with keys 'standard', 'deviant', 'congruent_standard',
            'congruent_deviant', 'incongruent_standard', and 'incongruent_deviant'. Each value is an
            MNE Epochs object with associated metadata.

        Returns
        -------
        dict
            A dictionary containing the indices of common elements for congruent and incongruent cases.
            Keys include 'congruent_standard', 'congruent_deviant', 'incongruent_standard', and 'incongruent_deviant'.
        """

        metadata = {
            "standard": parsed_epochs["standard"].metadata,
            "deviant": parsed_epochs["deviant"].metadata,
            "congruent_standard": parsed_epochs["congruent_standard"].metadata,
            "congruent_deviant": parsed_epochs["congruent_deviant"].metadata,
        }

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

        return indices

    def make_sklearn_compatible(
        parsed_epochs: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert parsed epochs data into a format compatible with sklearn.

        Parameters
        ----------
        parsed_epochs : dict
            A dictionary containing the parsed epochs data. Keys should include 'standard' and 'deviant'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the feature matrix X and the target vector y.
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

        This function constructs a directory path based on the provided subject,
        arguments, and condition. It ensures that the directory exists, creating
        it if necessary.

        Parameters
        ----------
        subject : str
            The subject identifier for which the directory is being created.
        args : argparse.Namespace
            The arguments containing various settings such as response type,
            distractor number, cropping, SSP, baseline, and grid search options.
        condition : str
            The specific condition for which the directory is being created.

        Returns
        -------
        str
            The path to the created or existing directory.
        """

        # Update based on response type
        if args.response_type == "correct":
            response_type = "correct_responses"
        elif args.response_type == "all":
            response_type = "all_responses"
        elif args.response_type == "false":
            response_type = "false_responses"

        # Update based on the grammatical number of the distractor
        if args.distractor_number == "singular":
            distractor_number = "singular_distractor"
        elif args.distractor_number == "plural":
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

        # Grid
        if args.grid == "yes":
            grid = "with_grid_search"
        elif args.grid == "no":
            grid = "without_grid_search"

        if condition:
            # Split at the subject level
            if subject:
                path2output = c.join(
                    path.to_output(),
                    "whole_coverage",
                    grid,
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
                    grid,
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
                    grid,
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
                    grid,
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
        Save the decoding scores for a given subject and condition.

        This function saves the scores for all conditions, incongruent conditions,
        and congruent conditions to separate files. It also saves the time points
        associated with the scores.

        Parameters
        ----------
        scores_all : np.ndarray
            The array containing scores for all conditions.
        scores_incongruent : np.ndarray
            The array containing scores for incongruent conditions.
        scores_congruent : np.ndarray
            The array containing scores for congruent conditions.
        times : np.ndarray
            The array of time points associated with the scores.
        subject : str
            The identifier for the subject whose scores are being saved.
        condition : str
            The condition under which the scores were obtained.

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

        This function verifies whether all required conditions specified in the
        standard tuple are present in the metadata of the given epochs.

        Parameters
        ----------
        standard : Tuple[str, str]
            A tuple representing the standard conditions for the analysis.
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data with metadata.

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
        folds: Iterator[Tuple[np.ndarray, np.ndarray]],
        indices: Dict[str, np.ndarray],
        classifier: GeneralizingEstimator,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Train and evaluate a classifier on different subsets of data, specifically focusing on
        violation versus non-violation per syntactic structure. This function handles the training
        and testing of a classifier across multiple cross-validation folds, and computes scores
        for incongruent, congruent, and all data conditions.

        Parameters
        ----------
        X : np.ndarray
            The input feature matrix for the classifier.
        y : np.ndarray
            The target labels corresponding to the input features.
        folds : Iterator[Tuple[np.ndarray, np.ndarray]]
            An iterator providing the training and testing indices for each cross-validation fold.
        indices : Dict[str, np.ndarray]
            A dictionary containing indices for different conditions (e.g., incongruent, congruent).
        classifier : GeneralizingEstimator
            The classifier to be trained and evaluated.
        offset : int
            An offset value used to adjust indices for certain conditions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the mean scores across folds for all, incongruent, and congruent conditions.
        """
        # Initialize lists to store scores for each fold
        scores_incongruent, scores_congruent, scores_all = (
            [] for _ in range(3)
        )

        for idx, (train_index, test_index) in enumerate(folds):
            # Determine indices for incongruent and congruent subsets
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

            # Skip the fold if any subset is empty or lacks class diversity
            if len(sub_incongruent) == 0 or len(sub_congruent) == 0:
                print(f"SKIPPING fold: {idx+1}")
                continue

            if (
                len(set(y[sub_incongruent])) < 2
                or len(set(y[sub_congruent])) < 2
            ):
                print(f"SKIPPING fold: {idx+1}")
                continue

            # Train the classifier
            print("Training")
            classifier.fit(X[train_index], y[train_index])

            # Evaluate the classifier on different subsets
            print(f"Testing on INCONGRUENT, fold: {idx+1}")
            score_incongruent = classifier.score(
                X[sub_incongruent], y[sub_incongruent]
            )

            print(f"Testing on CONGRUENT, fold: {idx+1}")
            score_congruent = classifier.score(
                X[sub_congruent], y[sub_congruent]
            )

            score_all = classifier.score(X[test_index], y[test_index])

            # Append scores for each condition
            scores_all.append(np.diag(score_all))
            scores_incongruent.append(np.diag(score_incongruent))
            scores_congruent.append(np.diag(score_congruent))

        # Compute the mean scores across all folds
        scores_incongruent = np.mean(scores_incongruent, axis=0)
        scores_congruent = np.mean(scores_congruent, axis=0)
        scores_all = np.mean(scores_all, axis=0)

        return scores_all, scores_incongruent, scores_congruent

    def loop_and_classify(
        conditions: dict, condition: str, construction: str
    ) -> None:
        """
        Loop through each subject, preprocess the data, train classifiers, and save the results.

        This function iterates over a list of subjects, loading and preprocessing their MEG/EEG data.
        It then trains classifiers on the data for different conditions (all, congruent, incongruent)
        and saves the classification scores for each subject.

        Parameters
        ----------
        conditions : dict
            A dictionary containing the conditions for parsing the epochs.
        condition : str
            The specific condition under which the analysis is being performed.
        construction : str
            The linguistic construction being analyzed.

        Returns
        -------
        None
        """
        # Initialize lists to store scores for each subject
        (
            all_subjects_all_scores,
            all_subjects_congruent_scores,
            all_subjects_incongruent_scores,
        ) = ([] for _ in range(3))

        # Loop through subjects and load data
        for subject in c.subjects_list:
            print(40 * "**")
            print(
                f"Processing: Construction: {construction}, Effect: {effect}, Subject: {subject}"
            )
            print(f"Response type: {args.response_type}")
            print(f"Distractor number: {args.distractor_number}")
            print(40 * "**")

            # Load the epochs for the current subject
            epochs = load_epochs(subject, c, args.events_of_interest, args)

            # Preprocess the epochs
            epochs, times = epochs_preprocessing(epochs)

            # Parse the epochs based on conditions and construction
            parsed_epochs = parse_conditions(epochs, conditions, construction)

            # Get the common indices for parsed epochs
            indices = get_common_indices(parsed_epochs)

            # Make the data compatible with sklearn
            X, y = make_sklearn_compatible(parsed_epochs)

            # Define offset for class separation
            offset = np.where(np.diff(y) == -1)[0][0] + 1

            # Preprocess the decoding input
            X = preprocess_decoding_input(X)

            # Build the classifier
            classifier = build_classifier()

            # Split the data into stratified folds
            folds = split_in_folds(X, y)

            # Train and test the classifier on different data subsets
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
        Save the mean Area Under the Curve (AUC) scores and associated error for a given construction and condition.

        This function calculates the mean AUC scores and the standard error of the mean (SEM) across subjects,
        and saves these values along with the time points to separate files. The files are saved in a directory
        structure based on the provided condition.

        Parameters
        ----------
        scores : np.ndarray
            The array containing AUC scores for all subjects.
        construction : str
            The linguistic construction being analyzed (e.g., "PP - Number").
        times : np.ndarray
            The array of time points associated with the scores.
        condition : str
            The condition under which the scores were obtained (e.g., "congruent", "incongruent").

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

        # Calculate the mean across subjects
        mean_score = np.mean(scores, axis=0)
        # Calculate the standard error of the mean across subjects
        error = stats.sem(scores, axis=0)
        # Save files
        np.save(fname_mean, mean_score)
        np.save(fname_error, error)
        np.save(fname_times, times)

    # =========================================================================
    #    RUN
    # =========================================================================
    conditions = parse_for_training(construction, args)
    loop_and_classify(
        conditions, condition, construction
    )  # takes place for all subjects simultaneously


# =============================================================================
# WRAP UP AND COMPILE
# =============================================================================
def run_decoding_for_all_conditions(
    effect: str, constructions: List[str], path: str, root_dir: str, args: Any
) -> None:
    """
    Run decoding analysis for all specified conditions and constructions.

    This function iterates over the given conditions (congruent and incongruent) and linguistic constructions,
    performing decoding analysis for each combination. The results are printed and processed accordingly.

    Parameters
    ----------
    effect : str
        The effect type to be analyzed (e.g., "number_effects").
    constructions : List[str]
        A list of linguistic constructions to be analyzed (e.g., ["PP - Number", "ObjRC - Number"]).
    path : str
        The base path for input data.
    root_dir : str
        The root directory for output data.
    args : Any
        Additional arguments required for the decoding process.

    Returns
    -------
    None
    """
    for congruency in ["congruent", "incongruent"]:
        for construction in constructions:
            print(f"{congruency, construction, args.distractor_number}")
            run_decoding(
                construction, effect, path, root_dir, args, congruency
            )


# Call the function
run_decoding_for_all_conditions(
    "number_effects", c.constructions, path, root_dir, args
)
