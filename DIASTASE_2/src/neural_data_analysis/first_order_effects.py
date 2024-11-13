#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural decoding analysis for main effects.

This script performs temporal generalization decoding analysis to investigate neural 
processing of linguistic violations using MEG/EEG data. It analyzes three main effects:

Main Effects:
- Violation: Grammatical vs Ungrammatical sentences (GLOBAL effect)
- Interference: Presence vs Absence of interfering elements (LOCAL effect)
- Congruency: Match vs Mismatch between elements of the two nouns
(grammatical number and animacy) (CONGRUENCY effect)

Across Three Construction Types:
    1. Prepositional Phrases/Grammatical Number manipulation (PP - Number)
    2. Object Relative Clauses/Grammatical Number manipulation (ObjRC - Number)
    3. Prepositional Phrases/Animacy manipulation (PP - Animacy)

The analysis pipeline:
1. Loads preprocessed MEG/EEG epochs
2. Applies additional preprocessing (baseline correction, filtering)
3. Splits data based on experimental conditions and response accuracy
4. Performs temporal generalization decoding using logistic regression
5. Evaluates classifier performance using AUC scores
6. Saves results for statistical analysis

Usage:
    python first_order_effects.py [-h] [-eoi EVENTS] [-rt RESPONSE_TYPE] 
                                [-dn DISTRACTOR_NUMBER] [-sensor SENSOR_TYPE]
                                [-data DATA_TYPE] [-baseline BASELINE] [-crop CROP]
                                [-reject REJECT] [-ssp SSP] [-tmin TMIN] [-tmax TMAX]
                                [-grid GRID]

Example:
    python first_order_effects.py -eoi first_word_onset -rt correct -sensor meg 
                                -data preprocessed -baseline yes
                                
Author: Christos-Nikolaos Zacharopoulos
                                
"""

from __future__ import annotations


# Standard library
import os
import sys
import argparse
from typing import Tuple, List, Optional, Dict, Any, Iterator, Union

# Scientific computing
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

# Machine learning
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# MEG/EEG analysis
import mne
from mne.decoding import GeneralizingEstimator

# Local imports
sys.path.append("../../")
import config as c
from repos import func_repo as f

# Global configuration
n_jobs: int = -1  # Use all available CPU cores

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="MEG/EEG decoding analysis for linguistic effects"
)
parser.add_argument(
    "-eoi",
    "--events_of_interest",
    nargs="+",
    default=["first_word_onset"],
    help="Select events to epoch",
)
parser.add_argument(
    "-rt",
    "--response_type",
    default="correct",
    help="Parse epochs based on response type (correct/false/all)",
)
parser.add_argument(
    "-dn",
    "--distractor_number",
    default="all",
    help="Parse epochs based on distractor number (all/sing/plur)",
)
parser.add_argument(
    "-sensor",
    "--sensor_type",
    default="all",
    help="Select sensor type (all/meg/mag/grad/eeg)",
)
parser.add_argument(
    "-data",
    "--data_type",
    default="raw",
    help="Select data type (raw/preprocessed)",
)
parser.add_argument(
    "-baseline", "--baseline", default="yes", help="Apply baseline correction"
)
parser.add_argument(
    "-crop", "--crop", default="no", help="Crop epochs around target"
)
parser.add_argument(
    "-reject", "--reject", default="no", help="Reject bad epochs"
)
parser.add_argument(
    "-ssp", "--ssp", default="yes", help="Use Signal Space Projection"
)
parser.add_argument(
    "-tmin", "--tmin", default=-0.5, type=float, help="Start time for epoch"
)
parser.add_argument(
    "-tmax", "--tmax", default=1.5, type=float, help="End time for epoch"
)
parser.add_argument(
    "-grid", "--grid", default="no", help="Use grid search for hyperparameters"
)

args = parser.parse_args()

# Global variables
root_dir: str = "first_order_effects"
path = c.path


# =============================================================================
# PARSE THE CONDITIONS BASED ON THE EFFECT ORDER
# =============================================================================


def first_order_effects(
    effect: str, construction: str
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Determine the standard and deviant conditions based on the specified effect and construction.

    Parameters
    ----------
    effect : str
        The type of effect to analyze. Options include:
        - "main_effect_of_violation": Analyzes the main effect of violation.
        - "main_effect_of_congruency": Analyzes the main effect of congruency.
        - "main_effect_of_transition": Analyzes the main effect of transition.
    construction : str
        The construction type to consider. Options include:
        - "pp_syntax": Prepositional phrase syntax.
        - "pp_semantics": Prepositional phrase semantics.
        - "objrc_syntax": Object relative clause syntax.

    Returns
    -------
    Tuple[Tuple[str, str], Tuple[str, str]]
        A tuple containing two tuples:
        - The first tuple represents the standard conditions.
        - The second tuple represents the deviant conditions.

    Raises
    ------
    ValueError
        If the provided effect or construction is not recognized.
    """

    if effect == "main_effect_of_violation":
        if construction == "pp_syntax":
            standard = ("GSLS/synt/PP", "GSLD/synt/PP")
            deviant = ("GDLS/synt/PP", "GDLD/synt/PP")
        elif construction == "pp_semantics":
            standard = ("GSLS/sem/PP", "GSLD/sem/PP")
            deviant = ("GDLS/sem/PP", "GDLD/sem/PP")
        elif construction == "objrc_syntax":
            standard = ("GSLS/synt/objRC", "GDLS/synt/objRC")
            deviant = ("GSLD/synt/objRC", "GDLD/synt/objRC")
        else:
            raise ValueError(f"Unrecognized construction: {construction}")

    elif effect == "main_effect_of_congruency":
        if construction == "pp_syntax":
            standard = ("GSLS/synt/PP", "GDLD/synt/PP")  # all congruent
            deviant = ("GSLD/synt/PP", "GDLS/synt/PP")  # all incongruent
        elif construction == "pp_semantics":
            standard = ("GSLS/sem/PP", "GDLD/sem/PP")  # all congruent
            deviant = ("GSLD/sem/PP", "GDLS/sem/PP")  # all incongruent
        elif construction == "objrc_syntax":
            standard = ("GSLS/synt/objRC", "GDLD/synt/objRC")  # all congruent
            deviant = ("GSLD/synt/objRC", "GDLS/synt/objRC")  # all incongruent
        else:
            raise ValueError(f"Unrecognized construction: {construction}")

    elif effect == "main_effect_of_transition":
        if construction == "pp_syntax":
            standard = ("GSLS/synt/PP", "GDLS/synt/PP")  # all LS
            deviant = ("GSLD/synt/PP", "GDLD/synt/PP")  # all LD
        elif construction == "pp_semantics":
            standard = ("GSLS/sem/PP", "GDLS/sem/PP")  # all LS
            deviant = ("GSLD/sem/PP", "GDLD/sem/PP")  # all LD
        elif construction == "objrc_syntax":
            standard = ("GSLS/synt/objRC", "GSLD/synt/objRC")
            deviant = ("GDLS/synt/objRC", "GDLD/synt/objRC")
        else:
            raise ValueError(f"Unrecognized construction: {construction}")

    else:
        raise ValueError(f"Unrecognized effect: {effect}")

    return standard, deviant


def run_decoding(
    construction, effect, standard, deviant, path, root_dir, args, condition
):
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

    def parse_based_on_distractor_number(
        epochs: mne.Epochs, args: argparse.Namespace, standard: Tuple[str, str]
    ) -> mne.Epochs:
        """
        Filter epochs based on the distractor number specified in the arguments.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data.
        args : argparse.Namespace
            Parsed command line arguments containing the distractor number.
        standard : Tuple[str, str]
            A tuple representing the standard conditions for the analysis.

        Returns
        -------
        mne.Epochs
            The filtered epochs object based on the specified distractor number.
        """
        if args.distractor_number == "sing":
            if "objRC" in standard[0]:
                epochs = epochs[epochs.metadata.G_number == "plur"]
            else:
                epochs = epochs[epochs.metadata.G_number == "sing"]
        elif args.distractor_number == "plur":
            if "objRC" in standard[0]:
                epochs = epochs[epochs.metadata.G_number == "sing"]
            else:
                epochs = epochs[epochs.metadata.G_number == "plur"]
        # If distractor_number is not "sing" or "plur", no filtering is applied.

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

    def parse_conditions(
        epochs: mne.Epochs, standard: Tuple[str, str], deviant: Tuple[str, str]
    ) -> Tuple[mne.Epochs, mne.Epochs]:
        """
        Parse the given epochs into standard and deviant conditions, applying rejection criteria if specified.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing MEG/EEG data to be parsed.
        standard : Tuple[str, str]
            A tuple representing the standard conditions for the analysis.
        deviant : Tuple[str, str]
            A tuple representing the deviant conditions for the analysis.

        Returns
        -------
        Tuple[mne.Epochs, mne.Epochs]
            A tuple containing two mne.Epochs objects:
            - The first element corresponds to the epochs for the standard conditions.
            - The second element corresponds to the epochs for the deviant conditions.
        """

        print(40 * "--")
        if args.sensor_type == "meg":
            reject = {"eeg": 200e-6}
        elif args.sensor_type == "mag":
            reject = {"mag": 4e-12}
        elif args.sensor_type == "grad":
            reject = {"grad": 4000e-13}
        elif args.sensor_type == "eeg":
            reject = {"eeg": 200e-6}
        elif args.sensor_type == "all":
            reject = {"grad": 4000e-13, "mag": 4e-12, "eeg": 200e-6}
        print(40 * "--")

        if args.reject == "yes":
            print("Dropping bad epochs")
            condition1 = epochs[standard].drop_bad(reject=reject)
            condition2 = epochs[deviant].drop_bad(reject=reject)
        else:
            condition1 = epochs[standard]
            condition2 = epochs[deviant]

        return condition1, condition2

    def make_sklearn_compatible(
        condition1: mne.Epochs, condition2: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert MNE Epochs data into a format compatible with scikit-learn.

        Parameters
        ----------
        condition1 : mne.Epochs
            The epochs object for the first condition.
        condition2 : mne.Epochs
            The epochs object for the second condition.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - X: A numpy array of shape (n_samples, n_features) with the combined data from both conditions.
            - y: A numpy array of shape (n_samples,) with labels (1 for condition1, 0 for condition2).
        """
        # Keep the seed constant (42)
        np.random.seed(c.random_state)

        # Make the data compatible with sklearn
        X = np.concatenate((condition1.get_data(), condition2.get_data()))
        y = np.concatenate(
            (
                np.ones(condition1.get_data().shape[0]),
                np.zeros(condition2.get_data().shape[0]),
            )
        )

        return X, y

    def preprocess_decoding_input(X: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to the input data using a Savitzky-Golay filter.

        Parameters
        ----------
        X : np.ndarray
            The input data array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The smoothed data array of the same shape as the input.
        """
        # Apply Savitzky-Golay filter with window length 11 and polynomial order 1
        X = savgol_filter(X, 11, 1)

        return X

    def build_classifier() -> GeneralizingEstimator:
        """
        Build a classifier pipeline for MEG/EEG decoding analysis.

        This function constructs a machine learning pipeline using logistic regression
        for temporal generalization decoding. It optionally includes a grid search
        for hyperparameter tuning based on command line arguments.

        Returns
        -------
        GeneralizingEstimator
            A GeneralizingEstimator object configured with the specified pipeline
            and scoring method (ROC AUC).

        Notes
        -----
        - If grid search is enabled via command line arguments, a GridSearchCV
          object is used to optimize hyperparameters.
        - The classifier is set to use all available CPU cores for parallel processing.
        """
        grid = {
            "logisticregression__C": np.logspace(-3, 3, 7),
            "logisticregression__penalty": ["l2"],
            "logisticregression__solver": ["lbfgs", "liblinear"],
        }

        if args.grid == "yes":
            from sklearn.model_selection import RepeatedStratifiedKFold

            cv = RepeatedStratifiedKFold(
                n_splits=10, n_repeats=3, random_state=c.random_state
            )
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    random_state=c.random_state,
                    max_iter=1000,
                ),
            )
            clf = GridSearchCV(clf, grid, cv=cv, n_jobs=-1)

        else:
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

        Parameters
        ----------
        X : np.ndarray
            The input data to be split into folds.
        y : np.ndarray
            The target labels corresponding to the input data.

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            An iterator that provides the train and test indices for each fold.

        Notes
        -----
        - The number of folds is determined based on the number of class members.
        - If the number of class members is less than 5, the number of folds is set to the number of class members.
        - Otherwise, the number of folds is set to 5.
        - The folds are stratified to ensure each fold has a representative distribution of the target labels.
        """
        n_class_members = int(np.round(y.shape[0] / 2))
        n_folds = n_class_members if n_class_members < 5 else 5

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=c.random_state
        )
        folds = skf.split(X, y)

        print(40 * "--")
        print(f"# of FOLDS: {n_folds}")
        print(40 * "--")

        return folds

    def get_score_per_fold(
        X: np.ndarray,
        y: np.ndarray,
        folds: Iterator[Tuple[np.ndarray, np.ndarray]],
        classifier: GeneralizingEstimator,
        method: str = "mean",
    ) -> Union[float, np.ndarray]:
        """
        Compute the score for each fold in cross-validation.

        Parameters
        ----------
        X : np.ndarray
            The input data to be split into folds.
        y : np.ndarray
            The target labels corresponding to the input data.
        folds : Iterator[Tuple[np.ndarray, np.ndarray]]
            An iterator that provides the train and test indices for each fold.
        classifier : GeneralizingEstimator
            The classifier to be trained and evaluated.
        method : str, optional
            The method to aggregate scores across folds. Default is "mean".

        Returns
        -------
        Union[float, np.ndarray]
            The aggregated score across folds. If method is "mean", returns the mean score.

        Notes
        -----
        - If the response type is "false" and there are no positive samples in the test set,
          the fold is skipped.
        - The classifier is trained on the training set and evaluated on the test set for each fold.
        - The scores are aggregated using the specified method.
        """
        scores = []  # list to hold scores per fold
        # Loop through folds
        for train_index, test_index in folds:
            # Skip fold if response type is "false" and no positive samples in test set
            if args.response_type == "false" and np.sum(y[test_index]) == 0:
                continue
            # Train the model
            classifier.fit(X[train_index], y[train_index])
            # Evaluate the model
            scores.append(classifier.score(X[test_index], y[test_index]))
        if method == "mean":
            # Get the mean score across folds
            scores = np.mean(np.array(scores), axis=0)
        return scores

    def create_dirs(
        subject: Optional[str],
        args: argparse.Namespace,
        condition: Optional[str],
    ) -> str:
        """
        Create directories for storing output based on various parameters.

        Parameters
        ----------
        subject : Optional[str]
            The subject identifier. If None, directories are created for mean scores.
        args : argparse.Namespace
            Parsed command line arguments containing various options such as response type,
            distractor number, crop, SSP, baseline, and grid search settings.
        condition : Optional[str]
            Specific condition for which directories are created. If None, directories are created
            without condition-specific subdirectories.

        Returns
        -------
        str
            The path to the output directory.

        Notes
        -----
        - The directory structure is determined by the combination of input parameters.
        - If the directory does not exist, it is created.
        """

        # Determine response type directory name
        response_type = {
            "correct": "correct_responses",
            "all": "all_responses",
            "false": "false_responses",
        }.get(args.response_type, "all_responses")

        # Determine distractor number directory name
        distractor_number = {
            "sing": "singular_distractor",
            "plur": "plural_distractor",
        }.get(args.distractor_number, "both_distractors")

        # Determine epoch length directory name
        length = (
            "cropped_around_target" if args.crop == "yes" else "whole_sentence"
        )

        # Determine SSP directory name
        ssp = "with_ssp" if args.ssp == "yes" else "without_ssp"

        # Determine baseline directory name
        baseline = (
            "with_baseline" if args.baseline == "yes" else "without_baseline"
        )

        # Determine grid search directory name
        grid = (
            "with_grid_search" if args.grid == "yes" else "without_grid_search"
        )

        # Construct the path to the output directory
        path_components = [
            path.to_output(),
            "whole_coverage",
            root_dir,
            "Decoding",
            construction,
            effect,
            args.sensor_type,
            args.data_type,
            subject if subject else "mean_scores",
            length,
            ssp,
            baseline,
            response_type,
            distractor_number,
        ]

        if condition:
            path_components.append(condition)

        path2output = c.join(*path_components)

        # Create the directory if it does not exist
        if not c.exists(path2output):
            c.make(path2output)

        return path2output

    def save_score_per_subject(
        scores: np.ndarray, times: np.ndarray, subject: str, condition: str
    ) -> None:
        """
        Save the decoding scores and corresponding times for a given subject and condition.

        Parameters
        ----------
        scores : np.ndarray
            The decoding scores to be saved.
        times : np.ndarray
            The time points corresponding to the scores.
        subject : str
            The identifier of the subject.
        condition : str
            The experimental condition.

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

        # Get the path
        path2output = create_dirs(subject, args, condition)

        fname_scores = c.join(path2output, construction + "_scores.npy")
        fname_times = c.join(path2output, construction + "_times.npy")

        # Save files
        np.save(fname_scores, scores)
        np.save(fname_times, times)

    def false_responses_checker(
        standard: Tuple[str, str], epochs: mne.Epochs
    ) -> bool:
        """
        Check if all required conditions are present in the epochs metadata.

        Parameters
        ----------
        standard : Tuple[str, str]
            A tuple containing the standard conditions.
        epochs : mne.Epochs
            The epochs object containing the data and metadata.

        Returns
        -------
        bool
            True if all required conditions are present, False otherwise.
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

    def loop_and_classify(
        standard: Tuple[str, str], deviant: Tuple[str, str], condition: str
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Process and classify MEG/EEG data for each subject, returning classification scores and time points.

        Parameters
        ----------
        standard : Tuple[str, str]
            A tuple containing the standard conditions for classification.
        deviant : Tuple[str, str]
            A tuple containing the deviant conditions for classification.
        condition : str
            The experimental condition being analyzed.

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            A tuple containing:
            - A list of numpy arrays with classification scores for each subject.
            - A numpy array of time points corresponding to the scores.
        """
        all_subjects_scores = []

        # Loop through subjects and load data
        for subject in c.subjects_list:
            print(
                f"Processing: Standard={standard}, Deviant={deviant}, Condition={condition}, Subject={subject}"
            )
            print(f"Arguments: {args}")

            # Load the epochs
            epochs = load_epochs(subject, c, args.events_of_interest, args)

            # Preprocess epochs
            epochs, times = epochs_preprocessing(epochs, standard)

            # Check for false responses
            if args.response_type == "false":
                if not false_responses_checker(standard, epochs):
                    print(
                        f"Skipping subject due to missing conditions: {subject}"
                    )
                    continue

            # Parse the epochs into conditions
            condition1, condition2 = parse_conditions(
                epochs, standard, deviant
            )

            # Make the data compatible with sklearn
            X, y = make_sklearn_compatible(condition1, condition2)

            # Preprocess decoding input
            X = preprocess_decoding_input(X)

            # Train classifier
            classifier = build_classifier()

            # Split data into folds
            folds = split_in_folds(X, y)

            # Get score per subject
            scores = get_score_per_fold(X, y, folds, classifier, method="mean")

            # Save scores per subject for post-processing
            save_score_per_subject(scores, times, subject, condition)

            # Append scores to the list for all subjects
            all_subjects_scores.append(scores)

        return all_subjects_scores, times

    def save_mean_AUC(
        scores: List[np.ndarray],
        construction: str,
        times: np.ndarray,
        condition: str,
    ) -> None:
        """
        Save the mean AUC scores and standard error across subjects.

        Parameters
        ----------
        scores : List[np.ndarray]
            A list of numpy arrays containing classification scores for each subject.
        construction : str
            The construction type being analyzed.
        times : np.ndarray
            A numpy array of time points corresponding to the scores.
        condition : str
            The experimental condition being analyzed.

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
    scores, times = loop_and_classify(
        standard, deviant, condition
    )  # takes place for all subjects simultaneously
    # Save the mean-AUC across all subjects for post-processing
    save_mean_AUC(scores, construction, times, condition)


# =============================================================================
# WRAP UP AND COMPILE
# =============================================================================


def wrap_up_and_compile(
    c: Any, path: Any, root_dir: str, args: argparse.Namespace
) -> None:
    """
    Run the decoding process for each effect and construction.

    Parameters
    ----------
    c : Any
        Configuration object containing the effects and constructions to be analyzed.
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
    for effect in c.first_order_effects:
        for construction in c.constructions:
            print(f"Processing effect: {effect}, construction: {construction}")

            # Determine the standard and deviant conditions based on the effect and construction
            standard, deviant = first_order_effects(effect, construction)

            # Run the decoding analysis for the given construction and effect
            run_decoding(
                construction,
                effect,
                standard,
                deviant,
                path,
                root_dir,
                args,
                [],
            )


# Execute the wrap-up and compile process
wrap_up_and_compile(c, path, root_dir, args)
