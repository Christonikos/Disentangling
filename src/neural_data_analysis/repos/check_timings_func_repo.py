#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains a collection of functions used for analyzing and verifying 
the timing and synchronization of neural data events with behavioral log files. 
The primary goal is to ensure that the events recorded in the neural data 
correspond accurately to the events logged during an experiment. 

The script includes functions for:
- Mapping and renaming event identifiers in log files to match those in the 
  neural data.
- Cleaning and synchronizing event triggers to remove spurious or overlapping 
  events.
- Conducting a series of tests to verify the existence and alignment of data 
  and log files, including:
  - Checking the existence of data and log files.
  - Matching the number of events between log files and neural data.
  - Comparing the durations of specific events to expected values.
  - Calculating the R-squared value to assess the linear relationship between 
    log file times and neural event times.
- Exporting synchronized events and logs for further analysis or storage.
- Detecting and logging bad sensors based on variance and gradient thresholds 
  to ensure data quality.

The script is designed to be used in a research setting where precise timing 
and synchronization between neural data and behavioral logs are critical. It 
leverages libraries such as MNE for handling neural data and employs various 
statistical and data processing techniques to ensure the integrity and 
accuracy of the data.

Note: The script assumes the presence of certain configuration settings and 
external dependencies, such as the 'config' module and specific data 
structures for paths and triggers.

Authors: Christos-Nikolaos Zacharopoulos
"""
# Standard library imports
import os.path
from collections import defaultdict
from typing import Tuple, Dict, Any


# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mne

# Local application imports
import config

# Alias
join = os.path.join


# Alias
where = np.where
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Supportive functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def map_log2triggers(
    run_log: pd.DataFrame, triggers: Any, subject: str, run: str
) -> pd.DataFrame:
    """
    Rename the log-file event names to ensure correspondence between the TTL-event names
    (manually created) and the log-file names.

    This function modifies the 'Event' column in the run_log DataFrame by replacing
    specific event names with standardized names. It ensures that the event names in
    the log file match those used in the TTL events.

    Parameters
    ----------
    run_log : pd.DataFrame
        The DataFrame containing log-file events with an 'Event' column to be renamed.
    triggers : Any
        The triggers object containing TTL-event names. This parameter is not used
        in this function but may be relevant for other operations.
    subject : str
        The identifier for the subject whose log-file is being processed.
    run : str
        The identifier for the run or session being processed.

    Returns
    -------
    pd.DataFrame
        The modified run_log DataFrame with renamed event entries.

    Raises
    ------
    ValueError
        If the number of events in the 'Event' column changes after renaming, indicating
        an unexpected modification in the DataFrame dimensions.
    """
    import warnings

    warnings.filterwarnings("ignore")
    dim_before = len(run_log["Event"])

    print(f"\n Renaming log-file entries: {subject} - {run}", "\n")

    event_replacements = {
        "Fix": "fixation",
        "FirstStimVisualOn": "first_word_onset",
        "FirstStimVisualOff": "first_word_offset",
        "StimVisualOn": "word_onset",
        "StimVisualOff": "word_offset",
        "LastStimVisualOn": "last_word_onset",
        "LastStimVisualOff": "last_word_offset",
        "Fix2DecisionON": "fix2panel_on",
        "Fix2DecisionOFF": "fix2panel_off",
        "PanelOn": "panel_on",
        "PanelOff": "panel_off",
        "KeyPress": "key_press",
        "FixFeedbackOn": "fix2feedback_on",
        "FixFeedbackOff": "fix2feedback_off",
    }

    run_log["Event"].replace(event_replacements, inplace=True)

    dim_after = len(run_log["Event"])
    if dim_before != dim_after:
        raise ValueError("The dimensions have been changed!")

    return run_log


def triggers_cleaning(
    events: np.ndarray, run_log: pd.DataFrame, triggers
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Clean the events and run_log by removing specific entries that are known to be shadowed due to trigger overlap.

    This function addresses the issue of overlapping triggers originating from stimuli pads and those sent from the paradigm.
    It removes specific entries from both the log-file and the event-file, particularly from the onset of the panel onwards.
    The entries removed are:
        1. panel_on
        2. key_press
        3. fix2feedback_on

    Note: This function should be executed after applying the map_log2triggers function.

    Parameters
    ----------
    events : np.ndarray
        The array of events, where each event is represented by a row.
    run_log : pd.DataFrame
        The DataFrame containing the log of events with an 'Event' column.
    triggers : object
        An object containing trigger values as attributes.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame]
        A tuple containing the cleaned events array and the modified run_log DataFrame.
    """

    # Identify spurious events
    shortest_event = 1
    n_short_events = np.sum(np.diff(events[:, 0]) < shortest_event)
    print(f"{n_short_events} spurious events found.")

    arr = events
    del events
    problematic_values = [
        "panel_off",
        "key_press",
        "fix2feedback_on",
        "fix2panel_off",
    ]

    for val in problematic_values:
        ttl_value = getattr(triggers, val)
        # Remove entries from the event-file
        index = np.where(arr[:, 2] == ttl_value)
        arr = np.delete(arr, index, axis=0)
        # Remove entries from the log-file
        run_log.drop(run_log[run_log["Event"] == val].index, inplace=True)

    # Also remove the multiple 255-events and only keep the latest to be in-sync with the log
    block_start_events = np.where(arr[:, 2] == 127)  # 255
    arr = np.delete(
        arr, block_start_events[0][0 : len(block_start_events[0]) - 1], axis=0
    )
    events = arr

    return events, run_log


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## functions for each test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test01_data_and_log_existence(
    path: Any, subject: str, run: str, join: Any, report: pd.DataFrame
) -> Tuple[Any, pd.DataFrame]:
    """
    TEST-01: Verify the existence and loadability of the data and log file into the workspace.

    This function checks if the raw data file for a given subject and run exists and can be loaded.
    It updates the report DataFrame to indicate the success of this test.

    Parameters
    ----------
    path : Any
        An object created with 'FetchPaths' (@class_repo) to manage file paths.
    subject : str
        The subject identifier, e.g., 'S01'.
    run : str
        The current run identifier.
    join : Any
        A function alias for os.path.join to concatenate file paths.
    report : pd.DataFrame
        A DataFrame that records the outcomes of various tests.

    Returns
    -------
    Tuple[Any, pd.DataFrame]
        A tuple containing:
        - raw: The MNE raw object for the specified run.
        - report: The updated DataFrame with the test outcome.
    """
    # Load the data
    raw_fname_in = join(path.to_data(), subject, "Raw", run + "_raw.fif")

    print(
        "",
        "--" * 20,
        f"\n Loading raw data: {subject} - {run}",
        "\n",
        "--" * 20,
    )
    raw = mne.io.read_raw_fif(
        raw_fname_in, allow_maxshield=True, preload=False, verbose="error"
    )
    report.loc[run, "test01"] = "OK"

    return raw, report


def test02_data_and_log_events_match(
    events: Any,
    report: pd.DataFrame,
    triggers: Any,
    run_log: pd.DataFrame,
    subject: str,
    run: str,
    path: Any,
) -> Tuple[
    pd.DataFrame,
    Any,
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, pd.Series],
]:
    """
    TEST-02 & 03: Verify the number of events per event of interest
    (e.g., fixations in log-file and TTL channel).

    This function checks the correspondence between the number of events recorded in the
    log file and the TTL channel for each event of interest. It updates the report DataFrame
    with the test results and returns the cleaned events and log data.

    Parameters
    ----------
    events : Any
        The MNE object containing event data.
    report : pd.DataFrame
        A DataFrame that records the outcomes of various tests.
    triggers : Any
        An object created with 'FetchTriggers' (@class_repo) to manage event triggers.
    run_log : pd.DataFrame
        The log file of the current run.
    subject : str
        The subject identifier, e.g., 'S01'.
    run : str
        The current run identifier.
    path : Any
        An object created with 'FetchPaths' (@class_repo) to manage file paths.

    Returns
    -------
    Tuple[pd.DataFrame, Any, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.Series]]
        A tuple containing:
        - report: The updated DataFrame with the test outcome.
        - events: The MNE object with mismatching entries and spurious triggers removed.
        - run_log: The DataFrame containing logs that match the events per run.
        - empirical: A dictionary containing the empirical values (TTL pulses) per event of interest.
        - matlab: A dictionary containing the Psychtoolbox time values per event of interest.
    """
    # Rename log-file entries
    run_log = map_log2triggers(run_log, triggers, subject, run)
    # Clean the triggers and remove equal entries from the log-file
    events, run_log = triggers_cleaning(events, run_log, triggers)
    # Initialize dictionaries
    empirical, matlab = [defaultdict(lambda: np.array([])) for _ in range(2)]
    # Get events of interest
    eois = [t for t in dir(triggers) if not t.startswith("__")]
    # Loop through the events of interest
    for counter, event_id in enumerate(sorted(eois)):
        ttl_value = getattr(triggers, eois[counter])
        # Extract the empirical time [ms]
        empirical[event_id] = events[(events[:, 2] == ttl_value), 0]
        # Extract the time from MATLAB [ms]
        matlab[event_id] = run_log["Time"][run_log["Event"] == event_id]

        # Check correspondence in the number of events between the two:
        # (e.g., fixation TTL vs fixation MATLAB)
        if len(empirical[event_id]) != len(matlab[event_id]):
            print(f"Log: {len(matlab[event_id])}")
            print(f"TTL: {len(empirical[event_id])}")
            raise ValueError(f"Mismatch in: {event_id}")

    # Check that the total number of events is the same
    if len(events) != len(run_log):
        raise ValueError(
            f"Mismatch in the total number of events.\n #Events: {len(events)} \n #Log-events: {len(run_log)}"
        )
    else:
        report.loc[run, "test02"] = "OK"

    return report, events, run_log, empirical, matlab


def test03_durations_match(
    empirical: Dict[str, np.ndarray],
    matlab: Dict[str, pd.Series],
    report: pd.DataFrame,
    run: str,
) -> None:
    """
    Calculate and compare selected durations from empirical and log-file values.

    This function computes the durations for specific events from both empirical data and MATLAB log files,
    compares them against expected values, and updates the report with the test results.

    Parameters
    ----------
    empirical : Dict[str, np.ndarray]
        A dictionary containing empirical values (TTL pulses) for each event of interest (e.g., fixation).
    matlab : Dict[str, pd.Series]
        A dictionary containing Psychtoolbox time values for each event of interest (e.g., fixation).
    report : pd.DataFrame
        A DataFrame used to record the test results.
    run : str
        The identifier for the current run.

    Raises
    ------
    ValueError
        If there is a problem with the durations, indicating a mismatch with expected values.
    """

    # Calculate durations in milliseconds
    empirical["fixation_time"] = (
        empirical["first_word_onset"] - empirical["fixation"]
    )
    matlab["fixation_time"] = (
        matlab["first_word_onset"].values - matlab["fixation"].values
    ) * 1e3

    empirical["word_ON"] = empirical["word_offset"] - empirical["word_onset"]
    matlab["word_ON"] = (
        matlab["word_offset"].values - matlab["word_onset"].values
    ) * 1e3

    empirical["soa"] = np.diff(empirical["word_offset"])[
        np.diff(empirical["word_offset"]) < 600
    ]
    matlab["soa"] = (
        np.diff(matlab["word_offset"])[np.diff(matlab["word_offset"]) < 4]
        * 1e3
    )

    dur_test = []
    print(40 * "--")
    print("---------- Expected VS Recorded durations --------------------")

    # Fixation time
    print(
        f"Fixation: \n*Fixation to first word onset\n\n\t Expected: {600} ms."
        f'\n\t Recorded: {np.mean(empirical["fixation_time"])} ms.'
        f'\n\t Log-files: {np.mean(matlab["fixation_time"])} ms.'
    )
    if np.abs(np.mean(empirical["fixation_time"]) - 600) < 20:
        tid = "OK"
        dur_test.append(0)
    else:
        tid = "REJECTED"
        dur_test.append(1)
    print(
        f'\n\t Difference of means: {np.abs(np.mean(empirical["fixation_time"]) - 600)} ms. \t\t\t {tid}'
    )

    # Word ON
    print(
        f"word ON: \n* Word offset - Word onset\n\n\t Expected: {250} ms."
        f'\n\t Recorded: {np.mean(empirical["word_ON"])} ms.'
        f'\n\t Log-files: {np.mean(matlab["word_ON"])} ms.'
    )
    if np.abs(np.mean(empirical["word_ON"]) - 250) < 20:
        tid = "OK"
        dur_test.append(0)
    else:
        tid = "REJECTED"
        dur_test.append(1)
    print(
        f'\n\t Difference of means: {np.abs(np.mean(empirical["word_ON"]) - 250)} ms. \t\t\t {tid}'
    )

    # SOA
    print(
        f"SOA: \n\t Expected: {500} ms."
        f'\n\t Recorded: {np.mean(empirical["soa"])} ms.'
        f'\n\t Log-files: {np.mean(matlab["soa"])} ms.'
    )
    if np.abs(np.mean(empirical["soa"]) - 500) < 20:
        tid = "OK"
        dur_test.append(0)
    else:
        tid = "REJECTED"
        dur_test.append(1)
    print(
        f'\n\t Difference of means: {np.abs(np.mean(empirical["soa"]) - 500)} ms. \t\t\t {tid}'
    )

    if np.sum(dur_test) == 0:
        report.loc[run, "test03"] = "OK"
    else:
        raise ValueError("Problem with the durations.")


def test04_calculate_r_squared(
    events: np.ndarray, run_log: pd.DataFrame, report: pd.DataFrame, run: str
) -> None:
    """
    Calculate the R-squared metric to evaluate the alignment between log-file times and TTL pulse events.

    This function performs a linear regression between the event times and the log-file times,
    and calculates the R-squared value to assess the degree of correspondence. If the R-squared
    value indicates a poor fit, an error is raised.

    Parameters
    ----------
    events : np.ndarray
        A 2D array where each row represents an event, and the first column contains the event times.
    run_log : pd.DataFrame
        A DataFrame containing the log-file data with a 'Time' column representing the event times.
    report : pd.DataFrame
        A DataFrame used to record the results of various tests, updated with the outcome of this test.
    run : str
        The identifier for the current run or session being processed.

    Raises
    ------
    ValueError
        If the R-squared value is not sufficiently close to zero, indicating a poor fit.
    """
    X = events[:, 0].reshape(-1, 1)
    Y = run_log["Time"].values.reshape(-1, 1)
    lm = LinearRegression()  # Create a LinearRegression object
    lm.fit(X, Y)  # Fit the model
    Y_pred = lm.predict(
        X
    )  # Predict the log-file times based on the event times

    if config.plot:
        plt.scatter(X, Y)
        plt.plot(
            X, Y_pred, color="red", label=f"R² = {r2_score(Y, Y_pred):.4f}"
        )
        plt.xlabel("Triggers")
        plt.ylabel("Log file")
        plt.legend()
        plt.show()

    if 1 - r2_score(Y, Y_pred) < 5e3:
        report.loc[run, "test04"] = "OK"
    else:
        raise ValueError("The R² value is not close to zero.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Export functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def export_events_and_logs(
    subject: str,
    path: Any,
    run: str,
    events: np.ndarray,
    run_log: pd.DataFrame,
) -> None:
    """
    Align the event files with the behavioral logs and save the updated files.

    This function modifies the event entries in the log file to reflect specific conditions
    and updates the event array with new trigger values. It then exports the updated event
    and log files to the specified directories.

    Parameters
    ----------
    subject : str
        The identifier for the subject whose data is being processed.
    path : Any
        An object that provides methods to construct file paths.
    run : str
        The identifier for the current run or session being processed.
    events : np.ndarray
        A 2D array where each row represents an event, and the third column contains the event triggers.
    run_log : pd.DataFrame
        A DataFrame containing the log-file data with columns for event details.

    Raises
    ------
    ValueError
        If there are empty values in the new event column, indicating an issue with event mapping.
    """
    make = os.makedirs
    check = os.path.exists

    # Define paths for events and logs
    path2events = join(path.to_data(), subject, "Events")
    path2log = join(path.to_data(), subject, "Log")

    # Create directories if they do not exist
    if not check(path2events):
        make(path2events)

    # Update log entries for the 5th word to indicate violation onset and offset
    violation_on_index = run_log[
        (run_log["Event"] == "word_onset") & (run_log["WordNum"] == "6")
    ].index
    violation_off_index = run_log[
        (run_log["Event"] == "word_offset") & (run_log["WordNum"] == "6")
    ].index

    run_log.loc[violation_on_index, "Event"] = "target_onset"
    run_log.loc[violation_off_index, "Event"] = "target_offset"

    # Initialize a new event column with default trigger values
    new_event_col = np.zeros(len(events))
    new_event_col[0] = 550  # Set the block-start value

    # Iterate through the log file to update event triggers
    for entry in range(1, len(events)):
        cond_id = run_log.iloc[entry]["Base_condition"]
        trial_type = run_log.iloc[entry]["Trial_type"]
        embedding = run_log.iloc[entry]["Embedding"]
        number = run_log.iloc[entry]["G_number"]
        event = run_log.iloc[entry]["Event"]

        # Match conditions with event_id dictionary to assign new trigger values
        for evID_key, evID_val in config.event_id.items():
            if all(
                [
                    list(evID_key.split("/"))[0].strip() == cond_id.strip(),
                    list(evID_key.split("/"))[1].strip() == trial_type.strip(),
                    list(evID_key.split("/"))[2].strip() == embedding.strip(),
                    list(evID_key.split("/"))[3].strip() == number.strip(),
                    list(evID_key.split("/"))[4].strip() == event.strip(),
                ]
            ):
                new_event_col[entry] = int(evID_val)

    # Ensure no empty values remain in the new event column
    if np.any(new_event_col == 0):
        raise ValueError("Cannot continue with updating the events file.")

    # Update the events array with new trigger values
    events[:, 2] = new_event_col

    # Export the updated event file and log file
    ev_file = join(path2events, run + "-eve.fif")
    mne.write_events(ev_file, events)

    log_file = join(path2log, run + "-log.csv")
    run_log.to_csv(log_file, sep="\t")
