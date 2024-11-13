#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trigger Validation and Synchronization Script

This script performs comprehensive validation of trigger events in neural data by running
four independent tests:
1. Data and log file existence/loading
2. Event count matching between log files and TTL channel
3. Duration validation between recorded and expected values
4. Correlation analysis (R²) between event timings

The script processes multiple subjects and runs, validating trigger data integrity
and synchronization between recorded TTL signals and experimental log files.

Dependencies:
    - MNE-Python
    - NumPy
    - Pandas
    - Custom repositories (func_repo, class_repo, check_timings_func_repo)
    
Author: Christos-Nikolaos Zacharopoulos

    
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================

# Standard library
import os
import pandas as pd
import numpy as np


# MNE specific
import mne
from mne import viz

# Local imports
from repos import class_repo
import config as c
from repos import check_timings_func_repo as check
from repos import func_repo as f


# =============================================================================
# ALLIASES
# =============================================================================
join = os.path.join

# =============================================================================
# GLOBALS
# =============================================================================
TRIGGER_CHANNEL = "STI101"  # Name of the TTL trigger channel in the raw data
VALID_TRIGGER_VALUES = {1, 10, 15, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110}
MIN_DURATION_DEFAULTS = {
    ("S02", "run_06"): 0.001,
    ("ICM01", "run_02"): 0.0001,
    "default": 0.002,
}


def drop_deviant_values(events: np.ndarray) -> np.ndarray:
    """
    Remove unexpected trigger values from the events array.

    Args:
        events (np.ndarray): MNE events array of shape (n_events, 3) containing trigger information
            [timestamp, duration, trigger_value]

    Returns:
        np.ndarray: Cleaned events array containing only valid trigger values
    """
    recorded_values = set(events[:, 2])
    deviant_values = [
        val for val in recorded_values if val not in VALID_TRIGGER_VALUES
    ]

    mask = np.isin(events[:, 2], list(VALID_TRIGGER_VALUES))
    return events[mask]


# get paths and available runs
path = c.FetchPaths(c.root, c.project_name)


# loop through subjects
for subject in c.subjects_list:
    runs = f.fetch_runs(path, subject)
    log_fname = f.fetch_logs(path, subject)
    triggers = class_repo.FetchTriggers()

    # initialize dataframe to hold test results
    colnames = ["test0" + str(n_test) for n_test in range(1, 5)]
    report = pd.DataFrame(index=runs, columns=colnames)
    report.index.name = subject

    if subject == "S02":
        # load the log-file:
        log = pd.read_csv(
            join(path.to_data(), subject, "Log", log_fname),
            encoding="ISO-8859-1",
            engine="python",
        )
        log = log.rename(
            columns={
                "Base+AF8-condition": "Base_condition",
                "G+AF8-number": "G_number",
                "Trial+AF8-type": "Trial_type",
            }
        )
    else:
        log = pd.read_csv(
            join(path.to_data(), subject, "Log", log_fname), sep="\t"
        )

    # perform all the tests per run -if all is succesfull, create BIDS dir.
    for idx, run in enumerate(runs):
        # update counter indexing
        counter = idx + 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## TEST01 - Check if data can be found and loaded
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        raw, report = check.test01_data_and_log_existance(
            path, subject, run, join, report
        )
        if report["test01"][run] == "OK":
            print("test#1 succesfull - Data and log-file found and loaded.")

        if subject in ("S02") and run in ("run_06"):
            min_duration = 0.001  # Adjusted threshold for S02/run_06
        elif subject == "ICM01" and run == "run_02":
            min_duration = 0.0001  # Adjusted threshold for ICM01/run_02
        else:
            min_duration = 0.002  # Default minimum duration threshold

        # load events
        events = mne.find_events(
            raw,
            stim_channel="STI101",
            verbose=True,
            min_duration=min_duration,
            consecutive="increasing",
            uint_cast=False,
            mask=256 + 512 + 1024 + 2048 + 4096 + 8192 + 16384 + 32768,
            mask_type="not_and",
            shortest_event=0.1,
        )

        events = drop_deviant_values(events)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## TEST02 - Check #events per event of interest (e.g: fixations in log-file and ttl channel)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        run_log = log[log["Block"] == counter]
        # drop the Blockstart if it exists in the run log
        run_log = run_log[run_log.Event != "BlockStart"]

        (
            report,
            events,
            run_log,
            empirical,
            matlab,
        ) = check.test02_data_and_log_events_match(
            events, report, triggers, run_log, subject, run, path
        )
        if report["test02"][run] == "OK":
            print(
                "test#2 succesfull - same #events between log and trigger-channel."
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## TEST03 - Check durations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        check.test03_durations_match(empirical, matlab, report, run)

        if report["test03"][run] == "OK":
            print("test#3 succesfull - No deviations in the durations.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## TEST04 - Calculate the R^2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        check.test04_calculate_r_squared(events, run_log, report, run)

        if report["test04"][run] == "OK":
            print("test#4 succesfull - R^2 close to one.")

        if c.plot:
            # visualize the corrected event-file
            viz.plot_events(events, raw.info["sfreq"])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## EXPORT - Event file and logs per run (updated and synchronized)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        check.export_events_and_logs(subject, path, run, events, run_log)
