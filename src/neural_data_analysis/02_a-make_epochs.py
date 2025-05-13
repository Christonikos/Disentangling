#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG/EEG Epoch Construction
=========================

This script constructs epochs from continuous MEG/EEG data using predefined events.

Processing Steps
--------------
1. Loads preprocessed or raw MEG/EEG data
2. Concatenates multiple runs
3. Applies automatic artifact rejection
4. Creates epochs around specified events
5. Adds detailed metadata to epochs
6. Saves epoched data to disk

Input Data
---------
- Raw or preprocessed continuous data (.fif files)
- Event files (-eve.fif)
- Log files with trial information
- Behavioral response files

Command Line Arguments
--------------------
-eoi, --events_of_interest : list of str, default=['first_word_onset']
    Events to epoch around (e.g., ['first_word_onset', 'second_word_onset'])
-data, --data_to_epoch : str, default='preprocessed'
    Type of data to epoch: 'raw' (Maxwell filtered) or 'preprocessed'

Output
------
Epoched data saved as -epo.fif files with associated metadata including:
- Trial information
- Stimulus properties
- Behavioral responses
- Linguistic features

Technical Details
---------------
- Supports parallel processing across subjects
- Handles both MEG and EEG channels
- Common Average Reference for EEG

Author: Christos-Nikolaos Zacharopoulos

"""


# Standard library imports
import os
import sys
import os.path as op
import argparse

# Third-party scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm

# MNE-specific imports
import mne
from mne.parallel import parallel_func

# Other third-party utilities
from termcolor import colored

# Local imports
sys.path.append("..")
import config as c
from repos import func_repo as f


def eeg_deviants(raw: mne.io.Raw) -> None:
    """
    Detect and interpolate deviant EEG channels after Common Average Reference.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to check and modify

    Notes
    -----
    - Uses variance-based detection (threshold: 6 * median)
    - Modifies raw object in place
    - Resets bad channel list after interpolation
    """
    print(colored(40 * "**", "green"))
    print(colored("Re-interpolating deviant sensors", "green"))
    eeg_names = [i for i in raw.info.ch_names if i.startswith("EEG")]
    eeg = raw.copy().get_data(picks="eeg")
    variance = np.var(eeg, axis=1)
    deviants = np.where(variance > 6 * np.median(variance))[0]
    bads = [eeg_names[d] for d in deviants]
    raw.info["bads"] = bads
    print(raw.info["bads"])
    raw.interpolate_bads(reset_bads=True)
    print(colored(40 * "**", "green"))


def run_epochs(subject: str) -> None:
    """
    Create epochs for a single subject from continuous data.

    Parameters
    ----------
    subject : str
        Subject identifier

    Processing Steps
    --------------
    1. Loads continuous data and events
    2. Applies EEG preprocessing if needed
    3. Concatenates multiple runs
    4. Creates metadata from log files
    5. Constructs epochs with metadata
    6. Saves epoched data

    Notes
    -----
    - Epochs are created around events specified in command line arguments
    - Metadata includes extensive trial and stimulus information
    - No baseline correction is applied
    - Optional visualization if config.plot is True
    """
    print("\n", 40 * "--")
    print(
        f"Subject: {subject},\
          Event of interest: {args.events_of_interest[0]},\
              {args.data_to_epoch}"
    )
    print("\n", 40 * "--")

    # =========================================================================
    # DIRECTORIES
    # =========================================================================
    if args.data_to_epoch == "raw":
        meg_subject_dir = op.join(c.data_path, subject, "Raw")
    elif args.data_to_epoch == "preprocessed":
        meg_subject_dir = op.join(c.data_path, subject, "Preprocessed")

    eve_subject_dir = op.join(c.data_path, subject, "Events")
    log_subject_dir = op.join(c.data_path, subject, "Log")
    epc_subject_dir = op.join(
        c.data_path, subject, "Epochs", args.data_to_epoch
    )
    beh_log_dir = op.join(c.data_path, subject, "Behavioral_Responses")

    if not c.exists(epc_subject_dir):
        c.make(epc_subject_dir)

    raw_list = list()
    events_list = list()
    logs_list = pd.DataFrame()

    # Get the available runs per subject
    runs = f.fetch_runs(c.path, subject)

    for run in tqdm(runs):
        if args.data_to_epoch == "raw":
            extension = run + "_sss_raw"
        elif args.data_to_epoch == "preprocessed":
            extension = run + "_preprocessed"

        raw_fname_in = op.join(
            meg_subject_dir, c.base_fname.format(**locals())
        )

        info = mne.io.read_info(raw_fname_in)

        eve_fname = op.join(eve_subject_dir, run + "-eve.fif")
        log_fname = op.join(
            log_subject_dir, f.fetch_logs(c.path, subject, run=run)
        )

        # ====================================================================
        # LOAD DATA
        # ====================================================================
        print("\n", 40 * "--")
        print(f"Loading {args.data_to_epoch} data")
        print("Input: ", "", raw_fname_in, "\n\t", eve_fname)
        raw = mne.io.read_raw_fif(
            raw_fname_in, allow_maxshield=False, preload=True
        )
        raw.del_proj()
        raw.set_annotations(None)
        print("\n", 40 * "--")
        # ====================================================================

        if args.data_to_epoch == "raw":
            # ================================================================
            # INTERPOLATE EEG DATA
            # ================================================================
            print("Interpolating bad EEGs")
            raw.interpolate_bads(reset_bads=True)
            ########################
            # Set common Reference #
            ########################
            raw = raw.copy().set_eeg_reference("average")
            eeg_deviants(raw)

        events = mne.read_events(eve_fname)
        events_list.append(events)

        raw_list.append(raw)
        # load the log-files per run:
        run_log = pd.read_csv(log_fname, sep="\t")

        # load the behavioral log-files per run
        block = "".join(c for c in run if c.isdigit())
        b = str(int(block))
        curr_file = [
            file
            for file in os.listdir(beh_log_dir)
            if ("block_" + b + "_") in file
        ]
        beh_log = pd.read_csv(
            c.join(beh_log_dir, curr_file[0]), "\t", encoding="latin-1"
        )
        print("updating the log file")
        run_log = f.update_log_file(run_log, beh_log)
        logs_list = logs_list.append(run_log, ignore_index=True)

    print("Concatenating runs")
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)

    picks = mne.pick_types(
        raw.info, meg=True, eeg=True, stim=True, eog=True, exclude=()
    )

    metadata = {
        # 'event_time': events[:, 0] / raw.info['sfreq'],
        #        'event_number': range(len(events)),
        "Event": logs_list["Event"],
        "Block": logs_list["Block"],
        "Trial": logs_list["Trial"],
        "StimNum": logs_list["StimNum"],
        "WordNum": logs_list["WordNum"],
        "Condition": logs_list["Condition"],
        "Token": logs_list["Token"],
        "Base_condition": logs_list["Base_condition"],
        "Embedding": logs_list["Embedding"],
        "G_number": logs_list["G_number"],
        "Trial_type": logs_list["Trial_type"],
        "pair_index": logs_list["pair_index"],
        "n1": logs_list["n1"],
        "pp": logs_list["pp"],
        "n2": logs_list["n2"],
        "v1": logs_list["v1"],
        "v2_n3_adverb": logs_list["v2_n3_adverb"],
        "sentence": logs_list["sentence"],
        "violIndex": logs_list["violIndex"],
        "pp_freq": logs_list["pp_freq"],
        "n1_freq": logs_list["n1_freq"],
        "n2_freq": logs_list["n2_freq"],
        "v1_freq": logs_list["v1_freq"],
        "v2_n3_adverb_freq": logs_list["v2_n3_adverb_freq"],
        "pp_nl": logs_list["pp_nl"],
        "n1_nl": logs_list["n1_nl"],
        "n2_nl": logs_list["n2_nl"],
        "v1_nl": logs_list["v1_nl"],
        "v2_n3_adverb_nl": logs_list["v2_n3_adverb_nl"],
        "violation_type": logs_list["violation_type"],
        "RT": logs_list["RT"],
        "Behavioral": logs_list["Behavioral"],
        "response": logs_list["subject_response"],
    }
    metadata = pd.DataFrame(
        metadata,
    )
    metadata.head()

    # Set the events of interest to epoch_around
    eoi = args.events_of_interest

    event_id = dict(f.parse_event_id(c, eoi))
    metadata = metadata[metadata["Event"].isin(eoi)]

    # User-feedback:
    print(
        "\n",
        40 * "--",
        f"\n Subject: {subject} \n",
        f"\n Events of interest: {args.events_of_interest}",
        "\n",
        40 * "--",
    )

    # get the tmax: Duration of sentence + ISI to panel
    tmax = 1.5

    # Epoch the data
    print(f"Epoching, tmin: {c.tmin}, tmax: {tmax}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        c.tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=False,
        reject=None,
    )
    epochs.metadata = metadata

    print("  Writing epochs to disk")
    if args.data_to_epoch == "raw":
        epochs_fname = op.join(
            epc_subject_dir,
            "_maxwell_filtered_".join(e for e in eoi) + "-epo.fif",
        )
    elif args.data_to_epoch == "preprocessed":
        epochs_fname = op.join(
            epc_subject_dir, "_preprocessed_".join(e for e in eoi) + "-epo.fif"
        )

    print("Output: ", epochs_fname)
    epochs.save(epochs_fname, overwrite=True)

    if c.plot:
        # epochs.plot()
        epochs["GSLD"].plot_image(combine="gfp", sigma=2.0, cmap="YlGnBu_r")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Epoching parameters")
    parser.add_argument(
        "-eoi",
        "--events_of_interest",
        nargs="+",
        default=["target_onset"], # target_onset, first_word_onset
        help="Select events to epoch.",
    )
    parser.add_argument(
        "-data",
        "--data_to_epoch",
        default="preprocessed",
        help="[raw, preprocessed]",
    )
    args = parser.parse_args()

    # Set up parallel processing
    n_jobs = 1
    print(args, n_jobs)

    # Run parallel processing
    for data_to_epoch in ["raw", "preprocessed"]:

        parallel, run_func, _ = parallel_func(run_epochs, n_jobs=n_jobs)
        parallel(run_func(subject) for subject in c.subjects_list)
