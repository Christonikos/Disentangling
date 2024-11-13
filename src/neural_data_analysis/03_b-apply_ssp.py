#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Signal Space Projections
============================

This script applies pre-computed SSP projections to epoched MEG/EEG data.

Processing Steps
--------------
1. Loads epoched MEG/EEG data
2. Reads pre-computed SSP projectors
3. Applies projectors to remove artifacts
4. Saves cleaned epochs

Input Data
---------
- Epoched data files (-epo.fif)
- SSP projection files (-proj.fif)
- Can process either:
  * Raw Maxwell filtered epochs
  * Preprocessed epochs

Command Line Arguments
--------------------
-eoi, --events_of_interest : list of str, default=['first_word_onset']
    Events to process (e.g., ['first_word_onset', 'second_word_onset'])
-d, --data : str, default='Raw'
    Type of data to process: 'Raw' or 'Preprocessed'

Output
------
- Cleaned epoched data saved with '_ssp_cleaned-epo.fif' suffix
- Saved in subject's Epochs/SSP directory

Technical Details
---------------
- Applies both ECG and EOG projectors
- Handles multiple epoch types
- Supports parallel processing
- Preserves epoch metadata

Author: Christos-Nikolaos Zacharopoulos

"""

# Standard library imports
import os
import argparse
from typing import List


# MNE-specific imports
import mne
from mne.parallel import parallel_func

# Local imports
import config as c
from repos import func_repo as f


def load_epochs(epc_subject_dir: str) -> mne.Epochs:
    """
    Load epoched data from a subject directory.

    Parameters
    ----------
    epc_subject_dir : str
        Path to directory containing epoch files

    Returns
    -------
    mne.Epochs
        Loaded epochs object

    Notes
    -----
    - Loads all epoch files in directory
    - Returns last file which contains all epochs
    """
    files = []
    for element in c.see(epc_subject_dir):
        file = c.join(epc_subject_dir, element)
        if os.path.isfile(file):
            files.append(file)
    # MNE splits the data and the last run contains all epochs
    epochs = mne.read_epochs(files[-1], preload=True)
    return epochs


def apply_ssp(subject: str) -> None:
    """
    Apply SSP projections to a subject's epoched data.

    Parameters
    ----------
    subject : str
        Subject identifier

    Processing Steps
    --------------
    1. Loads epoched data
    2. Reads SSP projectors
    3. Applies projectors
    4. Saves cleaned epochs

    Notes
    -----
    - Creates SSP output directory if needed
    - Applies all available projectors
    - Saves with '_ssp_cleaned' suffix
    """
    print("Processing subject: %s" % subject)
    epc_subject_dir = c.join(c.data_path, subject, "Epochs", args.data.lower())
    meg_subject_dir = c.join(c.data_path, subject, args.data)

    # Load epochs and SSP projectors
    epochs = load_epochs(epc_subject_dir)

    # Set up input/output paths
    extension = "_".join(e for e in eoi) + "-epo"
    fname_in = c.join(epc_subject_dir, c.base_fname.format(**locals()))

    run = f.fetch_runs(c.path, subject)[-1]
    extension = run + "_ssp-proj"
    proj_fname_in = c.join(meg_subject_dir, c.base_fname.format(**locals()))

    print("Reading SSP projections from : %s" % proj_fname_in)

    # Apply projectors
    projs = mne.read_proj(proj_fname_in)
    epochs.add_proj(projs).apply_proj()

    # Save cleaned epochs
    out_dir = c.join(c.data_path, subject, "Epochs", args.data.lower(), "SSP")
    if not c.exists(out_dir):
        c.make(out_dir, exist_ok=True)

    extension = "_".join(e for e in eoi) + "_ssp_cleaned-epo"
    fname_out = c.join(out_dir, c.base_fname.format(**locals()))

    print("Input: ", fname_in)
    print("Output: ", fname_out)
    print("Saving epochs")
    epochs.save(fname_out, overwrite=True)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Apply SSP projections to epochs"
    )
    parser.add_argument(
        "-eoi",
        "--events_of_interest",
        nargs="+",
        default=["first_word_onset"],
        help="Select events to process.",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="Raw",
        help="Type of data to process [Raw/Preprocessed].",
    )
    args = parser.parse_args()
    eoi = args.events_of_interest

    # Run parallel processing
    if c.use_ssp:
        parallel, run_func, _ = parallel_func(apply_ssp, n_jobs=c.N_JOBS)
        parallel(run_func(subject) for subject in c.subjects_list)
