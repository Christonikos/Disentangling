#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Space Projection (SSP) Computation
=======================================

This script computes Signal Space Projections for ECG and EOG artifact removal.

Processing Steps
--------------
1. Loads continuous MEG/EEG data
2. Computes rejection thresholds using autoreject
3. Calculates SSP projectors for ECG artifacts
4. Calculates SSP projectors for EOG artifacts
5. Saves projectors and visualization plots

Input Data
---------
- Raw or preprocessed continuous data (.fif files)
- Can process either:
  * Raw Maxwell filtered data
  * Preprocessed data

Command Line Arguments
--------------------
-d, --data : str, default='Raw'
    Type of data to process: 'Raw' or 'Preprocessed'

Output
------
- SSP projection vectors saved as -proj.fif files
- Visualization plots of ECG and EOG projectors
- Saved in the subject's data directory

Technical Details
---------------
- Computes separate projectors for grad/mag/eeg channels
- Uses autoreject for threshold determination
- Generates one projector per signal type (ECG/EOG)
- Creates topographic visualizations

Author: Christos-Nikolaos Zacharopoulos

"""


# Standard library imports
import sys
import os.path as op
import argparse

# MNE-specific imports
import mne
from mne.parallel import parallel_func
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

# Third-party utilities
import autoreject

# Local imports
sys.path.append("../")
import config
from repos import func_repo as f


def run_ssp(subject: str) -> None:
    """
    Compute SSP projectors for a single subject.

    Parameters
    ----------
    subject : str
        Subject identifier

    Processing Steps
    --------------
    1. Loads continuous data
    2. Sets up channel types
    3. Computes rejection thresholds
    4. Calculates ECG projectors
    5. Calculates EOG projectors
    6. Saves projectors and visualizations

    Notes
    -----
    - Handles both ECG and EOG artifacts
    - Creates one projector per channel type
    - Saves visualization plots for quality check
    - Uses autoreject for threshold determination
    """
    print(40 * "--")
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.data_path, subject, args.data)
    print(40 * "--")

    # Load data and prepare file names
    run = f.fetch_runs(config.path, subject)[-1]

    if config.use_maxwell_filter:
        if args.data == "Preprocessed":
            extension = run + "_preprocessed"
        else:
            extension = run + "_sss_raw"
    else:
        extension = run + "_filt_raw"

    raw_fname_in = op.join(
        meg_subject_dir, config.base_fname.format(**locals())
    )

    extension = run + "_ssp-proj"
    proj_fname_out = op.join(
        meg_subject_dir, config.base_fname.format(**locals())
    )

    print("Input: ", raw_fname_in)
    print("Output: ", proj_fname_out)

    # Load and prepare raw data
    raw = mne.io.read_raw_fif(raw_fname_in)
    bads = raw.info["bads"]
    peripheral = ["EEG064", "EEG061", "EEG062"]
    raw.info["bads"] = [b for b in bads if b not in peripheral]

    try:
        raw.set_channel_types({"EEG064": "ecg"})
    except BaseException:
        pass

    # Compute rejection thresholds
    print(40 * "--")
    print("Getting the rejection threshold for the SSP")
    tstep = 1.0
    events = mne.make_fixed_length_events(raw, duration=tstep)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=tstep, baseline=(0, 0), preload=True
    )
    epochs.drop_channels(bads)
    reject = autoreject.get_rejection_threshold(epochs)
    print(40 * "--")

    # Compute and save ECG projectors
    print("Computing SSPs for ECG")
    ecg_projs, ecg_events = compute_proj_ecg(
        raw, n_grad=1, n_mag=1, n_eeg=1, average=True, reject=reject
    )

    # Save ECG projector plots
    ecg_fig = mne.viz.plot_projs_topomap(ecg_projs, raw.info, show=False)
    path2figs = config.join(
        config.path.to_figures(), "check_data_quality", "SSP", "ECG", args.data
    )
    if not config.exists(path2figs):
        config.make(path2figs, exist_ok=True)
    fname = config.join(path2figs, subject + "_ecg.png")
    ecg_fig.savefig(fname)

    # Compute and save EOG projectors
    print("Computing SSPs for EOG")
    eog_projs, eog_events = compute_proj_eog(
        raw,
        n_grad=1,
        n_mag=1,
        n_eeg=1,
        average=True,
    )

    # Save EOG projector plots
    eog_fig = mne.viz.plot_projs_topomap(eog_projs, raw.info, show=False)
    path2figs = config.join(
        config.path.to_figures(), "check_data_quality", "SSP", "EOG", args.data
    )
    if not config.exists(path2figs):
        config.make(path2figs, exist_ok=True)
    fname = config.join(path2figs, subject + "_eog.png")
    eog_fig.savefig(fname)

    # Save all projectors
    mne.write_proj(proj_fname_out, eog_projs + ecg_projs)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SSP computation parameters")
    parser.add_argument(
        "-d",
        "--data",
        default="Raw",
        help="Select the data type. Options=[Raw, Preprocessed]",
    )
    args = parser.parse_args()

    if config.use_ssp:
        [run_ssp(subject) for subject in config.subjects_list]
