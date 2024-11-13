#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced MEG/EEG Data Preprocessing
=================================

This script performs preprocessing operations on Maxwell filtered MEG/EEG data.

Processing Steps
--------------
For all sensor types:
1. Robust scaling transformation (without actual scaling)
   - Uses RobustScaler from sklearn
   - Applies clipping at -3/3 bounds
2. Optional Gaussian kernel smoothing
   - Kernel size configurable via command line argument

For EEG specifically:
3. Bad channel interpolation using spherical splines
4. Common Average Reference (CAR)
5. Additional deviant channel detection and interpolation

Input Data Requirements
---------------------
- Data must be already:
  * Notch filtered
  * Detrended
  * Bandpass filtered
  * Maxwell filtered

Command Line Arguments
--------------------
-k, --kernel : int, default=10
    Smoothing kernel size in milliseconds (options: 10 or 25)

Technical Details
---------------
- Processes data at individual run level
- Supports parallel processing across subjects
- Preserves ECG/EOG channels during preprocessing
- Handles special cases for subjects with known EEG issues

Output
------
Preprocessed data saved with '_preprocessed' suffix in subject's Preprocessed directory.

Author: Christos-Nikolaos Zacharopoulos
"""


# Standard library imports
import sys
import argparse
from typing import List, Tuple, Optional

# Third-party scientific computing
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler

# MNE-specific imports
import mne
from mne.io import Raw
from mne.parallel import parallel_func

# Other third-party utilities
from termcolor import colored

# Local imports
sys.path.append("..")
import config as c
from repos import func_repo as f

# Type aliases
ChannelData = np.ndarray
ChannelList = List[str]


def replace_ecg_and_eog(
    ecg: ChannelData, eog: ChannelData, raw: Raw, subject: str
) -> Raw:
    """
    Replace ECG and EOG channels in raw data with preserved versions.

    This function preserves the original ECG/EOG data that might be affected by
    preprocessing steps applied to other channels.

    Parameters
    ----------
    ecg : np.ndarray
        Original ECG channel data
    eog : np.ndarray
        Original EOG channel data
    raw : mne.io.Raw
        Raw data object to modify
    subject : str
        Subject identifier

    Returns
    -------
    raw : mne.io.Raw
        Modified raw object with restored ECG/EOG channels
    """
    eog_locations = [i for i, j in enumerate(raw.info.ch_names) if "EOG" in j]
    ecg_locations = [i for i, j in enumerate(raw.info.ch_names) if "ECG" in j]
    data = raw.copy().get_data()

    data[eog_locations, :] = eog
    data[ecg_locations, :] = ecg
    raw._data = data

    return raw


def eeg_deviants(raw: Raw) -> None:
    """
    Detect and interpolate deviant EEG channels after CAR.

    Uses variance-based detection to identify channels that deviate significantly
    from the median variance across channels (threshold: 6 * median).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to check and modify

    Notes
    -----
    - Modifies raw object in place
    - Resets bad channel list after interpolation
    - Prints information about detected bad channels
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


def run_advanced_preprocessing(subject: str) -> None:
    """
    Apply advanced preprocessing steps to all runs of a subject.

    Processing steps:
    1. Robust scaling and clipping
    2. EEG channel interpolation
    3. Common Average Reference
    4. Additional EEG deviant detection
    5. ECG/EOG channel preservation

    Parameters
    ----------
    subject : str
        Subject identifier

    Notes
    -----
    - Creates Preprocessed directory if it doesn't exist
    - Handles both MEG and EEG data
    - Preserves original ECG/EOG data
    - Saves preprocessed data with '_preprocessed' suffix
    """
    print("Processing subject: %s" % subject)

    meg_subject_dir = c.join(c.data_path, subject, "Raw")
    prep_subject_dir = c.join(c.data_path, subject, "Preprocessed")

    if not c.exists(prep_subject_dir):
        c.make(prep_subject_dir)

    for run in f.fetch_runs(c.path, subject):
        extension = run + "_sss_raw"
        raw_fname_in = c.join(meg_subject_dir, c.base_fname.format(**locals()))

        extension = run + "_preprocessed"
        raw_fname_out = c.join(
            prep_subject_dir, c.base_fname.format(**locals())
        )

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        # INPUT: LOADS THE FILTERED .FIF FILES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw = mne.io.read_raw_fif(
            raw_fname_in, allow_maxshield=True, preload=True
        )

        # remove projs
        raw = raw.del_proj("all")
        # get the ecg and eog data, we will add them back to the data after clipping
        ecg = raw.copy().get_data(picks=["ecg"])
        eog = raw.copy().get_data(picks=["eog"])
        ################################################################

        ############################
        # Robust Scaling Transform #
        ############################
        print("Scaling to IQR range")
        data = raw.copy().get_data()
        transformer = RobustScaler().fit(np.transpose(data))
        data_scaled = np.transpose(
            transformer.transform(np.transpose(data))
        )  # num_channels X num_timepoints

        ############
        # CLIPPING #
        ############
        print("Clipping")
        lower, upper = -3, 3
        data_scaled[data_scaled > upper] = upper
        data_scaled[data_scaled < lower] = lower
        data = np.transpose(
            transformer.inverse_transform(np.transpose(data_scaled))
        )
        raw._data = data

        ###################
        # INTERPOLATE EEG #
        ###################
        print("Interpolating bad EEGs")
        print(raw.info["bads"])
        raw.interpolate_bads(reset_bads=True)

        ########################
        # Set common Reference #
        ########################
        raw = raw.copy().set_eeg_reference("average")

        # if subject in c.broken_eeg:
        eeg_deviants(raw)

        # replace the eog and ecg so the SSP works
        # if subject != 'ICM01':
        #    raw = replace_ecg_and_eog(ecg, eog, raw, subject)

        # OUTPUT: SAVES PREPROCESSED DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw.save(raw_fname_out, overwrite=True)
        ################################################################


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Advanced MEG/EEG preprocessing script"
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=int,
        default=10,
        help="Select the smoothing kernel in [ms]. Options=[25, 10]",
    )
    args = parser.parse_args()

    # Run parallel processing
    n_jobs = -1
    parallel, run_func, _ = parallel_func(
        run_advanced_preprocessing, n_jobs=n_jobs
    )
    parallel(run_func(subject) for subject in c.subjects_list)
