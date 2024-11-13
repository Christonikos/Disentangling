#!/usr/bin/env python3
"""
Automatically identify deviant MEG/EEG channels prior to Maxwell filtering.

This script performs the following steps for each subject and run:
1. Loads and preprocesses raw MEG/EEG data
2. Applies detrending, filtering, and SSP projection
3. Detects channels with deviant variance
4. Saves the bad channel information for further processing

The script can process multiple subjects in parallel using the configuration 
specified in config.py.

Author: Christos-Nikolaos Zacharopoulos

"""


# ============================================================================
# Imports
# ============================================================================

# Standard library
import sys
from collections import defaultdict
import pickle
from typing import Dict, List, Tuple, DefaultDict, Optional, Union

# Third-party
import numpy as np
from scipy import signal
from tqdm import tqdm

# MNE specific
import mne
from mne.io import Raw
from mne.parallel import parallel_func
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import autoreject

# Local imports
sys.path.append("..")
from repos import func_repo as f
import config as c

# Custom types
SensorDict = Dict[str, Union[List[str], np.ndarray, float]]
DeviantDict = Dict[str, List[str]]

# ============================================================================
# Data Loading Functions
# ============================================================================


def load_raw_per_run(subject: str, run: str, meg_subject_dir: str) -> Raw:
    """
    Load raw MEG/EEG data for a specific subject and run.

    Parameters
    ----------
    subject : str
        Subject identifier
    run : str
        Run identifier
    meg_subject_dir : str
        Path to the subject's MEG data directory

    Returns
    -------
    raw : mne.io.Raw
        Raw MEG/EEG data
    """
    print(f"Processing subject: {subject}-{run}")
    extension = run + "_raw"
    raw_fname_in = c.join(meg_subject_dir, c.base_fname.format(**locals()))
    print("Input: ", raw_fname_in)

    # INPUT: LOAD THE RAW .FIF FILES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ################################################################
    raw = mne.io.read_raw_fif(
        raw_fname_in,
        allow_maxshield=c.allow_maxshield,
        preload=True,
        verbose="error",
    )

    return raw


# ============================================================================
# Preprocessing Functions
# ============================================================================


def detrend_data(raw: Raw) -> Raw:
    """
    Remove linear trends from raw MEG/EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data

    Returns
    -------
    raw : mne.io.Raw
        Detrended MEG/EEG data
    """
    print(40 * "**")
    print("Detrending the data at the run level")
    print(40 * "**")
    data = raw._data
    data = signal.detrend(data)
    raw._data = data

    return raw


def notch_filter(raw: Raw) -> Raw:
    """
    Apply notch filter to remove line noise and harmonics.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data

    Returns
    -------
    raw : mne.io.Raw
        Filtered MEG/EEG data
    """
    print(40 * "**")
    notch = 50
    print(f"applying notch filter at {notch} Hz and 3 harmonics")
    notch_freqs = [notch, notch * 2, notch * 3, notch * 4]
    raw = raw.notch_filter(notch_freqs)
    print(40 * "**")

    return raw


def bandpass_filter(raw: Raw) -> Raw:
    """
    Apply bandpass filter to the data channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data

    Returns
    -------
    raw : mne.io.Raw
        Filtered MEG/EEG data
    """
    print(40 * "**")
    # Band-pass the data channels (MEG and EEG)
    print("Filtering data between %s and %s (Hz)" % (c.l_freq, c.h_freq))
    raw.filter(
        c.l_freq,
        c.h_freq,
        l_trans_bandwidth=c.l_trans_bandwidth,
        h_trans_bandwidth=c.h_trans_bandwidth,
        filter_length="auto",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
    )
    print(40 * "**")

    return raw


# ============================================================================
# SSP and Artifact Rejection
# ============================================================================


def calculate_ssp_threshold(raw: Raw) -> Dict[str, float]:
    """
    Get the rejection threshold for the SSP using AUTOREJECT.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data

    Returns
    -------
    reject : float
        Rejection threshold for the SSP
    """
    print(40 * "**")
    print("Getting the rejection threshold for the SSP using AUTOREJECT")
    print(40 * "**")
    tstep = 1.0
    events = mne.make_fixed_length_events(raw, duration=tstep)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=tstep, baseline=(0, 0))
    reject = autoreject.get_rejection_threshold(epochs)

    return reject


def apply_ssp_on_raw(raw: Raw, subject: str, run: str) -> Raw:
    """
    We do that here because we don't want channels to be marked as deviant
    due to the presence of eye-blinks or heart-beats.
    """
    print(40 * "**")
    print("Computing SSPs for ECG")
    ecg_projs, ecg_events = compute_proj_ecg(
        raw, n_grad=1, n_mag=1, n_eeg=1, average=True
    )

    print("Computing SSPs for EOG")
    eog_projs, eog_events = compute_proj_eog(
        raw,
        n_grad=1,
        n_mag=1,
        n_eeg=1,
        average=True,
    )

    print(f"Processing subject: {subject}-{run}")
    if ecg_projs is None:
        ecg_projs = []
    if eog_projs is None:
        eog_projs = []
    try:
        projs = eog_projs + ecg_projs
    except ValueError:
        projs = []

    raw.add_proj(projs).apply_proj()
    print(40 * "**")

    return raw


# ============================================================================
# Channel Analysis Functions
# ============================================================================


def get_data_per_sensor_type(
    raw: Raw, mag: SensorDict, grad: SensorDict, eeg: SensorDict
) -> Tuple[SensorDict, SensorDict, SensorDict]:
    """
    Extract data for each sensor type from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data
    mag : dict
        Magnetometer information
    grad : dict
        Gradiometer information
    eeg : dict
        EEG information

    Returns
    -------
    mag, grad, eeg : tuple of dict
        Updated sensor type dictionaries with extracted data
    """
    # Get the data per modality
    mag["data"] = raw.get_data(picks=mag["picks"])
    grad["data"] = raw.get_data(picks=grad["picks"])
    eeg["data"] = raw.get_data(picks=eeg["picks"])

    return mag, grad, eeg


def calculate_sensor_variance(
    mag: SensorDict, grad: SensorDict, eeg: SensorDict, raw: Raw, run: str
) -> Tuple[List[str], SensorDict, SensorDict, SensorDict]:
    """
    Calculate variance statistics for each sensor type and identify deviant channels.

    Uses the rejection threshold defined in config to mark channels as deviant
    if their variance deviates significantly from the median.

    Parameters
    ----------
    mag : dict
        Magnetometer data and metadata
    grad : dict
        Gradiometer data and metadata
    eeg : dict
        EEG data and metadata
    raw : mne.io.Raw
        Raw MEG/EEG data
    run : str
        Run identifier

    Returns
    -------
    variance_deviant : list
        Combined list of all deviant channels
    mag : dict
        Updated magnetometer information
    grad : dict
        Updated gradiometer information
    eeg : dict
        Updated EEG information
    """
    print(40 * "**")
    print(
        f"Calculating variance-deviant sensors. Threshold: {c.rejection_threshold}"
    )
    # Get the channel-variance per modality
    mag["var"], grad["var"], eeg["var"] = [
        np.var(d, axis=1) for d in [mag["data"], grad["data"], eeg["data"]]
    ]
    # Get the std of variance per modality
    mag["var_std"], grad["var_std"], eeg["var_std"] = map(
        np.std, [mag["var"], grad["var"], eeg["var"]]
    )
    # Get the median of variance per modality
    mag["var_median"], grad["var_median"], eeg["var_median"] = map(
        np.median, [mag["var"], grad["var"], eeg["var"]]
    )

    # Get the variance deviant sensors
    variance_deviant, mag, grad, eeg = f.find_variance_deviant_sensors(
        c.rejection_threshold, mag, grad, eeg, run
    )

    mag_percentage = round(
        (len(variance_deviant["mag"]) / len(mag["labels"])) * 1e2, 2
    )
    grad_percentage = round(
        (len(variance_deviant["grad"]) / len(grad["labels"])) * 1e2, 2
    )
    eeg_percentage = round(
        (len(variance_deviant["eeg"]) / len(eeg["labels"])) * 1e2, 2
    )

    print("\n")
    print(f" Bad magnetometers: {mag_percentage}%")
    print(f" Bad gradiometers: {grad_percentage}%")
    print(f" Bad eeg: {eeg_percentage}%")
    print("\n")
    print(40 * "**")

    variance_deviant = (
        variance_deviant["mag"]
        + variance_deviant["grad"]
        + variance_deviant["eeg"]
    )

    return variance_deviant, mag, grad, eeg


def combine_deviant_sensors(
    var_deviant: DeviantDict, grad_deviant: DeviantDict
) -> List[str]:
    """
    Combine deviant sensors from variance and gradient analyses.

    Parameters
    ----------
    var_deviant : dict
        Deviant sensors from variance analysis
    grad_deviant : dict
        Deviant sensors from gradient analysis

    Returns
    -------
    deviant : list
        Combined list of all deviant sensors
    """
    mag = list(set(var_deviant["mag"] + grad_deviant["mag"]))
    grad = list(set(var_deviant["grad"] + grad_deviant["grad"]))
    eeg = list(set(var_deviant["eeg"] + grad_deviant["eeg"]))

    deviant = mag + grad + eeg

    return deviant


# ============================================================================
# Main Processing Function
# ============================================================================


def detect_bad_sensors(subject: str) -> None:
    """
    Main function to detect bad channels for a single subject across all runs.

    Processes each run sequentially:
    1. Loads raw data
    2. Applies preprocessing steps
    3. Detects deviant channels
    4. Saves results to disk

    Parameters
    ----------
    subject : str
        Subject identifier
    """

    # ========================================================================
    # INITIALIZE CONTAINERS
    # ========================================================================

    # Make path constructor
    p = c.FetchPaths(c.root, c.project_name)
    # Path2raw
    meg_subject_dir = c.join(p.to_data(), subject, "Raw")
    # Get the available runs per subject
    runs = f.fetch_runs(c.path, subject)
    # Initialize dict to hold the bad channels
    bads = defaultdict()
    bads[subject] = defaultdict()
    # Read the info file of the first run to get the label names
    raw = mne.io.read_raw_fif(
        c.join(meg_subject_dir, runs[0] + "_raw.fif"),
        allow_maxshield=c.allow_maxshield,
        preload=False,
        verbose="error",
    )
    # Return sensor information based on the info
    mag, grad, eeg = f.fetch_sensor_information(raw, subject, runs)
    # Initialize drop report
    bad_log = f.initialize_rejection_report(runs, subject)

    # ========================================================================
    # DETECT DEVIANT SENSORS PER RUN
    # ========================================================================

    for run in tqdm(runs):
        # load raw.fif per run
        raw = load_raw_per_run(subject, run, meg_subject_dir)
        if "ICM" in subject:
            raw.drop_channels(["MISC001", "MISC002"])
            raw.set_channel_types(
                {"BIO001": "eog", "BIO002": "eog", "BIO003": "ecg"}
            )
        elif "ICM" not in subject:
            raw.set_channel_types(
                {"EOG061": "eog", "EOG062": "eog", "EEG064": "misc"}
            )

        # detrend data at the run level
        raw = detrend_data(raw)
        # calculate the rejection threshold for the SSP epochs
        # reject=calculate_ssp_threshold(raw)
        # notch filter the line-noise & the Harmonics
        raw = notch_filter(raw)
        # bandpass the raw
        raw = bandpass_filter(raw)
        # apply SSP on the raw to remove eye-blinks and heart-beats
        raw = apply_ssp_on_raw(raw, subject, run)
        # apply SSP a second time to a cleaner version of the data
        raw = apply_ssp_on_raw(raw, subject, run)
        # extract data per sensor-type
        mag, grad, eeg = get_data_per_sensor_type(raw, mag, grad, eeg)
        # get the variance deviant
        deviant, mag, grad, eeg = calculate_sensor_variance(
            mag, grad, eeg, raw, run
        )

        # Collect all the bads per run
        bads[subject][run] = deviant

        # Write 'bads' as pickle file per run
        dir_out = c.join(p.to_data(), subject, "Bad_Channels")
        if not c.exists(dir_out):
            c.make(dir_out)

        fname_out = c.join(dir_out, f"bads_{subject}_{run}.p")
        with open(fname_out, "wb") as fp:
            pickle.dump(
                bads[subject][run], fp, protocol=pickle.HIGHEST_PROTOCOL
            )

    # Write pickle file with the rejected labels for all runs
    pickle_fname_out = c.join(dir_out, f"bads_{subject}_all_runs.p")
    with open(pickle_fname_out, "wb") as fp:
        pickle.dump(bads, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # Save reports per subject
    f.plot_reports(
        c.rejection_threshold, mag, grad, eeg, subject, p.to_figures()
    )
    # Write the rejected log as a csv for all runs
    # csv_out=c.join(p.to_output(),subject,'Bad_Channels')
    # if not c.exists(csv_out):c.make(csv_out)
    # csv_fname_out=c.join(csv_out,'bads'+'_'+subject+'_'+
    #                    str(c.rejection_threshold)+'_all_runs.csv')
    # bad_log.to_csv(csv_fname_out)


# ============================================================================
# Script Execution
# ============================================================================

if __name__ == "__main__":
    parallel, run_func, _ = parallel_func(detect_bad_sensors, n_jobs=c.N_JOBS)
    parallel(run_func(subject) for subject in c.subjects_list)
