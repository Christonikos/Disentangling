"""
Basic MEG/EEG Data Preprocessing Pipeline
=======================================

This script performs essential preprocessing steps on MEG/EEG data using MNE-Python.

Processing Steps
--------------
1. Loads raw .fif files for each subject and run
2. Applies linear detrending to remove slow drifts
3. Performs notch filtering (50Hz and harmonics) to remove line noise
4. Applies bandpass filtering using parameters from config.py
5. Saves the filtered data to new files

Key Parameters (from config.py)
----------------------------
- l_freq: Lower frequency bound for bandpass filter
- h_freq: Upper frequency bound for bandpass filter
- l_trans_bandwidth: Lower transition bandwidth
- h_trans_bandwidth: Upper transition bandwidth
- plot: If True, generates raw data and PSD plots

Technical Details
---------------
- Uses linear-phase FIR filter with delay compensation
- Employs Hamming window for FIR filter design
- Handles both MEG and EEG channels
- Supports parallel processing across subjects

Output
------
Filtered data files are saved in the subject's MEG directory with '_filt_raw' suffix.

Author: Christos-Nikolaos Zacharopoulos

"""

# Standard library
import os.path as op
from warnings import warn
import sys

sys.path.append("..")

# MNE speific
import mne
from mne.parallel import parallel_func

# Misc
from scipy import signal

# Local
import config
from repos import func_repo as f


def run_filter(subject: str) -> None:
    """
    Process and filter raw MEG/EEG data for a single subject.

    This function performs the following steps:
    1. Loads raw data files
    2. Sets appropriate channel types (EOG, ECG, etc.)
    3. Applies linear detrending
    4. Performs notch filtering at 50Hz and harmonics
    5. Applies bandpass filtering
    6. Saves the processed data

    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'sub-001')

    Raises
    ------
    ValueError
        If no raw data files are found for the subject

    Notes
    -----
    - Bad channels are loaded from previous automatic rejection steps
    - For ICM subjects, specific channel type assignments are handled differently
    - All projection components are removed before filtering
    - If config.plot is True, generates visualization of the data and its spectrum
    """
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.data_path, subject, "Raw")


    n_raws = 0
    for run in f.fetch_runs(config.path, subject):
        # read bad channels using the automatic rejection step used in functions 00b & 00c
        if run:
            bads = f.fetch_bad_channel_labels(subject, config)[subject][run]
        else:
            bads = config.bads[subject]

        extension = run + "_raw"
        raw_fname_in = op.join(
            meg_subject_dir, config.base_fname.format(**locals())
        )

        extension = run + "_filt_raw"
        raw_fname_out = op.join(
            meg_subject_dir, config.base_fname.format(**locals())
        )

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        if not op.exists(raw_fname_in):
            warn("Run %s not found for subject %s " % (raw_fname_in, subject))
            continue

        # INPUT: LOADS THE RAW .FIF FILES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw = mne.io.read_raw_fif(
            raw_fname_in,
            allow_maxshield=config.allow_maxshield,
            preload=True,
            verbose="error",
        )

        # =====================================================================
        # Set channel types
        # =====================================================================

        if "ICM" in subject:
            raw.drop_channels(["MISC001", "MISC002"])
            raw.set_channel_types(
                {"BIO001": "eog", "BIO002": "eog", "BIO003": "ecg"}
            )
        elif "ICM" not in subject:
            raw.set_channel_types(
                {"EOG061": "eog", "EOG062": "eog", "EEG064": "misc"}
            )
        # remove projs
        raw = raw.del_proj("all")
        ################################################################

        # add bad channels
        raw.info["bads"] = bads
        print("added bads: ", raw.info["bads"])

        # =====================================================================
        # LINEAR DETRENDING
        # =====================================================================
        print(40 * "**")
        print("Detrending the data at the run level")
        print(40 * "**")
        data = raw._data
        data = signal.detrend(data)
        raw._data = data

        # =====================================================================
        # Noth filter line-noise & harmonics
        # =====================================================================
        print(40 * "**")
        notch = 50
        print(f"applying notch filter at {notch} Hz and 3 harmonics")
        notch_freqs = [notch, notch * 2, notch * 3, notch * 4]
        raw = raw.notch_filter(notch_freqs)
        print(40 * "**")

        # =====================================================================
        # Bandpass the data
        # =====================================================================
        # Band-pass the data channels (MEG and EEG)
        print(40 * "**")
        print(
            "Filtering data between %s and %s (Hz)"
            % (config.l_freq, config.h_freq)
        )
        raw.filter(
            config.l_freq,
            config.h_freq,
            l_trans_bandwidth=config.l_trans_bandwidth,
            h_trans_bandwidth=config.h_trans_bandwidth,
            filter_length="auto",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
        )
        print(40 * "**")

        # OUTPUT: SAVES FILTERED
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw.save(raw_fname_out, overwrite=True)
        fs = raw.info["sfreq"]
        print(f"Sampling rate: {fs}")
        ################################################################
        n_raws += 1

        if config.plot:
            # plot raw data
            raw.plot(
                n_channels=50,
                butterfly=True,
                group_by="position",
                bad_color="r",
            )

            # plot power spectral densitiy
            raw.plot_psd(
                area_mode="range",
                tmin=10.0,
                tmax=100.0,
                fmin=0.0,
                fmax=50.0,
                average=True,
            )

    if n_raws == 0:
        raise ValueError("No input raw data found.")


# %%
parallel, run_func, _ = parallel_func(run_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
