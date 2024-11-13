"""
Maxwell Filter Application for NeuroSpin Data
=========================================

This script applies Maxwell filtering to MEG data collected at NeuroSpin using MNE-Python's 
SSS implementation.

Processing Details
----------------
1. Applies Signal Space Separation (SSS) 
2. Performs movement compensation
3. Corrects head positions to a reference run
4. Handles NeuroSpin-specific digitization fixes

Key Features
-----------
- SSS/tSSS filtering for environmental noise reduction
- Movement compensation across runs
- Head position standardization
- Automatic handling of digitization issues
- Support for parallel processing

Configuration Parameters
----------------------
From config.py:
- mf_st_duration: Duration for tSSS (None for standard SSS)
- mf_reference_run: Run to use as reference for head position
- mf_head_origin: Origin for head coordinates
- use_maxwell_filter: Boolean to enable/disable processing
- plot: Boolean to enable/disable data visualization

Technical Details
---------------
- Uses NeuroSpin-specific calibration files from calibration_files/ns/
- Handles missing digitization data automatically
- Applies reference head position from first run
- Removes all SSP projectors before Maxwell filtering

Notes
-----
- Bad channels must be marked before Maxwell filtering
- tSSS with short duration (e.g., 10s) acts as a 0.1 Hz highpass filter
- Special handling for subjects with missing HPI digitization

Author: Christos-Nikolaos Zacharopoulos

"""


# Standard Library
import os.path as op

# MNE Specific
import mne
from mne.parallel import parallel_func

# Local
import config
from repos import func_repo as f



def fix_digitization(current_raw: mne.io.Raw, config: object, run: str) -> mne.io.Raw:
    """
    Fix missing or incorrect digitization data in raw files.

    This function updates empty digitization values by copying them from a reference
    subject (S06). This is needed when digitization information is missing or corrupted.

    Parameters
    ----------
    current_raw : mne.io.Raw
        The raw data object needing digitization fixes
    config : object
        Configuration object containing path information
    run : str
        Current run identifier

    Returns
    -------
    mne.io.Raw
        Raw object with updated digitization information

    Notes
    -----
    - Uses S06 as the reference subject for digitization data
    - Updates dev_head_t, dig, and chs fields
    - Critical for subjects with missing HPI digitization
    """

    join = op.join
    # fetch a reference raw to replace the digitization of this subject
    meg_subject_dir = op.join(config.data_path, "S06", "Raw")

    extension = run + "_filt_raw"
    raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))

    # Load the reference-subject info:
    raw = mne.io.read_raw_fif(raw_fname_in, preload=False, allow_maxshield=True)

    # Update the empty value of the current raw
    current_raw.info["dev_head_t"] = raw.info["dev_head_t"]
    current_raw.info["dig"] = raw.info["dig"]
    current_raw.info["chs"] = raw.info["chs"]

    return current_raw


def run_maxwell_filter(subject: str) -> None:
    """
    Apply Maxwell filtering to all runs of a subject's MEG data.

    This function performs the following steps:
    1. Loads NeuroSpin-specific calibration files
    2. Identifies reference run for head position
    3. Processes each run with Maxwell filtering
    4. Handles missing digitization data
    5. Saves the processed data

    Parameters
    ----------
    subject : str
        Subject identifier (must not be an ICM subject)

    Technical Details
    ---------------
    - Uses SSS or tSSS based on config.mf_st_duration
    - Applies movement compensation to reference run position
    - Automatically fixes missing digitization data
    - Removes all SSP projectors before processing
    - Saves output with '_sss_raw' suffix

    Notes
    -----
    - Requires NeuroSpin calibration files in calibration_files/ns directory
    - Handles special cases for subjects with missing HPI data
    - Can generate plots if config.plot is True
    - Skips ICM subjects (handled by separate script)
    """
    print("Processing subject: %s" % subject)

    # Load NeuroSpin-specific calibration files
    cal_files_path = op.join(config.project_path, "calibration_files", "ns")
    mf_ctc_fname = op.join(cal_files_path, "ct_sparse.fif")
    mf_cal_fname = op.join(cal_files_path, "sss_cal_171207.dat")

    # Set up paths and get reference run
    meg_subject_dir = op.join(config.data_path, subject, "Raw")
    reference_run = f.fetch_runs(config.path, subject)[0]

    # Get destination head position from reference run
    extension = reference_run + "_filt_raw"
    raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    info = mne.io.read_info(raw_fname_in)
    destination = info["dev_head_t"]

    # Process each run
    for run in f.fetch_runs(config.path, subject):
        extension = run + "_filt_raw"
        raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))

        extension = run + "_sss_raw"
        raw_fname_out = op.join(meg_subject_dir, config.base_fname.format(**locals()))

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        # INPUT: LOADS THE FILTERED .FIF FILES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw = mne.io.read_raw_fif(raw_fname_in, allow_maxshield=True)
        # remove projs
        raw = raw.del_proj("all")
        ################################################################

        raw.fix_mag_coil_types()

        if config.mf_st_duration:
            print("st_duration=%d" % (config.mf_st_duration,))

        # In some runs the raw.info["dev_head_t"] is not present.
        if not raw.info["dev_head_t"]:
            try:
                raw.info["dev_head_t"] = info["dev_head_t"]
                if not raw.info["dev_head_t"]:
                    try:
                        # Christos-manual hack for subj#7-which had not HPI digitization
                        raw = fix_digitization(raw, config, run)
                    except:
                        raise ValueError('Empty raw.info["dev_head_t"]!')
            finally:
                print(f"Subj-{subject}-Run-{run}")

        # In some runs the raw.info['dig'] is empty () - The digitization did not work
        if not raw.info["dig"]:
            # Christos-manual hack for subj#9-which had not HPI digitization
            raw = fix_digitization(raw, config, run)
            print(f"Using digitization parameters from S03 for: {subject}")

        raw_sss = mne.preprocessing.maxwell_filter(
            raw,
            calibration=mf_cal_fname,
            cross_talk=mf_ctc_fname,
            st_duration=config.mf_st_duration,
            origin=config.mf_head_origin,
            destination=destination,
            coord_frame="head",
        )

        # OUTPUT: SAVES FILTERED-SSS CORRECTED DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ################################################################
        raw_sss.save(raw_fname_out, overwrite=True)
        ################################################################

        if config.plot:
            # plot maxfiltered data
            raw_sss.plot(n_channels=50, butterfly=True, group_by="position")


if __name__ == "__main__":
    if config.use_maxwell_filter:
        parallel, run_func, _ = parallel_func(run_maxwell_filter, n_jobs=config.N_JOBS)
        parallel(
            run_func(subject)
            for subject in config.subjects_list
            if "ICM" not in subject
        )
