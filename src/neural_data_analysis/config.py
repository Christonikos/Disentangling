"""
MEG/EEG Analysis Configuration
============================

Configuration file for the lang_loc_glob project's MEG/EEG analysis pipeline.
Original source: https://github.com/brainthemind/CogBrainDyn_MEG_Pipeline

Configuration Sections
-------------------
1. Basic Setup
   - Project information
   - Path configurations
   - Processing flags

2. Experimental Design
   - Constructions
   - Effects
   - Event definitions
   - Timing parameters

3. Data Processing
   - Filtering parameters
   - Artifact rejection
   - Epoching settings
   - Channel configurations

4. Analysis Methods
   - SSP/ICA settings
   - Decoding parameters
   - Time-frequency analysis
   - Source reconstruction

5. Subject Information
   - Subject lists
   - Bad channel markings
   - Special cases

Author: Christos-Nikolaos Zacharopoulos
"""

# ==============================================================================
# Imports and Basic Setup
# ==============================================================================


# Standard library imports
import sys
import os
import os.path as op
import warnings
from collections import defaultdict

# Scientific computing
import numpy as np

# Local imports
sys.path.append("../")
from repos import func_repo as f
from repos.class_repo import FetchPaths

# Suppress warnings
warnings.filterwarnings("ignore")

# Path operation aliases
op = os.path
see = os.listdir
join = op.join
exists = op.exists
make = os.makedirs


# ==============================================================================
# Project Configuration
# ==============================================================================

# Processing flags
plot = False
autoreject = False
use_maxwell_filter = True
use_raw = False
use_ssp = True
use_ica = False

# Project identification
project_name = "LocalGlobal@ENGLISH"

# Experimental constructions
constructions = ["pp_syntax", "objrc_syntax", "pp_semantics"]
first_order_effects = [
    "main_effect_of_violation",
    "main_effect_of_congruency",
    "main_effect_of_transition",
]
second_order_effects = ["violation_congruency_interaction"]

# ==============================================================================
# Directory Configuration
# ==============================================================================

# Root directory setup
root = os.path.join(
    os.sep,
    "Volumes",
    "Transcend",
)

path = FetchPaths(root, project_name)

# Path definitions
project_path = path.to_project()
data_path = path.to_data()
figures_path = path.to_figures()
output_path = path.to_output()
supportive_dir = path.to_calibration_files()

N_JOBS = -1
# ==============================================================================
# Processing Parameters
# ==============================================================================

# Rejection thresholds
rejection_threshold = 6
reject = {"grad": 4000e-13, "mag": 4e-12, "eeg": 200e-6}

# Channel configuration
ch_types = ["meg", "eeg"]
rename_channels = None
set_channel_types = None

# Frequency filtering
l_freq = 0.4  # High-pass cutoff (Hz)
h_freq = 50  # Low-pass cutoff (Hz)
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"

# Epoching parameters
tmin = -0.5
tmax = 0.5 * 10  # 2*SOA=1000ms
decim = 4

# Timing parameters
timings = defaultdict(lambda: 0)
timings["soa"] = 500
timings["n_words"] = 7

# ==============================================================================
# Subject Configuration
# ==============================================================================

subjects_list = [
    "P01",
    "P02",
    "S01",
    "S02",
    "S03",
    "S04",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S12",
    "S14",
    "S15",
    "ICM01",
    "ICM02",
    "ICM03",
    "ICM04",
    "ICM06",
    "ICM07",
]

###############################################################################
# MAXFILTER PARAMETERS
# --------------------
allow_maxshield = True
mf_st_duration = None
mf_head_origin = "auto"


cal_files_path = os.path.join(
    supportive_dir,
)
mf_ctc_fname = os.path.join(cal_files_path, "ct_sparse.fif")
mf_cal_fname = os.path.join(cal_files_path, "sss_cal_171207.dat")

mf_reference_run = 0


resample_sfreq = None


decim = 4
# ==============================================================================
# Analysis Configuration
# ==============================================================================


def default_reject_comps():
    return dict(meg=[], eeg=[])


rejcomps_man = defaultdict(default_reject_comps)


# ==============================================================================
# File Naming
# ==============================================================================

base_fname = "{extension}.fif"
base_fname_trans = "{subject}_" + project_name + "_raw-trans.fif"

# ==============================================================================
# Validation Checks
# ==============================================================================

if (
    use_maxwell_filter
    and len(set(ch_types).intersection(("meg", "grad", "mag"))) == 0
):
    raise ValueError("Cannot use maxwell filter without MEG channels.")

if use_ssp and use_ica:
    raise ValueError("Cannot use both SSP and ICA.")
