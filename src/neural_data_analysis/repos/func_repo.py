"""
The local repository of the functions used for the analysis of the neural data:
"""


from collections import defaultdict
import os
import re
import mne
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# import sys
# sys.path.append('../../')
# import config as c


def fetch_bad_channel_labels(subject, config):
    """
    Return the bad-channels as those have been detected with the function
    00-detect_deviant_sensors.py as a dictionary.
    INPUTS:
        1. subject: Str: e.g 'S01'
        2. config: Python module
    """
    join = os.path.join
    path2bads = join(config.data_path, subject, "Bad_Channels")
    bads_fname = join(path2bads, "bads" + "_" + subject + "_all_runs.p")
    bads = pd.read_pickle(bads_fname)

    return bads


def fetch_runs(path, subject):
    """
    Fetch the runs of a subject. Resistant to filename similarities
    (e.g: run_01_raw, run_01_filt_raw)
    INPUTS:
        1. path - Object: created with 'fetch_paths' (@class_repo)
                  !Do not instatinate the object, just pass it as an argument.
        2. subject - Str: e.g 'S01'
    OUTPUTS:
        1. LIST: run names of the specified subject
    """
    join = os.path.join
    see = os.listdir
    split = re.sub
    if "BIDS" not in path.to_project():
        # list files
        files = see(join(path.to_data(), subject, "Raw"))
        # Keep only files that contain the word 'run'
        files = [i for i in files if "run" in i]
        # count only once
        runs = set([split("\D", "", file) for file in files])
        # add string
        add = lambda x: "run_" + x
        runs = list(map(add, runs))

    return sorted(runs)


def fetch_logs(path, subject, run="raw_log"):
    """
    Fetch the log-files of each subject.
    (e.g: run_01_raw, run_01_filt_raw)
    INPUTS:
        1. path - Object: created with 'fetch_paths' (@class_repo)
                  !Do not instatinate the object, just pass it as an argument.
        2. subject - Str: e.g 'S01'
    OUTPUTS:
        1. LIST: log-file names of the specified subject
    """
    join = os.path.join
    see = os.listdir

    if run == "raw_log":
        return see(join(path.to_data(), subject, "Log"))[0]
    else:
        matching = [
            idx
            for idx, s in enumerate(see(join(path.to_data(), subject, "Log")))
            if run in s
        ]
        return see(join(path.to_data(), subject, "Log"))[matching[0]]


def parse_event_id(config, eoi):
    """
    Keep only entries of interest from the event ID dictionary.
        INPUTS:
            1. config: Extract the config.event_id
            2. events-of-interest: list of strings, e.g: ["fixation", "last_word_onset"]
    """
    event_id = defaultdict(lambda: 0)
    for event in eoi:
        for k in config.event_id:
            if event in k:
                event_id[k] = config.event_id[k]

    return event_id


def load_epochs(subject, config, eoi, args):
    """
    Load the epochs of a given subject and event of interest.
    """
    op = os.path

    eoi = args.events_of_interest[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~ DEFINE PATHS ~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # This is the "base path", meaning the path that allows for loading
    # of the epochs without any advanced preprocessing applied
    # (e.g: Autoreject, ICA, PCA)

    path = op.join(config.data_path, subject, "Epochs_version_02")

    # find the file that contains the event of interest
    fname = [file for file in os.listdir(path) if (eoi in file) and (".fif" in file)][0]
    file = op.join(path, fname)

    print(40 * "--")
    print("Input: ", file)
    epochs = mne.read_epochs(file, preload=True, proj=True)
    print(40 * "--")
    return epochs


def fetch_channel_category(channels, info):
    """
    Separate channels to MAG, GRAD, EEG
    INPUTS:
        1. channels = epochs.ch_names
        2. info=epochs.info
    OUTPUTS:
        1. mag, grad, eeg: Channel Indices
        2. sig_mag, sig_grad, sig_eeg: Empty lists to be sig_magsig_mag
        populated with significant channel indices.
    """
    fetch = mne.channel_type
    ## ISOLATE CHANNELS PER CHANNEL-CATEGORY (MAG, GRAD, EEG)
    grad, mag, eeg, sig_grad, sig_mag, sig_eeg = ([] for i in range(0, 6))

    for channel, _ in enumerate(channels):
        chan_type = fetch(info, channel)
        if chan_type == "grad":
            grad.append(channel)
        elif chan_type == "mag":
            mag.append(channel)
        elif chan_type == "eeg":
            eeg.append(channel)

    return grad, mag, eeg, sig_grad, sig_mag, sig_eeg


def parse_epochs_per_effect_of_interest(epochs, case):
    """
    Given the effect of interest, parse the epochs and return
    a deviant and standard epochs-object.
    """
    ## PP - SYNTAX #########
    if case == "PP_SYNTAX_main_effect":
        gd = epochs["GDLS/PP/synt", "GDLD/PP/synt"]
        gs = epochs["GSLS/PP/synt", "GSLD/PP/synt"]
        title = "PP-SYNTAX-(GD VS GS)"
        plotting_dict = {"GD": gd.average(), "GS": gs.average()}
        plotting_linestyles = {"GD": "-", "GS": "-"}
        colors = {"GD": "red", "GS": "green"}

    ## PP - SEMANTICS #########
    elif case == "PP_SEMANTICS_main_effect":
        # Exclude trials that contain the words brake or honk
        gdld = epochs["GDLD/PP/sem"][
            (~epochs["GDLD/PP/sem"].metadata["Token"].str.contains("brake"))
            & (~epochs["GDLD/PP/sem"].metadata["Token"].str.contains("honk"))
        ]
        gdls = epochs["GDLS/PP/sem"][
            (~epochs["GDLS/PP/sem"].metadata["Token"].str.contains("brake"))
            & (~epochs["GDLS/PP/sem"].metadata["Token"].str.contains("honk"))
        ]
        gd = mne.concatenate_epochs([gdls, gdld])
        gs = epochs["GSLS/PP/sem", "GSLD/PP/sem"]
        title = "PP-SEMANTICS-(GD VS GS)"
        plotting_dict = {"GD": gd.average(), "GS": gs.average()}
        plotting_linestyles = {"GD": "-", "GS": "-"}
        colors = {"GD": "red", "GS": "green"}

    ## objRC - SYNTAX #########
    elif case == "OBJRC_SYNTAX_main_effect":
        gd = epochs["GDLD/objRC/synt", "GSLD/objRC/synt"]
        gs = epochs["GDLS/objRC/synt", "GSLS/objRC/synt"]
        title = "OBJRC-SYNTAX-(GD VS GS)"
        plotting_dict = {"GD": gd.average(), "GS": gs.average()}
        plotting_linestyles = {"GD": "-", "GS": "-"}
        colors = {"GD": "red", "GS": "green"}

    ## PP - GDLS VS GSLS #########
    elif case == "PP_SYNTAX_GDLS_VS_GSLS":
        # See the effect of long-range dependencies
        gd = epochs["GDLS/PP/synt"]
        gs = epochs["GSLS/PP/synt"]
        title = "PP-SYNTAX-(GDLS VS GSLS)"
        plotting_dict = {"GDLS": gd.average(), "GSLS": gs.average()}
        plotting_linestyles = {"GDLS": "-", "GSLS": "-"}
        colors = {"GDLS": "red", "GSLS": "green"}

    ## PP - GSLD VS GSLS #########
    elif case == "PP_SYNTAX_GSLD_VS_GSLS":
        # See the effect of transition-probability dependencies
        gd = epochs["GSLD/PP/synt"]
        gs = epochs["GSLS/PP/synt"]
        title = "PP-SYNTAX-(GSLD VS GSLS)"
        plotting_dict = {"GSLD": gd.average(), "GSLS": gs.average()}
        plotting_linestyles = {"GSLD": "-", "GSLS": "-"}
        colors = {"GSLD": "red", "GSLS": "green"}

    ## GSLD - PP VS OBJRC ######### - # CORRECT IN PP-INCORRECT IN OBJRC
    elif case == "GSLD_PP_VS_OBJRC":
        # See effect of structure
        gd = epochs["GSLD/objRC/synt"]  # wrong
        gs = epochs["GSLD/PP/synt"]  # correct
        title = "GSLD-(PP VS OBJRC)"
        plotting_dict = {"GSLD-objRC": gd.average(), "GSLD-PP": gs.average()}
        plotting_linestyles = {"GSLD-objRC": "-", "GSLD-PP": "-"}
        colors = {"GSLD-objRC": "red", "GSLD-PP": "green"}

    ## GDLS - PP VS OBJRC ######### - # INCORRECT IN PP-CORRECT IN OBJRC
    elif case == "GDLS_PP_VS_OBJRC":
        # See effect of structure
        gd = epochs["GDLS/PP/synt"]  # wrong
        gs = epochs["GDLS/objRC/synt"]  # correct
        title = "GDLS-(PP VS OBJRC)"
        plotting_dict = {"GDLS-objRC": gd.average(), "GDLS-PP": gs.average()}
        plotting_linestyles = {"GDLS-objRC": "-", "GDLS-PP": "-"}
        colors = {"GDLS-objRC": "red", "GDLS-PP": "green"}

    return gd, gs, title, plotting_dict, plotting_linestyles, colors


def f_cluster_permutation_test(
    threshold,
    permutations,
    condition1,
    condition2,
    channel,
    times,
    sig_mag,
    config,
    subject,
    sensor,
    case,
    join,
):
    """
    Perform Permutation F-test on sensor data with 1D cluster level.
    Plot only the sensors that contain a significant cluster.
    OUTPUTS:
        sig_mag: List that gets updated if the channel contains sifnigicant clusters
    """
    # get number of clusters and p-vals per cluster
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [condition1, condition2],
        n_permutations=permutations,
        threshold=threshold,
        tail=1,
        n_jobs=1,
    )

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            plt.close("all")
            plt.title("Channel : " + channel)
            h = plt.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
            plt.legend((h,), ("cluster p-value < 0.05",))
            sig_mag.append(sensor)
            hf = plt.plot(times, T_obs, "g")

            plt.axvline(x=0, color="r", linestyle="--")
            plt.axvline(x=0.5, color="r", linestyle="--")
            plt.ylim([0, 30])
            plt.xlabel("time (s)")
            plt.ylabel("f-values")
            fig_fn = join(
                config.path.to_figures(),
                "SIGNIFICANT_PERMUTED_MAG"
                + "_"
                + case
                + "_"
                + subject
                + "_"
                + channel
                + ".png",
            )
            plt.savefig(fig_fn)
            plt.show()

    return sig_mag


def fetch_sensor_information(raw, subject, runs):
    mag, grad, eeg = [defaultdict() for i in range(0, 3)]

    channels = raw.info["ch_names"]
    # Picks
    mag["picks"] = mne.pick_types(raw.info, "mag").tolist()
    grad["picks"] = mne.pick_types(raw.info, "grad").tolist()
    eeg["picks"] = mne.pick_types(raw.info, meg=False, eeg=True, misc=False).tolist()
    # Labels
    mag["labels"] = [channels[m] for m in mag["picks"]]
    grad["labels"] = [channels[m] for m in grad["picks"]]
    eeg["labels"] = [channels[m] for m in eeg["picks"]]
    ## REPORTS ##############
    # Magnetometers report
    mag["report"] = pd.DataFrame(0, index=mag["labels"], columns=runs)
    mag["report"].index.name = subject + "-Magnetometers"
    # Gradiometers report
    grad["report"] = pd.DataFrame(0, index=grad["labels"], columns=runs)
    grad["report"].index.name = subject + "-Gradiometers"
    # EEG report
    eeg["report"] = pd.DataFrame(0, index=eeg["labels"], columns=runs)
    eeg["report"].index.name = subject + "-EEG sensors"

    return mag, grad, eeg


def find_variance_deviant_sensors(thr, mag, grad, eeg, run):
    deviant = defaultdict()

    ### DETECTION #######
    # MAGNETOMETERS
    high_mags = np.where(mag["var"] > thr * mag["var_median"])[0].tolist()
    low_mags = np.where(mag["var"] < mag["var_median"] / thr)[0].tolist()
    deviant_mags = low_mags + high_mags
    deviant["mag"] = [mag["labels"][i] for i in deviant_mags]
    mag["report"][run].loc[deviant["mag"]] = 1

    # GRADIOMETERS
    high_grads = np.where(grad["var"] > thr * grad["var_median"])[0].tolist()
    low_grads = np.where(grad["var"] < grad["var_median"] / thr)[0].tolist()
    deviant_grads = low_grads + high_grads
    deviant["grad"] = [grad["labels"][i] for i in deviant_grads]
    grad["report"][run].loc[deviant["grad"]] = 1

    # EEG
    high_eeg = np.where(eeg["var"] > thr * eeg["var_median"])[0].tolist()
    low_eeg = np.where(eeg["var"] < eeg["var_median"] / thr)[0].tolist()
    deviant_eeg = low_eeg + high_eeg
    deviant["eeg"] = [eeg["labels"][i] for i in deviant_eeg]
    eeg["report"][run].loc[deviant["eeg"]] = 1

    return deviant, mag, grad, eeg


def initialize_rejection_report(runs, subject):
    # Initialize report to hold rejected stats and metadata of the rejection
    clnames = [
        "median-mag",
        "median-grad",
        "median-eeg",
        "%rejected mags",
        "%rejected grads",
        "%rejected eeg",
        "%rejected meg",
    ]
    bad_log = pd.DataFrame(index=clnames, columns=runs)
    bad_log.index.name = subject + "-Rejection metadata"

    return bad_log


def collect_rejection_metadata(thr, mag, grad, eeg, subject, bad_log, run, deviant):
    """
    Collect metadata for the rejection log per run
    """

    bad_log[run].loc["median-mag"] = mag["var_median"]
    bad_log[run].loc["median-grad"] = grad["var_median"]
    bad_log[run].loc["median-eeg"] = eeg["var_median"]

    bad_log[run].loc["%rejected mags"] = (
        len(deviant["mag"]) / len(mag["labels"])
    ) * 1e2
    bad_log[run].loc["%rejected grads"] = (
        len(deviant["grad"]) / len(grad["labels"])
    ) * 1e2
    bad_log[run].loc["%rejected eeg"] = (len(deviant["eeg"]) / len(eeg["labels"])) * 1e2
    bad_log[run].loc["%rejected meg"] = (
        (len(deviant["mag"]) + len(deviant["grad"]))
        / (len(mag["labels"]) + len(grad["labels"]))
        * 1e2
    )

    return bad_log


def plot_deviant_chanels_per_run(thr, mag, grad, eeg, subject, run, path2figs):
    # Change the backend to save fig full screen
    # -----------------------------------------
    import matplotlib

    gui = "TKAgg"
    matplotlib.use(gui, warn=False, force=True)
    from matplotlib import pyplot as plt

    # -----------------------------------------

    # Alliases
    join = os.path.join
    exists = os.path.exists
    make = os.makedirs

    fig_dir = join(
        path2figs,
        "Bad_Channels",
        subject,
    )
    if not exists(fig_dir):
        make(fig_dir)

    fig_name = join(fig_dir, subject + "_" + run + "_" + "thr_" + str(thr) + ".png")

    fnt_size = 8

    ## Plot the variance per sensor and modality along with the distribution of variance
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 8))
    fig.suptitle(f"{run}-{subject}", fontsize=fnt_size * 1.5)
    ## -- MAGNETOMETERS -- ##
    axes[0, 0].bar(
        np.arange(len(mag["var"])), mag["var"], color="royalblue", label="mag"
    )
    axes[0, 0].axhline(
        thr * mag["var_median"],
        linewidth=2,
        color="r",
    )
    axes[0, 0].axhline(
        mag["var_median"] / thr,
        linewidth=2,
        color="r",
    )
    axes[0, 0].axhline(
        mag["var_median"],
        linewidth=1,
        color="k",
    )
    axes[0, 0].set_xlabel("#Magnetometers", fontsize=fnt_size)
    axes[0, 0].set_ylabel(r"$Variance \quad [fT^2]$", fontsize=fnt_size)
    # axes[0 ,0].legend(loc="best",)

    # Plot the variance of the magnetomers per channel
    axes[0, 1].hist(
        mag["var"], bins=int(np.round(len(mag["var"]))), label="mag", color="royalblue"
    )
    axes[0, 1].axvline(
        thr * mag["var_median"], linewidth=2, color="r", label=f"{thr}*median"
    )
    axes[0, 1].axvline(
        mag["var_median"] / thr, linewidth=2, color="r", label=f"median/{thr}"
    )
    axes[0, 1].axvline(mag["var_median"], linewidth=1, color="k", label=f"median")
    axes[0, 1].set_xlabel("Variance/channel", fontsize=fnt_size)
    axes[0, 1].set_ylabel("Counts", fontsize=fnt_size)
    # axes[0 ,1].legend(loc="best",)

    ## -- GRADIOMETERS -- ##
    axes[1, 0].bar(
        np.arange(len(grad["var"])), grad["var"], label="grad", color="forestgreen"
    )
    axes[1, 0].axhline(
        thr * grad["var_median"], linewidth=2, color="r", label=f"{thr}*median"
    )
    axes[1, 0].axhline(
        grad["var_median"] / thr, linewidth=2, color="r", label=f"median/{thr}"
    )
    axes[1, 0].axhline(grad["var_median"], linewidth=1, color="k", label=f"median")
    axes[1, 0].set_xlabel("#Gradiometers", fontsize=fnt_size)
    axes[1, 0].set_ylabel(r"$Variance \quad [(\frac{fT}{cm})^2]$", fontsize=fnt_size)
    #    axes[1 ,0].legend(loc="best", fancybox=True)

    # Plot the variance of the magnetomers per channel
    axes[1, 1].hist(
        grad["var"],
        bins=int(np.round(len(grad["var"]))),
        label="grad",
        color="forestgreen",
    )
    axes[1, 1].axvline(
        thr * grad["var_median"], linewidth=2, color="r", label=f"{thr}*median"
    )
    axes[1, 1].axvline(
        grad["var_median"] / thr, linewidth=2, color="r", label=f"median/{thr}"
    )
    axes[1, 1].axvline(grad["var_median"], linewidth=1, color="k", label=f"median")
    axes[1, 1].set_xlabel("Variance/channel", fontsize=fnt_size)
    axes[1, 1].set_ylabel("Counts", fontsize=fnt_size)
    #    axes[1 ,1].legend(loc="best", fancybox=True)

    ## -- GRADIOMETERS -- ##
    axes[2, 0].bar(
        np.arange(len(eeg["var"])), eeg["var"], label="eeg", color="darkorange"
    )
    axes[2, 0].axhline(
        thr * eeg["var_median"], linewidth=2, color="r", label=f"{thr}*median"
    )
    axes[2, 0].axhline(
        eeg["var_median"] / thr, linewidth=2, color="r", label=f"median/{thr}"
    )
    axes[2, 0].axhline(eeg["var_median"], linewidth=1, color="k", label=f"median")
    axes[2, 0].set_xlabel("#EEG sensors", fontsize=fnt_size)
    axes[2, 0].set_ylabel(r"$Variance \quad [uV^2]$", fontsize=fnt_size)
    #    axes[2 ,0].legend(loc="best", fancybox=True)

    # Plot the variance of the magnetomers per channel
    axes[2, 1].hist(
        eeg["var"], bins=int(np.round(len(eeg["var"]))), label="eeg", color="darkorange"
    )
    axes[2, 1].axvline(
        thr * eeg["var_median"], linewidth=2, color="r", label=f"{thr}*median"
    )
    axes[2, 1].axvline(
        eeg["var_median"] / thr, linewidth=2, color="r", label=f"median/{thr}"
    )
    axes[2, 1].axvline(eeg["var_median"], linewidth=1, color="k", label=f"median")
    axes[2, 1].set_xlabel("Variance/channel", fontsize=fnt_size)
    axes[2, 1].set_ylabel("Counts", fontsize=fnt_size)
    #    axes[2 ,1].legend(loc="best", fancybox=True)

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    #    plt.show()
    plt.savefig(fig_name, bbox_inches="tight")


def plot_reports(thr, mag, grad, eeg, subject, path2figs):
    import seaborn as sns

    # Alliases
    join = os.path.join
    exists = os.path.exists
    make = os.makedirs

    fig_dir = join(
        path2figs,
        "Bad_Channels",
        subject,
    )
    if not exists(fig_dir):
        make(fig_dir)

    # File names
    mag_fig_name = join(fig_dir, "mag_" + subject + "_" + "thr_" + str(thr) + ".png")
    grad_fig_name = join(fig_dir, "grad_" + subject + "_" + "thr_" + str(thr) + ".png")
    eeg_fig_name = join(fig_dir, "eeg_" + subject + "_" + "thr_" + str(thr) + ".png")

    plt.clf()
    # Magnetometers
    sns_plot = sns.heatmap(mag["report"], cbar=False)
    plt.title("Magnetometers")
    plt.tight_layout()
    sns_plot.figure.savefig(mag_fig_name)

    plt.clf()
    # Gradiometers
    sns_plot = sns.heatmap(grad["report"], cbar=False)
    plt.title("Gradiometers")
    plt.tight_layout()
    sns_plot.figure.savefig(grad_fig_name)

    plt.clf()
    # EEG
    sns_plot = sns.heatmap(eeg["report"], cbar=False)
    plt.title("EEG sensors")
    plt.tight_layout()
    sns_plot.figure.savefig(eeg_fig_name)


def fix_digitization(current_raw, config, run):
    """
    Update the empty digitization value of the current raw-file with the
    one of the reference-run.
    """
    op = os.path
    join = os.path.join
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


def fetch_conditions(base_condition, grammatical_type, embedding, grammatical_number):
    """
    Return a string to parse the MNE epochs object.
    """

    # Parse the input accordingly to create the parsing object

    # --Base condition--#
    # This argument needs to be capitalized
    base_condition = base_condition.upper()
    # --Grammatical type--#
    # This argument can either be 'synt' or 'sem' and lower case
    grammatical_type = grammatical_type.lower()
    # --Embedding--#
    # This argument can either be 'PP' or 'objRC'
    if "pp" in embedding.lower():
        embedding = embedding.upper()
    elif "objrc" in grammatical_type.lower():
        embedding = "objRC"
    # --Grammatical number--#
    # The grammatical number can have three values:
    # 'sing', 'plur' or 'both'
    if "sing" in grammatical_number.lower():
        grammatical_number = "sing"
    elif "plur" in grammatical_number.lower():
        grammatical_number = "plur"
    elif "both" in grammatical_number.lower():
        grammatical_number = ""

    # Construct the parsing object:
    parsing_object = base_condition + "/" + grammatical_type + "/" + embedding
    if grammatical_number == "sing" or grammatical_number == "plur":
        parsing_object = parsing_object + "/" + grammatical_number

    return parsing_object


def plot_ems_grand_average(
    c,
    t,
    times,
    keys,
    colors,
    linestyles,
    gsls_1,
    gsld,
    gsls_2,
    gdls,
    gsls_3,
    gdld,
    gs,
    gd,
    N,
    factorial,
):
    """
    c: config,
    t: type of sensor
    times: epochs duration
    keys: labels of conditions
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from scipy.stats import sem

    join = os.path.join
    exists = os.path.exists
    make = os.makedirs

    path2type = join(
        c.path.to_figures(), "EMS_Spatial_Filter", factorial, "Grand_Average", t
    )
    # Check if the output directory exists
    if not exists(path2type):
        make(path2type)

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5, forward=True)
    # GSLS-GSLD
    ax = plt.subplot(411)
    plt.plot(
        times,
        np.mean(gsls_1, axis=0),
        label=keys[0],
        color=colors[0],
        linestyle=linestyles[0],
    )
    error = sem(gsls_1, axis=0)
    plt.fill_between(
        times,
        np.mean(gsls_1, axis=0) - error,
        np.mean(gsls_1, axis=0) + error,
        alpha=0.5,
        edgecolor="lightgreen",
        facecolor="palegreen",
        label="SEM GSLS ",
    )
    plt.plot(
        times,
        np.mean(gsld, axis=0),
        label=keys[1],
        color=colors[1],
        linestyle=linestyles[1],
    )
    error = sem(gsld, axis=0)
    plt.fill_between(
        times,
        np.mean(gsld, axis=0) - error,
        np.mean(gsld, axis=0) + error,
        alpha=0.5,
        edgecolor="mistyrose",
        facecolor="lightcoral",
        label="SEM GSLD",
    )
    plt.axvline(0, color="k", linestyle="-.")
    plt.axvline(0.5, color="gray", linestyle="-.")
    ax.axvspan(0.05, 0.15, alpha=0.15, color="darkorange")
    ax.axvspan(0.35, 0.45, alpha=0.15, color="darkorange")
    ax.axvspan(0.55, 0.65, alpha=0.15, color="darkorange")
    plt.xlabel("Time (ms)")
    plt.ylabel("a.u.")
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1
    )
    plt.ylim([-2.5, 2.5])

    # GSLS-GSLS
    ax = plt.subplot(412)
    plt.plot(
        times,
        np.mean(gsls_2, axis=0),
        label=keys[2],
        color=colors[0],
        linestyle=linestyles[0],
    )
    error = sem(gsls_2, axis=0)
    plt.fill_between(
        times,
        np.mean(gsls_2, axis=0) - error,
        np.mean(gsls_2, axis=0) + error,
        alpha=0.5,
        edgecolor="lightgreen",
        facecolor="palegreen",
        label="SEM GSLS",
    )
    plt.plot(
        times,
        np.mean(gdls, axis=0),
        label=keys[3],
        color=colors[1],
        linestyle=linestyles[1],
    )
    error = sem(gdls, axis=0)
    plt.fill_between(
        times,
        np.mean(gdls, axis=0) - error,
        np.mean(gdls, axis=0) + error,
        alpha=0.5,
        edgecolor="mistyrose",
        facecolor="lightcoral",
        label="SEM GDLS",
    )
    plt.axvline(0, color="k", linestyle="-.")
    plt.axvline(0.5, color="gray", linestyle="-.")
    ax.axvspan(0.05, 0.15, alpha=0.15, color="darkorange")
    ax.axvspan(0.35, 0.45, alpha=0.15, color="darkorange")
    ax.axvspan(0.55, 0.65, alpha=0.15, color="darkorange")
    plt.xlabel("Time (ms)")
    plt.ylabel("a.u.")
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1
    )
    plt.ylim([-2.5, 2.5])

    # GSLS-GDLD
    ax = plt.subplot(413)
    plt.plot(
        times,
        np.mean(gsls_3, axis=0),
        label=keys[4],
        color=colors[0],
        linestyle=linestyles[0],
    )
    error = sem(gsls_3, axis=0)
    plt.fill_between(
        times,
        np.mean(gsls_3, axis=0) - error,
        np.mean(gsls_3, axis=0) + error,
        alpha=0.5,
        edgecolor="lightgreen",
        facecolor="palegreen",
        label="SEM GSLS",
    )
    plt.plot(
        times,
        np.mean(gdld, axis=0),
        label=keys[5],
        color=colors[1],
        linestyle=linestyles[1],
    )
    error = sem(gdld, axis=0)
    plt.fill_between(
        times,
        np.mean(gdld, axis=0) - error,
        np.mean(gdld, axis=0) + error,
        alpha=0.5,
        edgecolor="mistyrose",
        facecolor="lightcoral",
        label="SEM GDLD",
    )
    plt.axvline(0, color="k", linestyle="-.")
    plt.axvline(0.5, color="gray", linestyle="-.")
    ax.axvspan(0.05, 0.15, alpha=0.15, color="darkorange")
    ax.axvspan(0.35, 0.45, alpha=0.15, color="darkorange")
    ax.axvspan(0.55, 0.65, alpha=0.15, color="darkorange")
    plt.xlabel("Time (ms)")
    plt.ylabel("a.u.")
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1
    )
    plt.ylim([-2.5, 2.5])

    # GSLS-GDLD
    ax = plt.subplot(414)
    plt.plot(
        times,
        np.mean(gs, axis=0),
        label=keys[6],
        color=colors[0],
        linestyle=linestyles[0],
    )
    error = sem(gs, axis=0)
    plt.fill_between(
        times,
        np.mean(gs, axis=0) - error,
        np.mean(gs, axis=0) + error,
        alpha=0.5,
        edgecolor="lightgreen",
        facecolor="palegreen",
        label="SEM STANDARD",
    )
    plt.plot(
        times,
        np.mean(gd, axis=0),
        label=keys[7],
        color=colors[1],
        linestyle=linestyles[1],
    )
    error = sem(gd, axis=0)
    plt.fill_between(
        times,
        np.mean(gd, axis=0) - error,
        np.mean(gd, axis=0) + error,
        alpha=0.5,
        edgecolor="mistyrose",
        facecolor="lightcoral",
        label="SEM DEVIANT",
    )
    plt.axvline(0, color="k", linestyle="-.")
    plt.axvline(0.5, color="gray", linestyle="-.")
    ax.axvspan(0.05, 0.15, alpha=0.15, color="darkorange")
    ax.axvspan(0.35, 0.45, alpha=0.15, color="darkorange")
    ax.axvspan(0.55, 0.65, alpha=0.15, color="darkorange")
    plt.xlabel("Time (ms)")
    plt.ylabel("a.u.")
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1
    )
    plt.ylim([-2.5, 2.5])

    plt.suptitle(
        f"All subjects (N:{N}) - Average EMS signal - " f"{t.upper()} - ({factorial})",
        y=1.05,
        fontsize=20,
        fontname="Oswald",
    )
    fname = join(path2type, "ems_pp_syntax_grand_average_" + t + ".pdf")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=1800)


def update_log_file(run_log, beh_log):
    import pandas as pd

    target_columns = [
        "pair_index",
        "n1",
        "pp",
        "n2",
        "v1",
        "v2_n3_adverb",
        "condition",
        "sentence",
        "violIndex",
        "pp_freq",
        "n1_freq",
        "n2_freq",
        "v1_freq",
        "v2_n3_adverb_freq",
        "pp_nl",
        "n1_nl",
        "n2_nl",
        "v1_nl",
        "v2_n3_adverb_nl",
        "violation_type",
        "subject_response",
        "RT",
        "Behavioral",
    ]
    # get the number of trials
    trials = [c + 1 for c in beh_log.index.values.tolist()]
    # initialize the run-log to receive the enriched metadata
    behs_list = pd.DataFrame()
    for trial in trials:
        curr_df = run_log[run_log["Trial"] == trial]
        for t in target_columns:
            curr_df[t] = beh_log.iloc[trial - 1][t]
        # Update the behavioral log file
        curr_df["subject_response"][
            (curr_df["Behavioral"] == "TP") | (curr_df["Behavioral"] == "TN")
        ] = "correct"

        curr_df["subject_response"][
            (curr_df["Behavioral"] == "FP") | (curr_df["Behavioral"] == "FN")
        ] = "false"

        behs_list = behs_list.append(curr_df, ignore_index=True)

    return behs_list


import socket


def get_ip():
    """
    Get the local IP.
    Used to set the matplotlib backed dynamically.
    When ssh-ing the backed should be set to 'Agg'

    Returns
    -------
    IP : String
        Returns the local IP of the workstation used

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP
