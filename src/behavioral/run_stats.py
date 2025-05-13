#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical analysis module for behavioral response data.

This module reads the CSV file created by 'calculate_error_rate.py' and performs
statistical analyses on it. It includes functions for calculating error rates,
standard errors, and creating interaction plots for different experimental conditions.

Author: Christos
"""

from typing import Dict, List, Tuple, Any
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from bioinfokit.analys import stat
from tabulate import tabulate
import config as c


def get_error_and_sem(data: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Calculate error rates, standard error of mean (SEM), and standard deviation for different conditions.

    Args:
        data (pd.DataFrame): Input DataFrame containing experimental data

    Returns:
        Tuple[Dict, Dict, Dict]: Tuple containing dictionaries for error rates, SEM, and STD
    """
    # Initialize nested dictionaries for storing results
    error: Dict[str, Dict[str, float]] = {"grammatical": {}, "violation": {}}
    sem: Dict[str, Dict[str, float]] = {"grammatical": {}, "violation": {}}
    std: Dict[str, Dict[str, float]] = {"grammatical": {}, "violation": {}}

    for v in ["grammatical", "violation"]:
        for con in ["congruent", "incongruent"]:
            # Set conditions for filtering
            violation_cond = 1 if v == "violation" else 0
            congruency_cond = 1 if con == "congruent" else 0
            
            # Filter data and calculate statistics
            filtered_data = data[
                (data.violation == violation_cond) & (data.congruency == congruency_cond)
            ].error_per_combination
            
            error[v][con] = filtered_data.mean()
            sem[v][con] = filtered_data.sem()
            std[v][con] = filtered_data.std()

    return error, sem, std


def plot_interaction_plots(data: pd.DataFrame) -> None:
    """
    Create interaction plots for different experimental conditions.

    Args:
        data (pd.DataFrame): Input DataFrame containing experimental data
    """
    # Set up the figure
    fig = plt.figure(dpi=100, facecolor="w", edgecolor="w")
    fig.set_size_inches(12, 4)
    
    labels = ["Congruent", "Incongruent"]
    lines = []
    marker_size = 12
    x_coords = (1, 2)

    # Plot for each construction type
    for idx, construction in enumerate(c.constructions):
        # Define structure and feature based on construction type
        structure, feature, title = get_plot_parameters(construction)
        
        # Filter data for current construction
        df = data.query(f"structure=='{structure}' & feature=='{feature}'")
        error, sem, _ = get_error_and_sem(df)

        # Create subplot
        plt.subplot(1, 3, idx + 1)
        plot_condition_data(
            x_coords, error, sem, marker_size, lines, idx == 0
        )
        
        # Customize plot appearance
        set_plot_aesthetics(x_coords, marker_size)

    # Finalize and save plot
    fig.tight_layout(pad=2)
    plt.savefig(fname="main_effects.png", bbox_inches="tight", dpi=1200)
    plt.show()


def get_plot_parameters(construction: str) -> Tuple[str, str, str]:
    """
    Get plot parameters based on construction type.

    Args:
        construction (str): Type of construction

    Returns:
        Tuple[str, str, str]: Structure, feature, and title for the plot
    """
    if construction == "pp_syntax":
        return "pp", "number", r"$\mathcal{PP-Number}$"
    elif construction == "objrc_syntax":
        return "obj", "number", r"$\mathcal{ObjRC-Number}$"
    elif construction == "pp_semantics":
        return "pp", "animacy", r"$\mathcal{PP-Animacy}$"
    else:
        raise ValueError(f"Unknown construction type: {construction}")


def plot_condition_data(
    x_coords: Tuple[int, int],
    error: Dict[str, Dict[str, float]],
    sem: Dict[str, Dict[str, float]],
    marker_size: int,
    lines: List[Any],
    show_ylabel: bool,
) -> None:
    """
    Plot data for a specific condition.

    Args:
        x_coords (Tuple[int, int]): X-coordinates for plotting
        error (Dict[str, Dict[str, float]]): Error rates
        sem (Dict[str, Dict[str, float]]): Standard error of mean
        marker_size (int): Size of markers
        lines (List[Any]): List to store line objects
        show_ylabel (bool): Whether to show y-axis label
    """
    x1, x2 = x_coords
    
    # Plot grammatical conditions
    plot_grammatical_conditions(x1, error, sem, marker_size, lines)
    
    # Plot violation conditions
    plot_violation_conditions(x2, error, sem, marker_size, lines)
    
    # Plot connecting lines
    plot_connecting_lines(x_coords, error)
    
    # Add ylabel if needed
    if show_ylabel:
        plt.ylabel(r"%Error", fontsize=1.2 * marker_size)


def plot_grammatical_conditions(
    x: int,
    error: Dict[str, Dict[str, float]],
    sem: Dict[str, Dict[str, float]],
    marker_size: int,
    lines: List[Any],
) -> None:
    """
    Plot grammatical conditions data points.

    Args:
        x (int): X-coordinate for plotting
        error (Dict[str, Dict[str, float]]): Error rates
        sem (Dict[str, Dict[str, float]]): Standard error of mean
        marker_size (int): Size of markers
        lines (List[Any]): List to store line objects
    """
    # Plot congruent condition
    l1 = plt.plot(
        x,
        error["grammatical"]["congruent"],
        c="gray",
        marker="o",
        markersize=marker_size,
        markeredgecolor="k",
        label="congruent",
    )
    lines.append(l1[0])
    
    plt.plot(
        x,
        error["grammatical"]["congruent"],
        c="g",
        marker="o",
        markersize=marker_size,
        markeredgecolor="k",
    )
    plt.errorbar(
        x,
        error["grammatical"]["congruent"],
        yerr=sem["grammatical"]["congruent"],
        c="g",
        uplims=False,
        lolims=False,
    )

    # Plot incongruent condition
    plt.plot(
        x,
        error["grammatical"]["incongruent"],
        c="g",
        marker="X",
        markersize=marker_size,
        markeredgecolor="k",
    )
    plt.errorbar(
        x,
        error["grammatical"]["incongruent"],
        yerr=sem["grammatical"]["incongruent"],
        c="g",
        uplims=False,
        lolims=False,
    )


def plot_violation_conditions(
    x: int,
    error: Dict[str, Dict[str, float]],
    sem: Dict[str, Dict[str, float]],
    marker_size: int,
    lines: List[Any],
) -> None:
    """
    Plot violation conditions data points.

    Args:
        x (int): X-coordinate for plotting
        error (Dict[str, Dict[str, float]]): Error rates
        sem (Dict[str, Dict[str, float]]): Standard error of mean
        marker_size (int): Size of markers
        lines (List[Any]): List to store line objects
    """
    # Plot congruent condition
    plt.plot(
        x,
        error["violation"]["congruent"],
        c="r",
        marker="o",
        markersize=marker_size,
        markeredgecolor="k",
    )
    plt.errorbar(
        x,
        error["violation"]["congruent"],
        yerr=sem["violation"]["congruent"],
        c="r",
        uplims=False,
        lolims=False,
    )

    # Plot incongruent condition (ghost data)
    l2 = plt.plot(
        x,
        error["violation"]["incongruent"],
        c="gray",
        marker="X",
        markersize=marker_size,
        markeredgecolor="k",
    )
    lines.append(l2[0])

    plt.plot(
        x,
        error["violation"]["incongruent"],
        c="r",
        marker="X",
        markersize=marker_size,
        markeredgecolor="k",
    )
    plt.errorbar(
        x,
        error["violation"]["incongruent"],
        yerr=sem["violation"]["incongruent"],
        c="r",
        uplims=False,
        lolims=False,
    )


def plot_connecting_lines(
    x_coords: Tuple[int, int], error: Dict[str, Dict[str, float]]
) -> None:
    """
    Plot lines connecting data points.

    Args:
        x_coords (Tuple[int, int]): X-coordinates for plotting
        error (Dict[str, Dict[str, float]]): Error rates
    """
    x1, x2 = x_coords
    plt.plot(
        [x1, x2],
        [error["grammatical"]["incongruent"], error["violation"]["incongruent"]],
        "k--",
        linewidth=1.2,
        zorder=1,
    )
    plt.plot(
        [x1, x2],
        [error["grammatical"]["congruent"], error["violation"]["congruent"]],
        "k-",
        linewidth=1.2,
        zorder=1,
    )


def set_plot_aesthetics(x_coords: Tuple[int, int], marker_size: int) -> None:
    """
    Set plot aesthetics including axes limits and labels.

    Args:
        x_coords (Tuple[int, int]): X-coordinates for plotting
        marker_size (int): Size of markers
    """
    x1, x2 = x_coords
    plt.xticks(
        [x1, x2], ["Grammatical", "Violation"], style="oblique", fontweight="bold"
    )
    plt.xlim([x1 - 0.2, x2 + 0.2])
    plt.ylim([0, 50])
    sns.despine(offset=10)


def get_summary_statistics(data: pd.DataFrame) -> None:
    """
    Calculate and print summary statistics for different conditions.

    Args:
        data (pd.DataFrame): Input DataFrame containing experimental data
    """
    for construction in c.constructions:
        structure, feature, _ = get_plot_parameters(construction)
        df = data.query(f"structure=='{structure}' & feature=='{feature}'")

        print(f"\n\n{'*' * 20}{construction}{'*' * 20}")
        
        # Print violation effect statistics
        print(f"{'+' * 20}VIOLATION EFFECT{'+' * 20}")
        print_condition_statistics(df, "violation")

        # Print congruency effect statistics
        print(f"{'+' * 20}CONGRUENCY EFFECT{'+' * 20}")
        print_condition_statistics(df, "congruency")


def print_condition_statistics(df: pd.DataFrame, condition: str) -> None:
    """
    Print statistics for a specific condition.

    Args:
        df (pd.DataFrame): Input DataFrame
        condition (str): Condition to analyze ('violation' or 'congruency')
    """
    for value in [0, 1]:
        condition_data = df[df[condition] == value].error_per_combination
        mean = round(condition_data.mean(), 2)
        sem = round(condition_data.sem(), 2)
        condition_name = "Violation" if value == 1 else "Grammatical"
        print(f"{condition_name}: {mean}±{sem}")


def run_anova_analysis(data: pd.DataFrame) -> None:
    """
    Run ANOVA analysis for different constructions and effects.

    Args:
        data (pd.DataFrame): Input DataFrame containing experimental data
    """
    construction_params = {
        "pp_number": ("pp", "number"),
        "obj_number": ("obj", "number"),
        "pp_animacy": ("pp", "animacy"),
    }

    for construction, (structure, feature) in construction_params.items():
        print(f"\n\n{'*' * 20}{construction}{'*' * 20}")
        df = data[(data.structure == structure) & (data.feature == feature)]

        for effect in ["congruency", "linear_interference"]:
            model = f"error_per_construction ~ C(violation)*C({effect})"
            res = stat()
            res.anova_stat(df=df, anova_model=model)
            
            # Create and print results table
            table = res.anova_summary.iloc[:-1]
            results = pd.DataFrame({"F": table.F, "p_val": table["PR(>F)"]})
            print(tabulate(results, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data.csv")

    # Create interaction plots
    plot_interaction_plots(data)

    # Run statistical analyses
    run_anova_analysis(data)

    # Calculate summary statistics
    get_summary_statistics(data)            