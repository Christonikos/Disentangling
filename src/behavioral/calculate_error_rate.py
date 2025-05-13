"""
This module processes behavioral response data from subjects, calculating error rates
and accuracy metrics across different experimental conditions.

The script handles data from an experiment involving linguistic structures (PP/OBJ),
features (number/animacy), and various conditions (violation, congruency, interference).
"""

# =============================================================================
# MODULES
# =============================================================================
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as c


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================
def load_data(subject: str) -> pd.DataFrame:
    """
    Load and preprocess behavioral response data for a given subject.

    Args:
        subject (str): Subject identifier

    Returns:
        pd.DataFrame: Processed dataset with added feature columns
    """
    print(f"Loading data for subject: {subject}")
    path2data = c.join(c.root, c.project_name, "Data", subject, "Behavioral_Responses")
    files = [pd.read_csv(c.join(path2data, f), sep="\t") for f in c.see(path2data)]
    dataset = pd.concat(files)
    dataset.reset_index(drop=True, inplace=True)

    # Data cleanup and feature engineering
    del dataset["Var1"]
    dataset = dataset[dataset["RT"].notna()]
    
    # Extract condition components
    dataset["category"] = dataset.condition.apply(lambda x: x.split("_")[-1])
    dataset["number"] = dataset.condition.apply(lambda x: x.split("_")[-2])
    is_objrc = dataset.condition.apply(lambda x: x.split("_")[1]).apply(lambda y: "C" in y)
    
    # Add structure information
    dataset["structure"] = ""
    dataset.loc[is_objrc, "structure"] = "obj"
    dataset.loc[~is_objrc, "structure"] = "pp"
    dataset["condition"] = dataset.condition.apply(lambda x: x.split("_")[0])

    # Add congruency information
    dataset["congruency"] = ""
    congruent = ["GSLS", "GDLD"]
    incongruent = ["GSLD", "GDLS"]
    dataset.loc[dataset["condition"].isin(congruent), "congruency"] = "yes"
    dataset.loc[dataset["condition"].isin(incongruent), "congruency"] = "no"

    # Add response classification
    dataset["response"] = ""
    correct = ["TP", "TN"]
    false = ["FP", "FN"]
    dataset.loc[dataset["Behavioral"].isin(correct), "response"] = "correct"
    dataset.loc[dataset["Behavioral"].isin(false), "response"] = "false"

    # Add violation information
    dataset["violation"] = ""
    dataset.loc[dataset["violIndex"] == 1, "violation"] = "yes"
    dataset.loc[dataset["violIndex"] == 0, "violation"] = "no"

    # Rename and replace feature values
    dataset = dataset.rename(columns={"category": "feature"})
    dataset["feature"] = dataset["feature"].replace(
        {"syntactic": "number", "semantic": "animacy"}
    )

    # Add linear interference information
    dataset["linear_interference"] = ""
    
    # PP structure interference
    pp_conditions = {
        ("GSLS", "pp"): 0,
        ("GSLD", "pp"): 1,
        ("GDLS", "pp"): 0,
        ("GDLD", "pp"): 1,
    }
    
    # OBJ structure interference
    obj_conditions = {
        ("GSLS", "obj"): 0,
        ("GSLD", "obj"): 0,
        ("GDLS", "obj"): 1,
        ("GDLD", "obj"): 1,
    }

    # Apply interference values
    for (cond, struct), value in {**pp_conditions, **obj_conditions}.items():
        mask = (dataset.condition == cond) & (dataset.structure == struct)
        dataset.loc[mask, "linear_interference"] = value

    return dataset


def tranform_dataframe_for_ANOVA(count: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the dataframe for ANOVA analysis by mapping categorical variables to numerical values.

    Args:
        count (int): Subject counter
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Transformed dataframe ready for ANOVA
    """
    # Map categorical variables to numerical values
    df["violation"] = df["violation"].map({"yes": 1, "no": 0})
    df["congruency"] = df["congruency"].map({"yes": 1, "no": 0})

    # Select relevant features
    features = [
        "response",
        "violation",
        "linear_interference",
        "congruency",
        "number",
        "structure",
        "feature",
        "sentence",
    ]
    df = df[features]
    
    # Convert columns to integer type
    int_columns = ["violation", "linear_interference", "congruency"]
    df = df.astype({col: "int" for col in int_columns})

    return df


def get_the_error_per_unique_distribution(
    df: pd.DataFrame, count: int
) -> pd.DataFrame:
    """
    Calculate error rates for each unique combination of conditions.

    Args:
        df (pd.DataFrame): Input dataframe
        count (int): Subject counter

    Returns:
        pd.DataFrame: DataFrame containing error rates for each condition combination

    Raises:
        ValueError: If the number of unique conditions is not 24
    """
    # Get unique combinations of conditions
    unique = df.drop(["response", "sentence"], axis=1).drop_duplicates()
    if unique.shape[0] != 24:
        raise ValueError("Problem with #conditions")

    n_trials_total = df.shape[0]
    
    # Initialize collectors
    collectors = {
        "false_responses": [],
        "total_trials": [],
        "trials_per_construction_and_feature": [],
        "trials_per_unique_combination": [],
    }

    # Calculate error rates for each condition
    for condition in range(unique.shape[0]):
        selected_values = unique.iloc[condition].to_dict()
        
        # Calculate trials per construction
        q = f"structure == '{selected_values['structure']}' & feature == '{selected_values['feature']}'"
        ntrials_per_construction = df.query(q).shape[0]

        # Filter data for current condition
        isolated = df[
            (df.violation == selected_values["violation"])
            & (df.congruency == selected_values["congruency"])
            & (df.linear_interference == selected_values["linear_interference"])
            & (df.number == selected_values["number"])
            & (df.structure == selected_values["structure"])
            & (df.feature == selected_values["feature"])
        ]

        # Calculate metrics
        false = isolated[isolated.response == "false"].shape[0]
        
        # Update collectors
        collectors["false_responses"].append(false)
        collectors["total_trials"].append(n_trials_total)
        collectors["trials_per_construction_and_feature"].append(ntrials_per_construction)
        collectors["trials_per_unique_combination"].append(isolated.shape[0])

    # Add collected data to the unique combinations DataFrame
    for key, value in collectors.items():
        unique[key] = value

    unique.index = [count] * unique.shape[0]
    return unique


def cacl_accuracy(data: pd.DataFrame) -> float:
    """
    Calculate accuracy percentage from the response data.

    Args:
        data (pd.DataFrame): Input dataframe containing response data

    Returns:
        float: Accuracy percentage
    """
    correct = data[data.response == "correct"].shape[0]
    total = data.shape[0]
    accuracy = round((correct / total) * 1e2, 2)
    return accuracy


def plot_accuracy(accuracy: List[float]) -> None:
    """
    Plot accuracy distribution across subjects.

    Args:
        accuracy (List[float]): List of accuracy values for each subject
    """
    fig = plt.figure(dpi=100, facecolor="w", edgecolor="w")
    fig.set_size_inches(8, 4)
    plt.bar(np.arange(len(accuracy)), accuracy)
    plt.xticks([])
    plt.xlabel("#Subjects", fontsize=12, fontstyle="oblique")
    plt.ylabel("% Accuracy", fontsize=12, fontstyle="oblique")
    plt.axhline(
        np.mean(accuracy),
        color="r",
        linestyle="--",
        alpha=0.3,
        label=f"{np.mean(accuracy):.2f} ± {np.std(accuracy):.2f}",
    )
    plt.axhline(50, color="k", linestyle="--", alpha=0.3, label="chance")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    sns.despine()
    fig.savefig("accuracy.png", bbox_inches="tight", pad_inches=0.2, dpi=1200)
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
collector: List[pd.DataFrame] = []
accuracy: List[float] = []

# Process data for each subject
for count, subject in enumerate(c.subjects_list, 1):
    # Load and process data
    data = load_data(subject)
    accuracy.append(cacl_accuracy(data))
    
    # Transform data for analysis
    df = tranform_dataframe_for_ANOVA(count, data)
    new = get_the_error_per_unique_distribution(df, count)
    collector.append(new)

# Combine all processed data
data = pd.concat(collector)
data.index.name = "subject"

# Calculate error metrics
data["error_per_construction"] = (data.false_responses / data.trials_per_construction) * 1e2
data["error_total"] = (data.false_responses / data.total_trials) * 1e2
data["error_per_combination"] = (data.false_responses / data.trials_per_combination) * 1e2

# Save results
fname = "data.csv"
data.to_csv(fname)

