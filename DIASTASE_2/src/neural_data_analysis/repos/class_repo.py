#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The local repository of the classes used for the analysis of the neural data.
"""

import os


class FetchPaths:
    """
    A class to manage and retrieve various project-related paths.

    Attributes:
        root (str): The root directory of the project.
        project_name (str): The name of the project.
    """

    def __init__(self, root: str, project_name: str) -> None:
        """
        Initializes the FetchPaths class with the root directory and project name.

        Parameters:
            root (str): The root directory of the project.
            project_name (str): The name of the project.
        """
        self.root = root
        self.project_name = project_name

    def to_project(self) -> str:
        """Returns the path to the project directory."""
        return os.path.join(self.root, self.project_name)

    def to_derivatives(self) -> str:
        """Returns the path to the derivatives directory."""
        return os.path.join(self.root, self.project_name, "Derivatives")

    def to_data(self) -> str:
        """Returns the path to the data directory."""
        return os.path.join(self.root, self.project_name, "Data")

    def to_figures(self) -> str:
        """Returns the path to the figures directory."""
        return os.path.join(self.root, self.project_name, "Figures")

    def to_output(self) -> str:
        """Returns the path to the output directory."""
        return os.path.join(self.root, self.project_name, "Output")

    def to_calibration_files(self) -> str:
        """Returns the path to the calibration files directory."""
        return os.path.join(self.root, self.project_name, "Calibration_files")

    def __str__(self) -> str:
        """Returns a string representation of the project."""
        return f"Project: {self.project_name}"


class FetchTriggers:
    """
    A class to map events with their corresponding trigger values.

    Attributes:
        fixation (int): Trigger value for fixation.
        fix2panel_on (int): Trigger value for fix2panel on.
        fix2panel_off (int): Trigger value for fix2panel off.
        fix2feedback_on (int): Trigger value for fix2feedback on.
        fix2feedback_off (int): Trigger value for fix2feedback off.
        first_word_onset (int): Trigger value for first word onset.
        first_word_offset (int): Trigger value for first word offset.
        last_word_onset (int): Trigger value for last word onset.
        last_word_offset (int): Trigger value for last word offset.
        word_onset (int): Trigger value for word onset.
        word_offset (int): Trigger value for word offset.
        panel_on (int): Trigger value for panel on.
        panel_off (int): Trigger value for panel off.
        key_press (int): Trigger value for key press.
    """

    fixation: int = 1
    fix2panel_on: int = 10
    fix2panel_off: int = 15
    fix2feedback_on: int = 100
    fix2feedback_off: int = 110
    first_word_onset: int = 40
    first_word_offset: int = 50
    last_word_onset: int = 60
    last_word_offset: int = 70
    word_onset: int = 80
    word_offset: int = 90
    panel_on: int = 30
    panel_off: int = 35
    key_press: int = 120
