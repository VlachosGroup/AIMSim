"""User Interface and associated methods to access the key functionalities of
AIMSim without having to use the command line.

Raises:
    IOError: When pressing 'open config' it is possible that no suitable
    program will be known to the OS to open .yaml files. Also if no tasks
    are selected, an IOError will be raised.

Author:
    Jackson Burns
"""
from aimsim.tasks.task_manager import TaskManager
from aimsim.ops.descriptor import Descriptor
from aimsim.ops.similarity_measures import SimilarityMeasure

import yaml
import os
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk
import webbrowser
import pkg_resources


import customtkinter as ctk
from tktooltip import ToolTip

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


class AIMSimUiApp(ctk.CTk):
    """User interface to access key functionalities of AIMSim."""
    WIDTH = 600
    HEIGHT = 400

    def __init__(self):
        """Constructor for AIMSim UI.
        """
        super().__init__()
        # build ui
        self.title("AIMSim")
        self.minsize(AIMSimUiApp.WIDTH, AIMSimUiApp.HEIGHT)
        self.geometry(f"{AIMSimUiApp.WIDTH}x{AIMSimUiApp.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8), weight=1)
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # add the logo
        resource_path = pkg_resources.resource_filename(
            __name__,
            "AIMSim-GUI-corner-logo.png",
        )
        self.wm_iconphoto(False, tk.PhotoImage(file=resource_path))

        # setup attributes to hold files, variables, etc.
        self.databaseFile = tk.StringVar(master=self)
        self.targetMolecule = tk.StringVar(master=self)
        self.similarityMeasure = tk.StringVar(master=self)
        self.molecularDescriptor = tk.StringVar(master=self)

        # row 0
        # title
        self.titleLabel = ctk.CTkLabel(
            master=self,
            text_font=("Consolas", "26"),
            text="AI Molecular Similarity",
        )
        self.titleLabel.grid(
            row=0,
            column=0,
            columnspan=5,
            padx=0,
            pady=(0, 0),
            sticky="",
        )

        # row 1
        # label for database entry line
        self.databaseFileLabel = ctk.CTkLabel(
            master=self,
            text="Database File:",
        )
        self.databaseFileLabel.grid(
            row=1,
            column=0,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="e",
        )
        # text entry field for molecule database
        self.databaseFileEntry = ctk.CTkEntry(
            master=self, textvariable=self.databaseFile
        )
        _text_ = """smiles_responses.txt"""
        self.databaseFileEntry.delete("0", "end")
        self.databaseFileEntry.insert("0", _text_)
        self.databaseFileEntry.grid(
            row=1,
            column=1,
            columnspan=3,
            padx=0,
            pady=(0, 0),
            sticky="we",
        )
        # database file browser button
        self.browseButton = ctk.CTkButton(
            master=self,
            text="Browse...",
            command=self.browseCallback,
        )
        self.browseButton.grid(
            row=1,
            column=4,
            columnspan=1,
            padx=20,
            pady=(0, 0),
            sticky="",
        )

        # row 2
        # checkbox for database similarity plots
        self.similarityPlotsCheckbutton = ctk.CTkCheckBox(
            master=self,
            text="Database Similarity Plot",
        )
        self.similarityPlotsCheckbutton.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # checkbox for property similarity plot
        self.propertySimilarityCheckbutton = ctk.CTkCheckBox(
            master=self,
            text="Property Similarity Plot",
        )
        self.propertySimilarityCheckbutton.grid(
            row=2,
            column=2,
            columnspan=3,
            padx=0,
            pady=(0, 0),
            sticky="",
        )

        # row 3
        # label for target molecule
        self.targetMoleculeLabel = ctk.CTkLabel(
            master=self,
            text="Target Molecule:",
        )
        self.targetMoleculeLabel.grid(
            row=3,
            column=0,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # entry field for target molecule
        self.targetMoleculeEntry = ctk.CTkEntry(
            master=self, textvariable=self.targetMolecule
        )
        _text_ = """optional"""
        self.targetMoleculeEntry.delete("0", "end")
        self.targetMoleculeEntry.insert("0", _text_)
        self.targetMoleculeEntry.grid(
            row=3,
            column=1,
            columnspan=3,
            padx=0,
            pady=(0, 0),
            sticky="we",
        )
        # target molecule file browser button
        self.browseTargetButton = ctk.CTkButton(
            master=self,
            text="Browse...",
            command=self.browseCallback,
        )
        self.browseTargetButton.grid(
            row=3,
            column=4,
            columnspan=1,
            padx=20,
            pady=(0, 0),
            sticky="",
        )

        # row 4
        # label for similarity metric
        self.similarityMeasureLabel = ctk.CTkLabel(master=self)
        self.similarityMeasureLabel.configure(text="Similarity Measure:")
        self.similarityMeasureLabel.grid(
            row=4,
            column=0,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # dropdown for similarity measure
        self.similarityMeasureCombobox = ctk.CTkOptionMenu(
            master=self,
            variable=self.similarityMeasure,
            takefocus=False,
            values=SimilarityMeasure.get_uniq_metrics(),
            hover=False,
        )
        self.similarityMeasureCombobox.set(
            self.similarityMeasureCombobox.values[0]
        )
        self.similarityMeasureCombobox.grid(
            row=4,
            column=1,
            columnspan=3,
            padx=20,
            pady=(0, 0),
            sticky="we",
        )
        # checkbox to automatically determine the similarity measure
        self.useMeasureSearchCheckbox = ctk.CTkCheckBox(
            master=self,
            cursor="arrow",
            command=self.useMeasureSearchCallback,
            state="normal",
            text="AI Search",
        )
        self.useMeasureSearchCheckbox.grid(
            row=4,
            column=4,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )

        # row 5
        # label for descriptor dropdown
        self.molecularDescriptorLabel = ctk.CTkLabel(master=self)
        self.molecularDescriptorLabel.configure(text="Molecular Descriptor:")
        self.molecularDescriptorLabel.grid(
            row=5,
            column=0,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # dropdown for molecular descriptor
        self.molecularDescriptorCombobox = ctk.CTkOptionMenu(
            master=self,
            variable=self.molecularDescriptor,
            cursor="arrow",
            takefocus=False,
            values=Descriptor.get_supported_fprints(),
        )

        # define the callback for the descriptor
        def updateCompatibleMetricsListener(event):
            """Show only compatible metrics, given a descriptor."""
            self.similarityMeasureCombobox.configure(
                True,
                values=[
                    metric for metric in SimilarityMeasure.get_compatible_metrics().get(
                        self.molecularDescriptor.get(), "Error"
                    ) if (metric in SimilarityMeasure.get_uniq_metrics())
                ]
            )
            self.similarityMeasureCombobox.current(
                self.similarityMeasureCombobox.values[0]
            )
            return

        # bind this listener to the combobox
        self.molecularDescriptorCombobox.bind(
            "<<ComboboxSelected>>", updateCompatibleMetricsListener
        )
        self.molecularDescriptorCombobox.grid(
            row=5,
            column=1,
            columnspan=3,
            padx=20,
            pady=(0, 0),
            sticky="we",
        )
        self.molecularDescriptorCombobox.set(
            self.molecularDescriptorCombobox.values[0]
        )
        # checkbox to show all descriptors in AIMSim
        self.showAllDescriptorsButton = ctk.CTkCheckBox(
            master=self,
            cursor="arrow",
            command=self.showAllDescriptorsCallback,
            state="normal",
            text="Exp. Descriptors",
        )
        self.showAllDescriptorsButton.grid(
            row=5,
            column=4,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )

        # row 6
        # button to run AIMSim
        self.runButton = ctk.CTkButton(
            master=self,
            text="Run",
            command=self.runCallback,
        )
        self.runButton.grid(
            row=6,
            column=1,
            columnspan=1,
            padx=20,
            pady=(0, 0),
            sticky="",
        )
        # uses default editor to open underlying config file button
        self.openConfigButton = ctk.CTkButton(
            master=self,
            text="Open Config",
            command=self.openConfigCallback,
        )
        self.openConfigButton.grid(
            row=6,
            column=3,
            columnspan=1,
            padx=20,
            pady=(0, 0),
            sticky="",
        )

        # row 7
        # checkbox for verbosity
        self.verboseCheckbutton = ctk.CTkCheckBox(
            master=self,
            cursor="arrow",
            state=tk.NORMAL,
            text="Verbose",
        )
        self.verboseCheckbutton.grid(
            row=7,
            column=0,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # checkbox for outlier checking
        self.identifyOutliersCheckbutton = ctk.CTkCheckBox(
            master=self,
            cursor="arrow",
            state=tk.NORMAL,
            text="Outlier Check",
        )
        self.identifyOutliersCheckbutton.grid(
            row=7,
            column=1,
            columnspan=1,
            padx=0,
            pady=(0, 0),
            sticky="",
        )
        # multiprocessing checkbox
        self.multiprocessingCheckbutton = ctk.CTkCheckBox(
            master=self,
            cursor="arrow",
            state=tk.NORMAL,
            text="Multiple Processes",
        )
        self.multiprocessingCheckbutton.grid(
            row=7,
            column=2,
            columnspan=2,
            padx=0,
            pady=(0, 0),
            sticky="w",
        )

        # add ToolTips
        ToolTip(
            self.openConfigButton,
            "Open the config file\nfor the last run",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.runButton,
            "Write a config file\nand call AIMSim",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.targetMoleculeEntry,
            "SMILES string or Filepath for an 'external'\nmolecule for comparison to the others",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.browseButton,
            "Open a File Explorer to locate molecules\nin a supported data format",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.useMeasureSearchCheckbox,
            "Automatically determines best metric\nfor molecules with responses",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.showAllDescriptorsButton,
            "Show experimental descriptors from\nother libraries in the dropdown",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.verboseCheckbutton,
            "Check this for additional output\non the terminal or debugging",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.identifyOutliersCheckbutton,
            "Isolation Forest to identify outliers\nin sets of molecules with responses",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.multiprocessingCheckbutton,
            "Allow use of multiple processing\ncores (automatically configured)",
            follow=True,
            delay=2.0,
        )
        ToolTip(
            self.molecularDescriptorCombobox,
            "Tip: Use AIMSim from the command line\nto access descriptors from Mordred and Padel",
            follow=True,
            delay=2.0,
        )

    def browseCallback(self):
        """launch a file dialog and set the databse field"""
        out = filedialog.askopenfilename(
            initialdir=".",
            title="Select Molecule Database File",
            filetypes=[
                ('SMILES', '.smi .txt .SMILES'),
                ('Protein Data Bank', '.pdb'),
                ('Comma-Separated Values', '.csv .tsv'),
                ('Excel Workbook', '.xlsx'),
            ],
        )
        if out:
            self.databaseFile.set(out)
        return

    def showAllDescriptorsCallback(self):
        """update the descriptors dropdown to show descriptors."""

        if self.showAllDescriptorsButton.get():
            self.molecularDescriptorCombobox.configure(
                True,
                values=Descriptor.get_all_supported_descriptors()[:15],
            )
        else:
            self.molecularDescriptorCombobox.configure(
                True,
                values=Descriptor.get_supported_fprints(),
            )
            # switch off unsupported descriptor
            if self.molecularDescriptorCombobox.current_value not in Descriptor.get_supported_fprints():
                self.molecularDescriptorCombobox.set(
                    self.molecularDescriptorCombobox.values[0]
                )
        return

    def useMeasureSearchCallback(self):
        """measure search dropdown disable/enable"""
        if self.useMeasureSearchCheckbox.get():
            self.similarityMeasureCombobox.configure(
                state='disabled'
            )
        else:
            self.similarityMeasureCombobox.configure(
                state='normal'
            )
        return

    def openConfigCallback(self):
        """
        Open the config file being used by the UI to allow the user to edit it.

        """
        webbrowser.open("AIMSim-ui-config.yaml")

    def runCallback(self):
        """Retrieves user input, writes to a .yaml configuration file, and calls AIMSim on that input.

        Raises:
            IOError: When opening the automatically generated config file from the UI, there is a chance
            that the OS will not know which program to use, raising the error.
        """
        tasks_dict = {}

        inner_dict = {}
        if self.similarityPlotsCheckbutton.get():
            inner_dict["similarity_plot_settings"] = {
                "plot_color": "green",
                "plot_title": "Molecule Database Similarity Distribution",
            }
            inner_dict["heatmap_plot_settings"] = {
                "annotate": False,
                "plot_title": "Molecule Database Pairwise Similarities",
                "cmap": "viridis",
            }
        if len(inner_dict) > 0:
            tasks_dict["visualize_dataset"] = inner_dict
        if self.identifyOutliersCheckbutton.get():
            tasks_dict["identify_outliers"] = {"output": "terminal"}
        if self.targetMolecule.get() not in ("", "optional"):
            tasks_dict["compare_target_molecule"] = {
                "target_molecule_smiles": self.targetMolecule.get() if not os.path.exists(self.targetMolecule.get()) else None,
                "target_molecule_src": self.targetMolecule.get() if os.path.exists(self.targetMolecule.get()) else None,
                "similarity_plot_settings": {
                    "plot_color": "orange",
                    "plot_title": "Molecule Database Compared to Target Molecule",
                },
                "identify_closest_furthest": {"out_file_path": "AIMSim-ui_output.txt"},
            }
        if self.propertySimilarityCheckbutton.get():
            tasks_dict["see_property_variation_w_similarity"] = {
                "property_file": self.databaseFile.get(),
                "most_dissimilar": True,
                "similarity_plot_settings": {"plot_color": "red"},
            }

        verboseChecked = self.verboseCheckbutton.get()
        if self.multiprocessingCheckbutton.get():
            n_workers = 'auto'
        else:
            n_workers = 1

        _, file_extension = os.path.splitext(self.databaseFile.get())
        if file_extension.lower() in (".txt", ".smi", ".smiles"):
            molecule_database_source_type = "text"
        elif file_extension == "":
            molecule_database_source_type = "folder"
        elif file_extension == ".xlsx":
            molecule_database_source_type = "excel"
        else:
            molecule_database_source_type = file_extension.replace(".", "")

        yamlOut = {
            "is_verbose": verboseChecked,
            "n_workers": n_workers,
            "molecule_database": self.databaseFile.get(),
            "molecule_database_source_type": molecule_database_source_type,
            "similarity_measure": 'determine' if self.useMeasureSearchCheckbox.get() else self.similarityMeasure.get(),
            "fingerprint_type": self.molecularDescriptor.get(),
            "tasks": tasks_dict,
        }

        with open("AIMSim-ui-config.yaml", "w") as outfile:
            yaml.dump(yamlOut, outfile, default_flow_style=False)

        configs = yaml.load(
            open("AIMSim-ui-config.yaml", "r"),
            Loader=yaml.FullLoader,
        )

        tasks = configs.pop("tasks")
        if not tasks:
            messagebox.showerror(
                "Unexpected error!",
                "No tasks were selected.",
            )
            return

        try:
            task_manager = TaskManager(tasks=tasks)
            task_manager(molecule_set_configs=configs)
        except Exception as e:
            messagebox.showerror(
                "Unexpected error!",
                e,
            )
            return

    def run(self):
        """Start the UI."""
        self.mainloop()


def main():
    """Start the app."""
    app = AIMSimUiApp()
    app.run()


if __name__ == "__main__":
    app = AIMSimUiApp()
    app.run()
