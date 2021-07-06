from molSim.task_manager import launch_tasks

import yaml
import os
import tkinter as tk
import tkinter.ttk as ttk


class MolsimUiApp:
    def __init__(self, master=None):
        # build ui
        self.window = tk.Tk() if master is None else tk.Toplevel(master)
        self.databaseFile = tk.StringVar(self.window)
        self.targetMolecule = tk.StringVar(self.window)
        self.similarityMeasure = tk.StringVar(self.window)
        self.molecularDescriptor = tk.StringVar(self.window)
        self.titleLabel = ttk.Label(self.window)
        self.titleLabel.configure(
            font='TkDefaultFont', text='molecular Similarity')
        self.titleLabel.place(anchor='center', relx='0.5',
                              rely='0.05', x='0', y='0')
        self.mainframe = ttk.Frame(self.window)
        self.verboseCheckbutton = ttk.Checkbutton(self.mainframe)
        self.verboseCheckbutton.configure(
            compound='top', cursor='arrow', offvalue='False', onvalue='True')
        self.verboseCheckbutton.configure(state='normal', text='Verbose')
        self.verboseCheckbutton.place(
            anchor='center', relx='0.1', rely='0.95', x='0', y='0')
        self.databaseFileEntry = ttk.Entry(
            self.mainframe, textvariable=self.databaseFile)
        _text_ = '''smiles_responses.txt'''
        self.databaseFileEntry.delete('0', 'end')
        self.databaseFileEntry.insert('0', _text_)
        self.databaseFileEntry.place(
            anchor='center', relx='0.5', rely='0.03', x='0', y='0')
        self.databaseFileLabel = ttk.Label(self.mainframe)
        self.databaseFileLabel.configure(text='Database File:')
        self.databaseFileLabel.place(
            anchor='center', relx='.24', rely='0.03', x='0', y='0')
        self.targetMoleculeEntry = ttk.Entry(
            self.mainframe, textvariable=self.targetMolecule)
        _text_ = '''FC(F)(F)C(F)(F)C(F)C(F)C(F)(F)F'''
        self.targetMoleculeEntry.delete('0', 'end')
        self.targetMoleculeEntry.insert('0', _text_)
        self.targetMoleculeEntry.place(
            anchor='center', relx='0.5', rely='0.27', x='0', y='0')
        self.targetMoleculeLabel = ttk.Label(self.mainframe)
        self.targetMoleculeLabel.configure(text='Target Molecule:')
        self.targetMoleculeLabel.place(
            anchor='center', relx='0.22', rely='0.27', x='0', y='0')
        self.similarityPDFCheckbutton = ttk.Checkbutton(self.mainframe)
        self.similarityPDFCheckbutton.configure(text='Similarity PDF')
        self.similarityPDFCheckbutton.place(
            anchor='center', relx='0.5', rely='0.1', x='0', y='0')
        self.similarityHeatmapCheckbutton = ttk.Checkbutton(self.mainframe)
        self.similarityHeatmapCheckbutton.configure(text='Similarity Heatmap')
        self.similarityHeatmapCheckbutton.place(
            anchor='center', relx='0.5', rely='0.15', x='0', y='0')
        self.propertySimilarityCheckbutton = ttk.Checkbutton(self.mainframe)
        self.propertySimilarityCheckbutton.configure(
            text='Property Similarity Plot')
        self.propertySimilarityCheckbutton.place(
            anchor='center', relx='0.5', rely='0.2', x='0', y='0')
        self.similarityPlotCheckbutton = ttk.Checkbutton(self.mainframe)
        self.similarityPlotCheckbutton.configure(text='Similarity Plot')
        self.similarityPlotCheckbutton.place(
            anchor='center', relx='0.5', rely='0.35', x='0', y='0')
        self.similarityMeasureCombobox = ttk.Combobox(
            self.mainframe, textvariable=self.similarityMeasure, state="readonly")
        self.similarityMeasureCombobox.configure(
            takefocus=False, values=['tanimoto'])
        self.similarityMeasureCombobox.current(0)
        self.similarityMeasureCombobox.place(
            anchor='center', relx='0.5', rely='0.45', x='0', y='0')
        self.similarityMeasureLabel = ttk.Label(self.mainframe)
        self.similarityMeasureLabel.configure(text='Similarity Measure:')
        self.similarityMeasureLabel.place(
            anchor='center', relx='0.19', rely='0.45', x='0', y='0')
        self.molecularDescriptorLabel = ttk.Label(self.mainframe)
        self.molecularDescriptorLabel.configure(text='Molecular Descriptor:')
        self.molecularDescriptorLabel.place(
            anchor='center', relx='0.17', rely='0.5', x='0', y='0')
        self.molecularDescriptorCombobox = ttk.Combobox(
            self.mainframe, textvariable=self.molecularDescriptor, state="readonly")
        self.molecularDescriptorCombobox.configure(
            cursor='arrow', justify='left', takefocus=False, values=['topological_fingerprint', 'morgan_fingerprint'])
        self.molecularDescriptorCombobox.place(
            anchor='center', relx='0.5', rely='0.5', x='0', y='0')
        self.molecularDescriptorCombobox.current(0)
        self.runButton = ttk.Button(self.mainframe)
        self.runButton.configure(text='Run')
        self.runButton.place(anchor='center', relx='0.5',
                             rely='0.75', x='0', y='0')
        self.runButton.configure(command=self.runCallback)
        self.mainframe.configure(height='400', width='400')
        self.mainframe.place(anchor='nw', relheight='0.9',
                             rely='0.1', x='0', y='0')
        self.window.configure(cursor='arrow', height='400',
                              relief='flat', takefocus=False)
        self.window.configure(width='400')

        # Main widget
        self.mainwindow = self.window

    def runCallback(self):
        '''
        to write the yaml configuration file, put all the use settings into a giant nested dictionary.

        then, call task_launcher on the config file.
        '''
        tasks_dict = {}

        inner_dict = {}
        if('selected' in self.similarityPDFCheckbutton.state()):
            inner_dict['plot_settings'] = {'plot_color': 'green', 'plot_title': 'Entire Dataset'}
        if('selected' in self.similarityHeatmapCheckbutton.state()):
            inner_dict['pairwise_heatmap_settings'] = {'annotate': False, 'cmap': 'viridis'}
        if(len(inner_dict)>0):
            tasks_dict['visualize_dataset'] = inner_dict
        if('selected' in self.similarityPlotCheckbutton.state()):
            tasks_dict['compare_target_molecule'] = {'target_molecule_smiles': self.targetMolecule.get(), 'plot_settings': {'plot_color': 'orange', 'plot_title': 'Compared to Target Molecule'}, 'identify_closest_furthest': {'out_file_path': 'molSim-ui_output.txt'}}
        if('selected' in self.propertySimilarityCheckbutton.state()):
            tasks_dict['show_property_variation_w_similarity'] = {'property_file': self.databaseFile.get(),
                      'most_dissimilar': True, 'similarity_plot_settings': {'plot_color': 'red'}}
        
        verboseChecked = 'selected' in self.verboseCheckbutton.state()

        _, file_extension = os.path.splitext(self.databaseFile.get())
        if(file_extension == '.txt'):
            molecule_database_source_type = 'text'
        elif(file_extension == ''):
            molecule_database_source_type = 'folder'
        elif(file_extension == '.xlsx'):
            molecule_database_source_type = 'excel'
        else:
            molecule_database_source_type = file_extension.replace('.','')

        yamlOut = {'is_verbose': verboseChecked, 'molecule_database': self.databaseFile.get(), 'molecule_database_source_type': 'text', 'similarity_measure': self.similarityMeasure.get(),
                   'molecular_descriptor': self.molecularDescriptor.get(), 'tasks': tasks_dict}

        with open('molSim-ui-config.yaml', 'w') as outfile:
            yaml.dump(yamlOut, outfile, default_flow_style=False)

        configs = yaml.load(open('molSim-ui-config.yaml', "r"), Loader=yaml.FullLoader)

        tasks = configs.pop('tasks', None)
        if tasks is None:
            raise IOError('<< tasks >> field not set in config file')

        launch_tasks(molecule_database_configs=configs, tasks=tasks)

    def run(self):
        self.mainwindow.mainloop()


if __name__ == '__main__':
    app = MolsimUiApp()
    app.run()
