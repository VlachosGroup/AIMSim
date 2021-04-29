from molSim.task_manager import launch_tasks

import yaml

yaml.load('config.yaml')

{'verbose': True, 'molecule_database': 'smiles_responses.txt', 'similarity_measure': 'tanimoto', 'molecular_descriptor': 'topological fingerprint', 'tasks': {'compare_target_molecule': {'target_molecule_smiles': 'FC(F)(F)C(F)(F)C(F)C(F)C(F)(F)F', 'plot_settings': {'plot_color': 'orange', 'plot_title': 'Compared to Target Molecule'}, 'identify_closest_furthest': {'out_file_path': 'output_jazz.txt'}}, 'visualize_dataset': {'plot_settings': {'plot_color': 'green', 'plot_title': 'Entire Dataset'}, 'pairwise_heatmap_settings': {'annotate': False, 'cmap': 'viridis'}}, 'show_property_variation_w_similarity': {'property_file': 'smiles_responses.txt', 'most_dissimilar': False, 'similarity_plot_settings': {'plot_color': 'red'}}}}