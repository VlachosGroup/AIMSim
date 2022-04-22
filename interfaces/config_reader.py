"""Reads a configuration file from the command line input and calls AIMSim.

Raises:
    IOError: If the tasks field is empty in the input file,
    an IOError will be raised.
"""
from argparse import ArgumentParser
import yaml
import random

from aimsim.tasks import TaskManager


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to config yaml file.")
    args = parser.parse_args()
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    tasks = configs.pop("tasks", None)
    if tasks is None:
        raise IOError('"tasks" field not set in config file')
    # if the global_random_seed has been specified, overwrite
    # other random seeds
    if configs.get('global_random_seed', None):
        if configs.get('global_random_seed') == 'random' or type(configs.get('global_random_seed')) is not int:
            configs['global_random_Seed'] = random.randint(0, 2**30)
        if 'cluster' in tasks:
            cluster_configs = tasks.get('cluster')
            if cluster_configs.get('cluster_plot_settings', None):
                cluster_plot_settings = cluster_configs.get(
                    'cluster_plot_settings')
                if cluster_plot_settings.get('embedding', None):
                    embedding_settings = cluster_plot_settings.get('embedding')
                    embedding_settings['random_state'] = configs['global_random_seed']
                else:
                    # need to make the embedding dict
                    cluster_plot_settings['embedding'] = {
                        'random_state': configs['global_random_seed']}
            else:
                # need to make all dictionaries
                cluster_configs['cluster_plot_settings'] = {
                    'embedding': {'random_state': configs['global_random_seed']}}
        if 'identify_outliers' in tasks:
            tasks['identify_outliers']['random_state'] = configs['global_random_seed']
    task_manager = TaskManager(tasks=tasks)
    task_manager(molecule_set_configs=configs)


if __name__ == "__main__":
    main()
