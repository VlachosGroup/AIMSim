"""Reads a configuration file from the command line input and calls molSim.

Raises:
    IOError: If the tasks field is empty in the input file, an IOError will be raised.
"""
from argparse import ArgumentParser

import yaml

from molSim.tasks import TaskManager

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    tasks = configs.pop('tasks', None)
    if tasks is None:
        raise IOError('"tasks" field not set in config file')

    task_manager = TaskManager(tasks=tasks)
    task_manager(molecule_set_configs=configs)