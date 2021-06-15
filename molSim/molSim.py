from argparse import ArgumentParser

import yaml

from molSim import task_manager


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    tasks = configs.pop('tasks', None)
    if tasks is None:
        raise IOError('<< tasks >> field not set in config file')
    task_manager.launch_tasks(molecule_database_configs=configs,
                              tasks=tasks)
