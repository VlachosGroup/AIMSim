"""Abstract task class."""
from abc import ABC, abstractmethod
from copy import deepcopy


class Task(ABC):
    def __init__(self, configs):
        """Get configurations.
        Args:
            configs (dict): Parameters of the task
        """
        self.configs = deepcopy(configs)

    @abstractmethod
    def _extract_configs(self):
        pass

    @abstractmethod
    def __call__(self, molecule_set):
        pass

    @abstractmethod
    def __str__(self):
        pass
