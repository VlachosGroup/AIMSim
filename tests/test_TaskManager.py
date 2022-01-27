"""Test the TaskManager class."""
import unittest

from molSim.exceptions import InvalidConfigurationError
from molSim.tasks.task_manager import TaskManager


class TestTaskManager(unittest.TestCase):

    test_smiles = [
        "CCCCCCC",
        "CCCC",
        "CCC",
        "CO",
        "CN",
        "C1=CC=CC=C1",
        "CC1=CC=CC=C1",
        "C(=O)(N)N",
    ]

    def smiles_seq_to_textfile(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a text file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses.. Defaults to None.

        Returns:
            str: Path to created file.
        """
        text_fpath = "temp_smiles_seq.txt"
        print(f"Creating text file {text_fpath}")
        with open(text_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return text_fpath

    def test_no_tasks_task_manager(self):
        """Instant stop test for TaskManager class.
        """
        with self.assertRaises(InvalidConfigurationError):
            task_man = TaskManager(
                {}
            )

    # def test_task_manager(self):
    #     """Complete run test for TaskManager class.
    #     """
    #     task_man = TaskManager(
    #         {}
    #     )


if __name__ == "__main__":
    unittest.main()
