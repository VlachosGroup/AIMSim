""" Test the ToolTip package (and all UI packages) """
import unittest


class TestToolTip(unittest.TestCase):
    """
    Test the ToolTip package (and all UI packages) by importing them

    """

    def test_import(self):
        """Attempt to import the UI app to ensure imports are correct
        """
        try:
            from interfaces.UI.AIMSim_ui_main import AIMSimUiApp
        except Exception:
            self.fail('Unable to import AIMSim UI')


if __name__ == "__main__":
    unittest.main()
