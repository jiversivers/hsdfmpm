import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from tests.__test_utils__ import patch_path_validators, add_patch_hsdfm_data


class TestAutofluorescenceImage(unittest.TestCase):
    def setUp(self):

    def test_set_metadata_path(self, mock_read_excel, mock_get_pvstate_values, mock_read_xml, mock_glob):

    def test_normalize_to_fluorescein(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
