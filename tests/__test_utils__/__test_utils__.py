from unittest.mock import patch
import numpy as np


def patch_path_validators(self):
    """Helper for testing to patch pathlib.Path methods used in path validators and searches."""
    # Patches for file-paths (one-time patch for the test)
    is_dir_patch = patch('pathlib.Path.is_dir', return_value=True)
    is_file_patch = patch('pathlib.Path.is_file', return_value=True)
    glob_patch = patch('pathlib.Path.glob', side_effect=lambda x:[f'/path/to/metadata_{x.strip('*')}'])

    self.mock_is_dir = is_dir_patch.start()
    self.mock_is_file = is_file_patch.start()
    self.mock_glob = glob_patch.start()

    self.addCleanup(is_dir_patch.stop)
    self.addCleanup(is_file_patch.stop)
    self.addCleanup(glob_patch.stop)


def add_patch_hsdfm_data(self):
    """Helper for testing to add patch hsdfm data to the test object."""
    self.md_vals = {'ExpTime': [2, 4, 6],
                    'Wavelength': [500, 510, 520]}
    self.sel_wl_idx = [0]
    self.hs_vals = np.array([[[1, 2], [3, 4]],
                             [[5, 6], [7, 8]],
                             [[9, 0], [1, 2]]], dtype=np.float64)
    self.scalar = 2

def add_patch_mpm_data(self):
    """Helper for testing to add patch mpm data to the test object."""
    pass