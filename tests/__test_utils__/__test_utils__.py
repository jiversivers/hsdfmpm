from itertools import product
from unittest.mock import patch
import numpy as np
from photon_canon.lut import LUT
from photon_canon.contrib.bio import model_from_hemoglobin

rng = np.random.default_rng(42)
lut = LUT(dimensions=['mu_s', 'mu_a'], scale=50000, simulation_id=88, extrapolate=True)
wavelengths = np.arange(500, 730, 10)

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
    self.md_vals = {'ExpTime': list(3 * rng.random(len(wavelengths))),
                    'Wavelength': list(wavelengths)}
    self.sel_wl_idx = [0]
    self.bio_params = np.array([5 * rng.random((10, 10)), 2 * rng.random((10, 10)), rng.random((10, 10)), rng.random((10, 10))])
    self.hs_vals = np.zeros((len(wavelengths), 10, 10))
    for y, x in product(range(10), range(10)):
        self.hs_vals[:, y, x] = model_from_hemoglobin(lut, wavelengths, *self.bio_params[:, y, x])
    self.scalar = 2

    # Mock normalization arrays
    self.std_arr = rng.random(self.hs_vals.shape, dtype=np.float64)
    self.bg_arr = rng.random(self.hs_vals.shape, dtype=np.float64)

def add_patch_mpm_data(self):
    """Helper for testing to add patch mpm data to the test object."""
    pass