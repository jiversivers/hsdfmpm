import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from hsdfmpm.hsdfm import MergedHyperspectralImage
from hsdfmpm.hsdfm.hsdfm import HyperspectralImage

class TestHyperspectralImage(unittest.TestCase):
    def setUp(self):
        # Patches for file-paths (one-time patch for the test)
        is_dir_patch = patch('pathlib.Path.is_dir', return_value=True)
        is_file_patch = patch('pathlib.Path.is_file', return_value=True)
        glob_patch = patch('pathlib.Path.glob', return_value=['/path/to/metadata.json'])

        self.mock_is_dir = is_dir_patch.start()
        self.mock_is_file = is_file_patch.start()
        self.mock_glob = glob_patch.start()

        self.addCleanup(is_dir_patch.stop)
        self.addCleanup(is_file_patch.stop)
        self.addCleanup(glob_patch.stop)

        # Mocking normalization arrays
        self.std_arr = np.array([[2, 1], [1, 2]])
        self.bg_arr = np.array([[0.5, 1.5], [1.5, 0.5]])

        # Patch for file-dependencies (metadata)
        self.md_vals = {'ExpTime': [2, 4]}
        self.hs_vals = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.scalar = 2

        # Create mock standards (in context to skip validation)
        with patch('hsdfmpm.hsdfm.hsdfm.read_metadata_json', return_value=self.md_vals):
            with patch.object(HyperspectralImage, 'load_data', return_value=None):
                self.mock_std = HyperspectralImage.model_construct(image_path='dummy/path', _active=self.std_arr)
                self.mock_bg = HyperspectralImage.model_construct(image_path='dummy/path', _active=self.bg_arr)

                with patch('hsdfmpm.hsdfm.hsdfm.read_hyperstack', return_value=self.hs_vals):
                    self.hsi = HyperspectralImage(image_path="dummy_path")
                    self.hsi_with_scalar = HyperspectralImage(image_path="dummy_path", scalar=self.scalar)
                    self.hsi_with_array_scalar = HyperspectralImage(image_path="dummy_path", scalar=self.hs_vals/self.scalar)
                    self.hsi_with_standard = HyperspectralImage.model_construct(
                        image_path="dummy_path",
                        standard=self.mock_std,
                        background=self.mock_bg,
                        _active=self.hs_vals,
                        metadata=self.md_vals
                    )


    def test_load_data(self):
        # Without scalar
        self.assertEqual(self.hsi.metadata['ExpTime'], self.md_vals['ExpTime'])
        self.assertTrue(np.all(self.hsi._hyperstack == self.hs_vals))
        self.assertTrue(np.all(self.hsi._active == self.hsi._hyperstack))

        # With scalar scalar
        self.assertEqual(self.hsi_with_scalar.metadata['ExpTime'], self.md_vals['ExpTime'])
        self.assertTrue(np.all(self.hsi_with_scalar._hyperstack == 0.5 * self.hs_vals))
        self.assertTrue(np.all(self.hsi_with_scalar._active == self.hsi_with_scalar._hyperstack))

        # With array scalar
        self.assertEqual(self.hsi_with_array_scalar.metadata['ExpTime'], self.md_vals['ExpTime'])
        self.assertTrue(np.all(self.hsi_with_array_scalar._hyperstack == self.scalar))
        self.assertTrue(np.all(self.hsi_with_array_scalar._active == self.hsi_with_array_scalar._hyperstack))

    def test_normalize_integration_time(self):
        self.hsi.normalize_integration_time()
        self.assertTrue(
            np.all(self.hsi == self.hs_vals / np.array(self.md_vals['ExpTime'])[..., np.newaxis, np.newaxis]))

    def test_normalize_to_standard(self):
        self.hsi_with_standard.normalize_to_standard()
        self.assertTrue(np.all(self.hsi_with_standard == (self.hs_vals - self.bg_arr) / (self.std_arr - self.bg_arr)))

    def test_normalize(self):
        self.hsi_with_standard.normalize_integration_time = MagicMock(side_effect=self.hsi_with_standard.normalize_integration_time)
        self.hsi_with_standard.normalize_to_standard = MagicMock(side_effect=self.hsi_with_standard.normalize_to_standard)
        self.hsi_with_standard.normalize()
        self.hsi_with_standard.normalize_integration_time.assert_called_once()
        self.hsi_with_standard.normalize_to_standard.assert_called_once()


if __name__ == '__main__':
    unittest.main()
