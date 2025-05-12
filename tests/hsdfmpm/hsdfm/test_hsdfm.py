import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
from unittest.mock import patch, MagicMock

from hsdfmpm.hsdfm.hsdfm import HyperspectralImage, MergedHyperspectralImage
from tests.__test_utils__ import patch_path_validators, add_patch_hsdfm_data


class TestHyperspectralImage(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_hsdfm_data(self)

        # Create mock standards (in context to skip validation)
        with patch('hsdfmpm.hsdfm.hsdfm.read_metadata_json', return_value=self.md_vals):
            with patch('hsdfmpm.hsdfm.hsdfm.read_hyperstack', return_value=self.std_arr):
                self.mock_std = MergedHyperspectralImage(image_paths=['dummy/path2', 'dummy/path2'])
                self.mock_std = HyperspectralImage(image_path='dummy/path2')

            with patch('hsdfmpm.hsdfm.hsdfm.read_hyperstack', return_value=self.bg_arr):
                self.mock_bg = HyperspectralImage(image_path='dummy/path')

            with patch('hsdfmpm.hsdfm.hsdfm.read_hyperstack', return_value=self.hs_vals):
                self.hsi = HyperspectralImage(image_path="dummy_path")
                self.hsi_subset = HyperspectralImage(image_path="dummy_path", wavelengths=[self.md_vals['Wavelength'][wl] for wl in self.sel_wl_idx])
                self.hsi_with_scalar = HyperspectralImage(image_path="dummy_path", scalar=self.scalar)
                self.hsi_with_array_scalar = HyperspectralImage(image_path="dummy_path", scalar=self.hs_vals/self.scalar)
                self.hsi_with_standard = HyperspectralImage(image_path="dummy_path", standard=self.mock_std, background=self.mock_bg)


    def test_load_data(self):
        # Without scalar
        self.assertEqual(self.hsi.metadata['ExpTime'], self.md_vals['ExpTime'])
        npt.assert_array_equal(self.hsi._hyperstack, self.hs_vals)
        npt.assert_array_equal(self.hsi._active, self.hsi._hyperstack)

        # With wavelength mask
        self.assertEqual(self.hsi_subset.metadata['ExpTime'], [self.md_vals['ExpTime'][wl] for wl in self.sel_wl_idx])
        self.assertEqual(self.hsi_subset.metadata['Wavelength'], [self.md_vals['Wavelength'][wl] for wl in self.sel_wl_idx])
        self.assertTrue(self.hsi_subset._active.shape == (1, *self.hs_vals.shape[1:]))
        npt.assert_array_equal(self.hsi_subset._active, self.hs_vals[self.sel_wl_idx])

        # With scalar scalar
        self.assertEqual(self.hsi_with_scalar.metadata['ExpTime'], self.md_vals['ExpTime'])
        npt.assert_array_equal(self.hsi_with_scalar._hyperstack, self.hs_vals / self.scalar)
        npt.assert_array_equal(self.hsi_with_scalar._active, self.hsi_with_scalar._hyperstack)

        # With array scalar
        self.assertEqual(self.hsi_with_array_scalar.metadata['ExpTime'], self.md_vals['ExpTime'])
        npt.assert_array_equal(self.hsi_with_array_scalar._hyperstack, self.hs_vals / (self.hs_vals / self.scalar))
        npt.assert_array_equal(self.hsi_with_array_scalar._active, self.hsi_with_array_scalar._hyperstack)

    def test_normalize_integration_time(self):
        self.hsi.normalize_integration_time()
        self.assertTrue(
            np.all(self.hsi == self.hs_vals / np.array(self.md_vals['ExpTime'])[..., np.newaxis, np.newaxis]))

    def test_normalize_to_standard(self):
        self.hsi_with_standard.normalize_to_standard()
        npt.assert_array_equal(self.hsi_with_standard, (self.hs_vals - self.bg_arr) / (self.std_arr - self.bg_arr))

    def test_normalize(self):
        self.hsi_with_standard.normalize_integration_time = MagicMock(side_effect=self.hsi_with_standard.normalize_integration_time)
        self.hsi_with_standard.normalize_to_standard = MagicMock(side_effect=self.hsi_with_standard.normalize_to_standard)
        self.hsi_with_standard.normalize()
        self.hsi_with_standard.normalize_integration_time.assert_called_once()
        self.hsi_with_standard.normalize_to_standard.assert_called_once()
        expected = (
                ((self.hs_vals / np.array(self.md_vals['ExpTime'])[..., np.newaxis, np.newaxis]) - self.bg_arr)
                / (self.std_arr - self.bg_arr)
        )
        npt.assert_array_equal(self.hsi_with_standard, expected)

    def test_persistence_when_nested(self):
        # Straight up
        npt.assert_array_equal(self.hsi_with_standard.standard, self.mock_std)
        npt.assert_array_equal(self.hsi_with_standard.background, self.mock_bg)

        # After normalization of nested object
        self.mock_std.normalize()
        npt.assert_array_equal(self.hsi_with_standard.standard, self.mock_std)

    def test_normalize_with_missing(self):
        self.assertRaises(ValueError, self.hsi.normalize_to_standard)

        self.hsi.normalize_integration_time = MagicMock(side_effect=self.hsi.normalize_integration_time)
        self.hsi.normalize_to_standard = MagicMock(side_effect=self.hsi.normalize_to_standard)
        self.hsi.normalize()
        self.hsi.normalize_integration_time.assert_called_once()
        self.hsi.normalize_to_standard.assert_not_called()

    def test_get(self):
        wavelength = self.md_vals['Wavelength'][1]
        npt.assert_array_equal(self.hsi.get(wavelength), self.hs_vals[1])

    # Covers baseclass imagedata tests below
    def test_shape(self):
        npt.assert_array_equal(self.hsi.shape, self.hs_vals.shape)

    def test_bin(self):
        self.hsi.bin(bin_factor=2)
        self.assertEqual(self.hsi.shape, (23, 5, 5))
        npt.assert_array_almost_equal(np.mean(self.hsi, axis=(1,2)), self.hs_vals.mean(axis=(1,2)))

    def test_subset_by_metadata(self):
        self.hsi.subset_by_metadata('Wavelength', np.arange(500, 520, 10))
        self.assertEqual(len(self.hsi), 2)
        npt.assert_array_equal(self.hsi.image, self.hs_vals[np.isin(self.md_vals['Wavelength'], np.arange(500, 520, 10))])

    def test_subset_and_superset(self):
        self.hsi.subset([1, 2])
        self.assertEqual(len(self.hsi), 2)
        npt.assert_array_equal(self.hsi.image, self.hs_vals[1:3])

        self.hsi.superset()
        self.assertEqual(len(self.hsi), 23)
        npt.assert_array_equal(self.hsi.image, self.hs_vals)

    def test_fit(self):
        param_img, score = self.hsi.fit(n_workers=10, x0=[1, 1, 1, 1], bounds=[(0, 0, 0, 0), (np.inf, np.inf, np.inf, 1)])
        npt.assert_allclose(param_img, self.bio_params, rtol=5, atol=1e-2)
        self.assertTrue(np.all(score < 2))
        

    # TODO: Test apply kernel bank
    # TODO: Test apply mask
class TestMergedHyperspectralImage(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_hsdfm_data(self)
        self.expected_list = [self.hs_vals - 1, self.hs_vals, self.hs_vals + 1]
        self.md_path = 'this/is/on/listed'

        with patch('hsdfmpm.hsdfm.hsdfm.read_metadata_json', return_value=self.md_vals):
            with patch('hsdfmpm.hsdfm.hsdfm.read_hyperstack') as hyperstacker:
                    hyperstacker.side_effect = self.expected_list
                    self.mhsi = MergedHyperspectralImage(image_paths=['dummy/path2', 'dummy/path2', 'dummy/path3'], metadata_path=self.md_path)

    def test_load_data(self):
        for mhsi, el in zip(self.mhsi.listed_hyperstack, self.expected_list):
            npt.assert_array_equal(mhsi._hyperstack, el)
            self.assertEqual(mhsi.metadata, self.md_vals)
            self.assertEqual(mhsi.metadata_path, Path(self.md_path))

    def test_listed_attr(self):
        metadata = self.mhsi.metadata
        for md in metadata:
            self.assertEqual(md, self.md_vals)

    def test_averagable_attr(self):
        npt.assert_array_equal(self.mhsi.hyperstack, np.mean(self.expected_list, axis=0))

    def test_dunders(self):
        self.assertEqual(len(self.mhsi), 3)
        npt.assert_array_equal(self.mhsi[0]._hyperstack, self.expected_list[0])
        for mhsi, el in zip(self.mhsi, self.expected_list):
            npt.assert_array_equal(mhsi._hyperstack, el)

    def test_array_dunder(self):
        npt.assert_array_equal(self.mhsi, np.mean(np.array(self.expected_list), axis=0))


if __name__ == '__main__':
    unittest.main()
