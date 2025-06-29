import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy.testing as npt
import pandas.testing as pdt

from tests.__test_utils__ import patch_path_validators, add_patch_af_data


class TestAutofluorescenceImage(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_af_data(self)

    def test_set_metadata_path(self):
        for img, pwr, wavel, date in zip(
            [self.mock_755, self.mock_855],
            self.power_used,
            self.laser_wavelength,
            self.mmddyyy,
        ):
            self.assertEqual(img.metadata_path, Path("/path/to/metadata_.xml"))
            self.assertEqual(
                img.power_file_path, Path(f"/path/to/metadata_{date}*.xls")
            )
            self.assertEqual(datetime.strftime(img.date, "%m%d%Y"), date)
            self.assertEqual(img.attenuation, pwr)
            self.assertEqual(img.wavelength, wavel)
            npt.assert_array_equal(img.gain, self.pmt_gains)
            pdt.assert_frame_equal(img.power, self.power)

    def test_normalize(self):
        for img, true_norm, offset, pwr, g_params, wavel in zip(
            [self.mock_755, self.mock_855],
            self.normalized,
            self.offsets,
            self.power_used,
            self.g_params,
            self.laser_wavelength,
        ):
            img.normalize()
            npt.assert_allclose(img, true_norm)


class TestOpticalRedoxRatio(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_af_data(self)

    def test_load_images(self):
        for orr in self.orr_objects:
            npt.assert_allclose(self.normalized[0], orr.ex755)
            npt.assert_allclose(self.normalized[1], orr.ex855)

    def test_computed_fields(self):
        nadh, fad = self.normalized[0][2], self.normalized[1][1]
        for orr in self.orr_objects:
            npt.assert_allclose(nadh, orr.nadh)
            npt.assert_allclose(fad, orr.fad)
            npt.assert_allclose(self.orr, orr.map)
            self.assertAlmostEqual(self.orr.mean(), orr.pixel_wise)
            self.assertAlmostEqual(
                fad.mean() / (nadh.mean() + fad.mean()), orr.fluorescence
            )

    def test_colorize(self):
        for orr in self.orr_objects:
            with patch(
                "hsdfmpm.mpm.af.af.colorize", side_effect=lambda x, y: x * y
            ) as colorize_patch:
                color = orr.colorize()
                for img, arg in zip(
                    [orr.map, (orr.fad + orr.nadh) / 2], colorize_patch.call_args[0]
                ):
                    npt.assert_array_equal(img, arg)
                npt.assert_array_equal(color, orr.map * (orr.fad + orr.nadh) / 2)
            colorize_patch.assert_called_once()

    def test_bin_and_resize(self):
        for orr in self.orr_objects:
            orr.resize_to(32)
            self.assertEqual(orr.ex755.shape, (4, 32, 32))
            self.assertEqual(orr.ex855.shape, (4, 32, 32))

            orr.bin(bin_factor=8)
            self.assertEqual(orr.ex755.shape, (4, 4, 4))
            self.assertEqual(orr.ex855.shape, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
