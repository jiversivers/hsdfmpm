import unittest
from types import NoneType
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from pathlib import Path
import hsdfmpm.utils as utils
from hsdfmpm.utils import ImageData
from tests.__test_utils__ import patch_path_validators, add_patch_hsdfm_data


class TestUtilFunctions(unittest.TestCase):
    def setUp(self):
        self.test_file_path = "/path/to/totally/real.file"
        self.test_dir_path = "/path/to/totally/real/dir"

    def test_path_helpers(self):
        self.assertRaises(FileNotFoundError, utils.is_file, "/path/to/nonexistent.file")
        self.assertRaises(NotADirectoryError, utils.is_dir, "/path/to/nonexistent/dir")
        with patch("pathlib.Path.is_file", return_value=True):
            self.assertEqual(
                utils.ensure_path(self.test_file_path), Path(self.test_file_path)
            )
            self.assertEqual(
                utils.is_file(self.test_file_path), Path(self.test_file_path)
            )
        with patch("pathlib.Path.is_dir", return_value=True):
            self.assertEqual(utils.is_dir(self.test_dir_path), Path(self.test_dir_path))

    def test_channel_check(self):
        # None in
        channels = utils.channel_check(None)
        self.assertIsInstance(channels, NoneType)

        # Single int
        val = 1
        channels = utils.channel_check(val)
        self.assertIsInstance(channels, list)
        self.assertEqual(channels, [val])

        # One-element list in
        channels = utils.channel_check([val])
        self.assertIsInstance(channels, list)
        self.assertEqual(channels, [val])

        # List in
        val = [1, 2, 3]
        channels = utils.channel_check([1, 2, 3])
        self.assertIsInstance(channels, list)
        self.assertEqual(channels, val)

        # Tuple in
        val = (1, 2, 3)
        channels = utils.channel_check((1, 2, 3))
        self.assertIsInstance(channels, tuple)
        self.assertEqual(channels, val)

    def test_iterable_array(self):
        # Int in
        val = 1
        result = utils.iterable_array(val)
        self.assertIsInstance(result, np.ndarray)
        self.assertListEqual(result.tolist(), [val])

        # List in
        val = [1, 2, 3]
        result = utils.iterable_array(val)
        self.assertIsInstance(result, np.ndarray)
        self.assertListEqual(result.tolist(), val)

        # NDArray in
        val = np.array([1, 2, 3])
        result = utils.iterable_array(val)
        self.assertIsInstance(result, np.ndarray)
        npt.assert_array_equal(result, val)

    def test_vectorize_img(self):
        # 2D img
        test_img = np.random.rand(10, 10)
        result = utils.vectorize_img(test_img)
        self.assertEqual(result.shape, (100, 1))
        npt.assert_array_equal(result, test_img.ravel()[..., np.newaxis])

        # 3D img
        test_img = np.random.rand(3, 10, 10)
        result = utils.vectorize_img(test_img)
        self.assertEqual(result.shape, (100, 3))
        npt.assert_array_equal(
            result,
            np.column_stack(
                [test_img[0].flatten(), test_img[1].flatten(), test_img[2].flatten()]
            ),
        )

        # With locations
        result = utils.vectorize_img(test_img, include_location=True)
        self.assertEqual(result.shape, (100, 5))
        y = np.repeat(10 * np.arange(-0.5, 0.51, 1 / 9), 10).reshape(10, 10)
        x = y.T
        npt.assert_array_equal(result[:, 0], x.flatten())
        npt.assert_array_equal(result[:, 1], y.flatten())

    def test_read_hyperstack(self):
        img = np.random.rand(10, 10)
        fake_paths = [
            Path("some_file.tif"),
            Path("some_file2.tif"),
            Path("some_file3.tif"),
        ]

        with patch("pathlib.Path.glob") as globber:
            with patch("cv2.imread") as opener:
                # 3D Image
                globber.return_value = fake_paths
                opener.side_effect = [img, 2 * img, 3 * img]
                hs = utils.read_hyperstack("some/dir", ext=".tif")
                self.assertEqual(opener.call_count, 3)
                self.assertEqual(hs.shape, (3, 10, 10))
                npt.assert_array_equal(hs, np.stack([img, 2 * img, 3 * img]))
                self.assertEqual(opener.call_args_list[0][0][0], fake_paths[0])
                self.assertEqual(hs.dtype, np.float64)

                # 2D image
                globber.return_value = [fake_paths[0]]
                opener.side_effect = [img]
                hs = utils.read_hyperstack("some/dir", ext=".tif")
                self.assertEqual(opener.call_count, 4)
                self.assertEqual(hs.shape, (1, 10, 10))


class TestImageData(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_hsdfm_data(self)
        self.img = ImageData(image_path="some/path", metadata_ext=".xml")

    def test_hyperstack(self):
        # When _hyperstack is yet unloaded
        with patch("hsdfmpm.utils.read_hyperstack", return_value=self.hs_vals):
            npt.assert_array_equal(self.img.hyperstack, self.hs_vals)

        # When _hyperstack has already loaded
        self.img._hyperstack = 2 * self.hs_vals
        npt.assert_array_equal(self.img.hyperstack, 2 * self.hs_vals)

        # When channels are selected
        self.img.channels = [0]
        npt.assert_array_equal(self.img.hyperstack, 2 * self.hs_vals[[0]])

    def test_reset(self):
        # This loads the hyperstack
        with patch("hsdfmpm.utils.read_hyperstack", return_value=self.hs_vals):
            npt.assert_array_equal(self.img.hyperstack, self.hs_vals)
        self.img._active = self.img._hyperstack + 1
        npt.assert_array_equal(self.img.image, self.hs_vals + 1)
        with patch("hsdfmpm.utils.read_hyperstack", return_value=self.hs_vals):
            self.img.reset()
        npt.assert_array_equal(self.img.image, self.hs_vals)


if __name__ == "__main__":
    unittest.main()
