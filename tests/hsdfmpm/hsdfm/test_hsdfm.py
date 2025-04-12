import os
import unittest

from hsdfmpm.hsdfm import HyperspectralImage


class TestHyperSpectralImageCube(unittest.TestCase):
    def test_image_path_field_validator(self):
        self.assertRaises(ValueError, HyperspectralImage, image_path='not/a/good/path')
        self.assertRaises(ValueError, HyperspectralImage, image_path=os.getcwd())  # Good path, but no json


if __name__ == '__main__':
    unittest.main()
