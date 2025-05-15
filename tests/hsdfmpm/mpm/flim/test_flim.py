import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import numpy.testing as npt

from hsdfmpm.mpm import LifetimeImage, InstrumentResponseFunction
from hsdfmpm.mpm.flim.utils import cartesian_from_lifetime

from tests.__test_utils__ import patch_path_validators, add_patch_flim_data

class TestFlim(unittest.TestCase):
    def setUp(self):
        patch_path_validators(self)
        add_patch_flim_data(self)
        self.frequency = 80e6
        with patch('hsdfmpm.mpm.flim.flim.open_sdt_file_with_json_metadata', return_value=(self.decay, self.sdt_md)):
            self.mock_decay = LifetimeImage(image_path='dummy/path', channels=0, frequency=self.frequency)

            with patch('hsdfmpm.mpm.flim.flim.open_sdt_file_with_json_metadata', return_value=(self.irf, self.sdt_md)):
                with tempfile.TemporaryDirectory() as tmpdir:
                    InstrumentResponseFunction.load(path='dummy/path/file.sdt', reference_lifetime=0, channels=0).store(Path(tmpdir))
                    self.mock_decay.load_irf(Path(tmpdir) / 'irf.pkl')

    def test_add_flim_metadata(self):
        self.assertEqual(self.mock_decay.metadata, self.sdt_md)
        self.assertEqual(self.mock_decay.period, self.bins)
        npt.assert_array_equal(self.mock_decay, self.decay)
        npt.assert_array_equal(self.mock_decay.calibration, self.irf)

    def test_phasor_coordinates(self):
        g, s = self.mock_decay.phasor_coordinates()
        g1, s1 = cartesian_from_lifetime(self.tau1, 2 * np.pi * self.frequency)
        g2, s2 = cartesian_from_lifetime(self.tau2, 2 * np.pi * self.frequency)
        gt = self.alpha * g1 + (1 - self.alpha) * g2
        st = self.alpha * s1 + (1 - self.alpha) * s2
        npt.assert_allclose(g.squeeze(), gt.squeeze(), atol=0.2, rtol=0.1)
        npt.assert_allclose(s.squeeze(), st.squeeze(), atol=0.2, rtol=0.1)
        self.assertAlmostEqual(g.mean(), gt.mean(), delta=0.1)
        self.assertAlmostEqual(s.mean(), st.mean(), delta=0.1)


if __name__ == '__main__':
    unittest.main()