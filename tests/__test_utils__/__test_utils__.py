import itertools
from datetime import datetime
from itertools import product
import pathlib
from unittest.mock import patch, MagicMock
from xml.etree.ElementTree import Element, ElementTree

import numpy as np
import pandas as pd
from hsdfmpm.mpm import AutofluorescenceImage, OpticalRedoxRatio
from photon_canon.lut import LUT
from photon_canon.contrib.bio import model_from_hemoglobin

rng = np.random.default_rng(42)
lut = LUT(dimensions=['mu_s', 'mu_a'], scale=50000, simulation_id=88, extrapolate=True)
wavelengths = np.arange(500, 730, 10)

def patch_path_validators(self):
    """Helper for testing to patch pathlib.Path methods used in path validators and searches."""
    # Patches for file-paths (one-time patch for the test)
    is_dir_patch = patch.object(pathlib.Path, 'is_dir', new=lambda x: '.' not in str(x))
    is_file_patch = patch.object(pathlib.Path, 'is_file', new=lambda x: '.' in str(x))
    glob_patch = patch('pathlib.Path.glob', side_effect=lambda x:[pathlib.Path(f'/path/to/metadata_{x.strip('*')}')])

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
    self.bio_params = np.array([2.5 * rng.random((10, 10)) + 0.5, 1.5 * rng.random((10, 10)) + 0.5, rng.random((10, 10)), rng.random((10, 10))])
    self.hs_vals = np.zeros((len(wavelengths), 10, 10), dtype=np.float32)
    for y, x in product(range(10), range(10)):
        self.hs_vals[:, y, x] = model_from_hemoglobin(lut, wavelengths, *self.bio_params[:, y, x])
    self.scalar = 2

    # Mock normalization arrays
    self.std_arr = rng.random(self.hs_vals.shape, dtype=np.float64)
    self.bg_arr = rng.random(self.hs_vals.shape, dtype=np.float64)

def add_patch_af_data(self):
    """Helper for testing to add patch mpm data to the test object."""
    self.dates = ['12/11/1009 8:07:06 PM', '01/02/3456 7:01:23 AM']
    self.mmddyyy = [
        datetime.strftime(datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p"), "%m%d%Y")
        for date in self.dates
    ]
    self.power_used = [30, 35]
    self.ref_attenuation = [25, 30, 35, 40]
    self.laser_power = [[m * a - b for a in self.ref_attenuation] for m, b in zip([4.5, 5.5], [-0.25, 0.25])]
    self.laser_wavelength = [755, 855]
    self.laser = 'InsightX3'
    self.pmt_gains = [400, 500, 600, 700]
    self.g_params = [[1.00e-22, 7.8731],
                     [6.55e-24, 8.029222607]]
    self.offsets = [[0.0, 0.0, 0.0, 0.0],
                    [379.7346952,446.6135238,379.3671238,71.04764762]]
    self.md_dicts = [{
        'laserPower': {'elements': {'IndexedValue': [{'value': power}]}},
        'laserWavelength': {'elements': {'IndexedValue': [{'value': wavelength, 'description': self.laser}]}},
        'pmtGain': {'elements': {'IndexedValue': [{'value': gain} for gain in self.pmt_gains]}}
    } for power, wavelength in zip(self.power_used, self.laser_wavelength)]

    self.power = pd.DataFrame(
        {"Unnamed: 0":self.ref_attenuation,
         self.laser_wavelength[0]:self.laser_power[0],
         self.laser_wavelength[1]:self.laser_power[1]}
    )

    self.ex755 = rng.integers(0, 2**16, size=(4, 256, 256))
    self.ex855 = rng.integers(0, 2**16, size=(4, 256, 256))

    # Add actual mocks to self
    root_mock = MagicMock(spec=Element)
    root_mock.iter.side_effect = itertools.cycle([[{'date': date}] for date in self.dates])

    tree_mock = MagicMock(spec=ElementTree)
    tree_mock.getroot.return_value = root_mock

    with patch('xml.etree.ElementTree.parse', return_value=tree_mock):
        with patch('hsdfmpm.mpm.af.af.get_pvstate_values', side_effect=itertools.cycle(self.md_dicts)):
            with patch('hsdfmpm.utils.read_hyperstack', side_effect=itertools.cycle([self.ex755.copy(), self.ex855.copy()])):
                with patch('pandas.read_excel', return_value=self.power):
                    self.mock_755 = AutofluorescenceImage(
                        image_path='path/to/images/redox755',
                        power_file_path='power/dir',
                    )
                    self.mock_855 = AutofluorescenceImage(
                        image_path='path/to/images/redox855',
                        power_file_path='power/dir',
                    )

                    self.mock_orr_path = OpticalRedoxRatio(
                        ex755='path/to/images/redox755',
                        ex855='path/to/images/redox855',
                        power_file_path='power/dir',
                    )

                    self.mock_orr_obj = OpticalRedoxRatio(
                        ex755=self.mock_755,
                        ex855=self.mock_855,
                    )

    self.normalized = [(
            ((img - np.asarray(offset)[..., np.newaxis, np.newaxis]) /
             (self.power[wavel][self.power["Unnamed: 0"] == pwr].values.item() ** 2)) /
            (g_params[0] * np.asarray(self.pmt_gains)[..., np.newaxis, np.newaxis] ** g_params[1])
    ) for img, offset, wavel, pwr, g_params in zip([self.ex755, self.ex855], self.offsets, self.laser_wavelength, self.power_used, self.g_params)]

    self.orr_objects = [self.mock_orr_obj, self.mock_orr_path]
    self.orr = self.normalized[1][1] / (self.normalized[1][1] + self.normalized[0][2])


def generate_decay_histogram(tau1, tau2=None, alpha=1.0, n_photons=1e3, bin_count=256, t_max=10.0e-9):
    tau1 = np.asanyarray(tau1, dtype=float)
    if tau2 is None:
        tau2 = tau1
        alpha = 1.0
    tau2 = np.asanyarray(tau2, dtype=float)
    alpha = np.asanyarray(alpha, dtype=float)

    t_edges = (np.linspace(0, t_max, bin_count + 1))[np.newaxis, np.newaxis, ...]
    t0, t1 = t_edges[..., :-1], t_edges[..., 1:]

    p1 = np.exp(-t0 / tau1) - np.exp(-t1 / tau1)
    p2 = np.exp(-t0 / tau2) - np.exp(-t1 / tau2)
    probs = alpha * p1 + (1 - alpha) * p2

    probs /= np.sum(probs, axis=-1, keepdims=True)
    flat = probs.reshape(-1, bin_count)
    hists = np.asarray([rng.multinomial(n_photons, p).astype(np.uint32) for p in flat]).reshape(probs.shape)
    return hists


def convolve_with_irf(decay, irf):
    while decay.ndim < irf.ndim:
        decay = np.expand_dims(decay, axis=0)
    T = decay.shape[-1]
    conv_L = decay.shape[-1] + irf.shape[-1] - 1
    decay = np.fft.fft(decay, n=conv_L, axis=-1)
    irf = np.fft.fft(irf / irf.sum(axis=-1, keepdims=-1), n=conv_L, axis=-1)
    decay *= irf
    decay = np.fft.ifft(decay, axis=-1).real
    return decay[..., :T]


def gaussian_irf(fwhm: float,
                 period_ns: float = 10,
                 bin_count: int   = 256,
                 centre: float    = 2,
                 oversample: int  = 8):
    # time axis at oversampled resolution
    T      = period_ns        # repetition period
    dt     = T / (bin_count * oversample)
    t_fast = np.arange(0, T, dt)                                # (N_fast,)

    # Gaussian parameters
    sigma  = fwhm / 2.35  # (FWHM=2.355σ)
    irf_f  = np.exp(-0.5 * ((t_fast - centre)/sigma)**2)

    # fold into [0, T) so we keep periodicity
    irf_folded = np.zeros(bin_count * oversample)
    idx = (t_fast / dt).astype(int)
    irf_folded[idx % irf_folded.size] += irf_f

    # bin down to TCSPC resolution
    irf = irf_folded.reshape(bin_count, oversample).sum(axis=1)

    return irf


def add_patch_flim_data(self):
    """Helper for testing to add patch flim data to the test object."""
    self.bins = 256
    self.total_photons = 1e3

    self.sdt_md = {
        "measurementInfo": {
            "adc_re": self.bins,
            "tac_r": 5.0033573728569536e-08,
            "tac_g": 5
        }}
    tau1, tau2 = rng.random(2, dtype=np.float64)
    self.tau1 = tau1 * 1e-9
    self.tau2 = tau2 * 5e-9
    self.alpha = rng.uniform(0.4, 0.6, size=(256, 256, 1))
    tau1 = np.full_like(self.alpha, self.tau1)
    tau2 = np.full_like(self.alpha, self.tau2)
    decay = generate_decay_histogram(tau1, tau2, alpha=self.alpha, n_photons=self.total_photons, bin_count=self.bins)
    self.true_decay = decay
    self.irf = gaussian_irf(fwhm=1.0, period_ns=10, bin_count=self.bins, centre=1, oversample=8)[np.newaxis, np.newaxis, np.newaxis, ...]
    self.decay = convolve_with_irf(decay, self.irf)

