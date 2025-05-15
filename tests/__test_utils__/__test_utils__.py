from itertools import product
import pathlib
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
    is_dir_patch = patch.object(pathlib.Path, 'is_dir', new=lambda x: '.' not in str(x))
    is_file_patch = patch.object(pathlib.Path, 'is_file', new=lambda x: '.' in str(x))
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
    print(self.type)
    pass

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
    self.tau2 = tau2 * 10e-9
    self.alpha = rng.uniform(0.4, 0.6, size=(256, 256, 1))
    tau1 = np.full_like(self.alpha, tau1)
    tau2 = np.full_like(self.alpha, tau2)
    decay = generate_decay_histogram(tau1, tau2, alpha=self.alpha, n_photons=self.total_photons, bin_count=self.bins)
    self.true_decay = decay
    self.irf = gaussian_irf(fwhm=1.0, period_ns=10, bin_count=self.bins, centre=0, oversample=8)[np.newaxis, np.newaxis, np.newaxis, ...]
    self.decay = convolve_with_irf(decay, self.irf)

