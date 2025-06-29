import warnings
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pydantic import model_validator, computed_field
from scipy.ndimage import generic_filter

from hsdfmpm.mpm.flim.utils import (
    open_sdt_file_with_json_metadata,
    polar_from_cartesian,
    lifetime_from_cartesian,
    get_phasor_coordinates,
    find_intersection_with_circle,
    project_to_line,
    fit_phasor,
    get_endpoints_from_projection,
    complex_phasor,
    cartesian_from_lifetime,
)
from hsdfmpm.utils import SerializableModel, DATA_PATH, ImageData, ensure_path

# Make IRF dir
IRF_PATH = DATA_PATH / "irf"
IRF_PATH.mkdir(parents=True, exist_ok=True)


class LifetimeImage(ImageData):
    metadata_ext: str = ".xml"
    image_ext: str = ".sdt"
    calibration: Optional["InstrumentResponseFunction"] = None
    frequency: float = 80e6
    harmonic: int = 1

    @model_validator(mode="after")
    def add_flim_metadata(self):
        self.metadata = self.sdt_data[1]
        self.period = self.metadata["measurementInfo"]["adc_re"]
        self.bin_width = (
            self.metadata["measurementInfo"]["tac_r"]
            / self.metadata["measurementInfo"]["tac_g"]
            / self.period
        )  # in ns
        self.omega = 2 * np.pi * self.harmonic * self.frequency  # in rad/s
        self._active = self.decay
        return self

    @computed_field
    @property
    def sdt_data(self) -> tuple[np.ndarray, dict]:
        """Helper method to keep metadata and hyperstack DRY"""
        filename = list(self.image_path.glob(f"*{self.image_ext}"))[0]
        self._hyperstack, sdt_metadata = open_sdt_file_with_json_metadata(filename)
        return self._hyperstack, sdt_metadata

    @computed_field
    def decay(self) -> np.ndarray:
        """Aliasing for clarity with other resources."""
        if self._active is None:
            self._active = self.hyperstack
        return self.image

    def load_irf(
        self,
        irf: Optional[Union["InstrumentResponseFunction", str]] = None,
        reference_lifetime: float = 0,
    ):
        # Load the IRF object
        self.calibration = get_irf(irf)

    def deconvolve(self):
        # TODO: Adde weiner correction term for tail-truncation of true convolution
        T = self.shape[-1]
        conv_L = T + self.calibration.shape[-1] - 1
        decay = np.fft.fft(self.decay, n=conv_L, axis=-1)
        irf = np.fft.fft(
            self.calibration / self.calibration.decay.sum(axis=-1, keepdims=-1),
            n=conv_L,
            axis=-1,
        )
        decay /= irf
        decay = np.fft.ifft(decay, axis=-1).real
        self._active = decay[..., :T]

    def fit(
        self, model: Callable[[float, float, ...], np.ndarray[float]], **kwargs
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        if not hasattr(self, "deconvolved"):
            self.deconvolve()
        _active = self._active.copy()
        out_image = np.zeros(self.shape[:-1])
        red_chi_sq = np.zeros(self.shape[:-1])
        for i, ch in enumerate(_active):
            self._active = ch.permute(2, 0, 1)  # Move time axis to front
            out_image[i], red_chi_sq[i] = super().fit(model, **kwargs)
        self._active = _active
        return out_image, red_chi_sq

    def phasor_coordinates(
        self,
        correction: bool = True,
        threshold: float = 0,
        median_filter_count: int = 0,
        k_size: Union[tuple[int, int], int] = (3, 3),
        as_complex: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        P, self.photons = get_phasor_coordinates(
            self.decay,
            bin_width=self.bin_width,
            frequency=self.frequency,
            harmonic=self.harmonic,
            as_complex=True,
        )

        # Apply correction
        if self.calibration is not None and correction:
            # Apply correction
            P *= self.calibration.correction
        self.g, self.s = P.real, P.imag

        # Convert to polar coordinates
        phase, modulation = polar_from_cartesian(self.g, self.s)

        # Add derivative attributes
        # Phi: Phase, M: Modulation
        self.phi = phase
        self.m = modulation

        # Mask the coordinates by photon counts
        self.g = np.where(self.photons > threshold, self.g, np.nan)
        self.s = np.where(self.photons > threshold, self.s, np.nan)

        # Median filter G and S for median_filter_count passes with input size kernel
        for _ in range(median_filter_count):
            self.g = generic_filter(self.g, np.nanmedian, size=k_size, mode="nearest")
            self.s = generic_filter(self.s, np.nanmedian, size=k_size, mode="nearest")

        with np.errstate(divide="ignore", invalid="ignore"):
            # Tau_phi: Phi-based lifetime, Tau_m: Modulation based lifetime
            self.tau_phi = (1 / self.omega) * np.tan(phase)
            self.tau_m = (1 / self.omega) * np.sqrt((modulation**-2) - 1)

        if as_complex:
            return complex_phasor(self.g, self.s)
        return self.g, self.s

    def get_phasor_line(self, **kwargs):
        if not (hasattr(self, "g") and hasattr(self, "s")) or kwargs:
            self.phasor_coordinates(**kwargs)

        # Fit a line to the cloud
        fit_results = fit_phasor(self.g, self.s)
        b, m = fit_results["fit_y_intercept"], fit_results["fit_slope"]
        for k, v in fit_results.items():
            setattr(self, k, v)

        # find circle interseciton points
        x, y = find_intersection_with_circle(b, m)

        return {"intersection": (x, y), "point-slope": (b, m)}

    def fit_for_lifetime_approximations(self, **kwargs):
        if not (hasattr(self, "g") and hasattr(self, "s")) or kwargs:
            self.phasor_coordinates(**kwargs)

        x, y = self.get_phasor_line(**kwargs)["intersection"]

        # Project to the line segment
        gp, sp = project_to_line(self.g, self.s, x, y)

        # Find lifetime components
        tau = lifetime_from_cartesian(x, y, self.omega)

        # Use projected points to find fraction and lifetime
        self.alphas, self.tau_m = get_endpoints_from_projection(gp, sp, x, y, tau)

        return self.alphas, self.tau_m, tau

    def phasor_plot(
        self, circ: bool = True, ax: Optional[plt.Axes] = None, **kwargs
    ) -> plt.Axes:
        g, s = self.phasor_coordinates(**kwargs)
        for ch, (g_ch, s_ch) in enumerate(zip(g, s)):
            x, y = g_ch.flatten(), s_ch.flatten()
            sns.scatterplot(
                x=x, y=y, s=1, color=".15", label=f"Phasor Coordinates, CH: {ch}"
            )
            sns.histplot(x=x, y=y, bins=50, pthresh=0.1, cmap="mako")
            sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

        ax = ax if ax is not None else plt.gca()
        ax.set(xlim=(0, 1), ylim=(0, 0.7), xlabel="G", ylabel="S")
        ax.legend()
        return ax

    def bin(self, bin_factor: int = 4):
        bands, h, w, t = self._active.shape
        h_binned, w_binned = h // bin_factor, w // bin_factor

        # Crop to multiples of bin_factor along H and W only
        cropped = self._active[:, : h_binned * bin_factor, : w_binned * bin_factor, :]

        # Reshape → (C, H', f, W', f, T) so the two *f* axes can be averaged
        reshaped = cropped.reshape(bands, h_binned, bin_factor, w_binned, bin_factor, t)

        # Average over the two binning axes (2 and 4)
        self._active = reshaped.sum(axis=(2, 4))


class InstrumentResponseFunction(LifetimeImage, SerializableModel):
    reference_lifetime: float
    new_size: Optional[int] = None

    @model_validator(mode="after")
    def calculate_correction(self):
        if self.new_size is not None:
            self.resize_to(self.new_size)

        # Get raw coordinates
        P_ref = self.phasor_coordinates(correction=False, as_complex=True)

        # Get references
        P_true = cartesian_from_lifetime(
            self.reference_lifetime, self.omega, as_complex=True
        )

        # Calculate complex correction factor
        self.correction = P_true / P_ref

        return self

    def store(self, path: Optional[str] = None):
        if path is None:
            path = IRF_PATH
        path = ensure_path(path)
        if path.is_dir():
            self.save_pickle(path / "irf.pkl")
        else:
            self.save_pickle(path)

    @classmethod
    def load(
        cls,
        path: Optional[Union[Path, str]] = None,
        reference_lifetime: Optional[float] = None,
        **kwargs,
    ) -> "InstrumentResponseFunction":
        if path is None:
            irf = get_irf(path)
            return irf
        elif isinstance(path, (str, Path)):
            path = ensure_path(path)

            # Load (extension-based method)
            ext = path.suffix.lower()
            if ext == ".sdt":
                image_ext = path.name
                path = path.parent
                return InstrumentResponseFunction(
                    image_path=path,
                    image_ext=image_ext,
                    reference_lifetime=reference_lifetime,
                    **kwargs,
                )
            if ext == ".pkl":
                if reference_lifetime is not None:
                    warnings.warn(
                        "Reference lifetime is not used when .pkl is loaded. "
                        "To update IRF reference lifetime, reload the .sdt file with new reference and <obj>.store again.",
                        Warning,
                        stacklevel=2,
                    )
                return cls.load_pickle(path)


# TODO: Add a user input pop up to get the reference lifetime if a LifetimeImage is input
# TODO: Add file chooser for IRF files in datapath when more than one is present.
def get_irf(
    irf: Optional[Union["InstrumentResponseFunction", str]] = None,
) -> "InstrumentResponseFunction":
    """
    This is a helper function to parse the input IRF as either a file, a preloaded InstrumentResponseFunction, or None.
    In the case of None, the default path is from in .hsdfm package path. This is searched and the most recent IRF
    filepath is used. In this case and the case of an input filepath, the .sdt file is loaded into a LifetimeImage
    object and returned. In the case of an input LifetimeImage object, it is simply returned.

    :param irf: The path to the IRF file, or a preloaded IRF in a InstrumentResponseFunction object. Optional.
    :type irf: Union['InstrumentResponseFunction', str]

    :raises FileNotFoundError: If no file is selected when required.
    :raises ValueError: If the IRF file format is unsupported or corrupt.

    :return: A InstrumentResponseFunction object for the IRF file.
    :rtype: InstrumentResponseFunction
    """
    if isinstance(irf, InstrumentResponseFunction):
        return irf

    # Search data path if no path input
    if irf is None:
        file = list(IRF_PATH.glob("irf.pkl"))[-1]
        warnings.warn(f"Loading default IRF file, '{file}'.", Warning, stacklevel=2)
    # Look for pickles if dir given
    else:
        irf = Path(irf)
        if irf.is_dir():
            file = list(irf.glob("irf.pkl"))[-1]
        else:
            file = irf

    # Check for file existence
    if not file:
        raise FileNotFoundError(f"No IRF file found at {irf}.")

    irf = InstrumentResponseFunction.load(path=file)

    return irf
