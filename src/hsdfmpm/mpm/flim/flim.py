import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.polynomial.polynomial import polyfit
from matplotlib import pyplot as plt
import seaborn as sns
from pydantic import model_validator, computed_field, root_validator, field_validator, BaseModel
from scipy.ndimage import median_filter

from hsdfmpm.mpm.flim.utils import open_sdt_file_with_json_metadata, cartesian_from_polar, polar_from_cartesian, \
    lifetime_from_cartesian, polar_from_lifetime, get_phasor_coordinates, find_intersection_with_circle, \
    project_to_line, fit_phasor, get_endpoints_from_projection
from hsdfmpm.utils import SerializableModel, DATA_PATH, autoversion, ImageData, ensure_path

# Make IRF dir
IRF_PATH = DATA_PATH / "irf"
IRF_PATH.mkdir(parents=True, exist_ok=True)

class LifetimeImage(ImageData):
    metadata_ext: str = '.xml'
    image_ext: str = '.sdt'
    calibration: Optional['InstrumentResponseFunction'] = None
    frequency: float = 80e6
    harmonic: int = 1

    @model_validator(mode='after')
    def add_flim_metadata(self):
        self.period = self.metadata['measurementInfo']['adc_re']
        self.bin_width = (
                self.metadata['measurementInfo']['tac_r'] / self.metadata['measurementInfo']['tac_g'] / self.period
        )  # in ns
        self.omega = 2 * np.pi * self.harmonic * self.frequency  # in rad/s
        return self

    @computed_field
    @property
    def sdt_data(self) -> tuple[np.ndarray, dict]:
        """Helper method to keep metadata and hyperstack DRY"""
        filename = list(self.image_path.glob(f'*{self.image_ext}'))[0]
        self._hyperstack, sdt_metadata = open_sdt_file_with_json_metadata(filename)
        return self._hyperstack, sdt_metadata

    @computed_field
    @property
    def metadata(self) -> dict:
        """Get built-in SDT metadata"""
        _, metadata = self.sdt_data
        return metadata

    @computed_field
    def decay(self) -> np.ndarray:
        """Aliasing hyperstack for clarity with other resources."""
        return self.hyperstack

    def load_irf(self,
                 irf: Optional[Union['InstrumentResponseFunction', str]] = None,
                 reference_lifetime: float = 0):
        # Load the IRF object
        self.calibration = get_irf(irf)

    def phasor_coordinates(self,
                           correction: bool = True,
                           threshold: float = 0,
                           median_filter_count: int = 0,
                           k_size: Union[tuple[int, int], int] = (3, 3)
                           ) -> np.ndarray:

        g, s, photons = get_phasor_coordinates(self.decay,
                                               bin_width=self.bin_width,
                                               frequency=self.frequency,
                                               harmonic=self.harmonic,
                                               threshold=threshold)

        # Convert to polar coordinates
        phase, modulation = polar_from_cartesian(g, s)

        # Apply correction
        if self.calibration is not None and correction:
            # Apply correction
            phase += self.calibration.phase_offset
            modulation *= self.calibration.modulation_factor

            # Convert to cartesian
            g, s = cartesian_from_polar(phase, modulation)

        # Add derivative attributes
        # Phi: Phase, M: Modulation
        self.phi = phase
        self.m = modulation

        # Median filter G and S for median_filter_count passes with input size kernel
        for _ in range(median_filter_count):
            g = median_filter(g, k_size)
            s = median_filter(s, k_size)

        # Tau_phi: Phi-based lifetime, Tau_m: Modulation based lifetime
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tau_phi = (1 / self.omega) * np.tan(phase)
            self.tau_m = (1 / self.omega) * np.sqrt((modulation ** -2) - 1)

        return g, s

    def fit_for_lifetime_approximations(self, **kwargs):
        # Get the coords
        g, s = self.phasor_coordinates(**kwargs)

        # Fit a line to the cloud
        b, m = fit_phasor(g, s)

        # find circle interseciton points
        x, y = find_intersection_with_circle(b, m)

        # Project to the line segment
        gp, sp = project_to_line(g, s, x, y)

        # Find lifetime components
        tau = lifetime_from_cartesian(x, y, self.omega)

        # Use projected points to find fraction and lifetime
        self.alphas, self.tau_m = get_endpoints_from_projection(gp, sp, x, y, tau)

        return self.alphas, self.tau_m, tau

    def phasor_plot(self, circ: bool = True, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        g, s = self.phasor_coordinates(**kwargs)
        for ch in self.channels:
            x, y = g[ch].flatten(), s[ch].flatten()
            sns.scatterplot(x=x, y=y, s=1, color='.15', label=f'Phasor Coordinates, CH: {ch}')
            sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
            sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

        ax = ax if ax is not None else plt.gca()
        ax.set(xlim=(0, 1), ylim=(0, 0.7), xlabel='G', ylabel='S')
        ax.legend()
        return ax

@autoversion(major=1, minor=0)
class InstrumentResponseFunction(LifetimeImage, SerializableModel):
    reference_lifetime: float

    @model_validator(mode='after')
    def calculate_correction(self):
        # Get raw coordinates
        g, s = self.phasor_coordinates(correction=False)

        # Convert to polar
        phase, modulation = polar_from_cartesian(g, s)

        # Get references
        self.reference_phase, self.reference_modulation = polar_from_lifetime(self.reference_lifetime, self.omega)

        # Calculate correction factors
        self.phase_offset = self.reference_phase - phase
        self.modulation_factor = self.reference_modulation / modulation
        return self

    def store(self, path: Optional[str] = None):
        if path is None:
            path = IRF_PATH
        if path.is_dir():
            self.save_pickle(path / f'irf.pkl')
        else:
            self.save_pickle(path)

    @classmethod
    def load(cls,
             path: Optional[Union[Path, str]] = None,
             reference_lifetime: Optional[float] = None,
             **kwargs) -> 'InstrumentResponseFunction':
        if path is None:
            irf = get_irf(path)
            return irf
        elif isinstance(path, (str, Path)):
            path = ensure_path(path)

            # Load (extension-based method)
            ext = path.suffix.lower()
            if ext == '.sdt':
                image_ext = path.name
                path = path.parent
                return InstrumentResponseFunction(image_path=path,
                                                  image_ext=image_ext,
                                                  reference_lifetime=reference_lifetime,
                                                  **kwargs)
            if ext == '.pkl':
                if reference_lifetime is not None:
                    warnings.warn(
                        'Reference lifetime is not used when .pkl is loaded. '
                        'To update IRF reference lifetime, reload the .sdt file with new reference and <obj>.store again.',
                        Warning, stacklevel=2
                    )
                return cls.load_pickle(path)


# TODO: Add a user input pop up to get the reference lifetime if a LifetimeImage is input
# TODO: Add file chooser for IRF files in datapath when more than one is present.
def get_irf(
        irf: Optional[Union['InstrumentResponseFunction', str]] = None
) -> 'InstrumentResponseFunction':
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
        file = list(IRF_PATH.glob(f'irf.pkl'))[-1]
        warnings.warn(f"Loading default IRF file, '{file}'.", Warning, stacklevel=2)
    # Look for pickles if dir given
    else:
        irf = Path(irf)
        if irf.is_dir():
            file = list(irf.glob('irf.pkl'))[-1]
        else:
            file = irf

    # Check for file existence
    if not file:
        raise FileNotFoundError(f'No IRF file found at {irf}.')

    irf = InstrumentResponseFunction.load(path=file)

    return irf
