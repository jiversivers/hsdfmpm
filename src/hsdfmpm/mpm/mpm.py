import glob
import os
import warnings
from functools import cached_property
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import field_validator, model_validator, BaseModel

from .sdt import open_sdt_file_with_json_metadata, get_irf
from .utils import get_transfer_function, PrairieViewImage, polar_from_cartesian, cartesian_from_polar
from ..utils import autoversion, DATA_PATH, SerializableModel


class MultiphotonImage(PrairieViewImage):
    metadata_ext: str = '.xml'
    image_ext: str = '.ome.tif'
    power_file_path: str

    @field_validator('power_file_path')
    @classmethod
    def validate_power_file(cls, power_file):
        if not os.path.isfile(power_file):
            raise FileNotFoundError(f"'{power_file}' is not a valid file.")
        return power_file

    def normalize_to_fluorescein(self):
        # Get reference attenuation and measured power
        refAtt = self.power['Unnamed: 0'].values
        refPwr = self.power[self.wavelength].values

        # Fit a line to the measures
        design_matrix = np.column_stack((np.ones(len(refAtt)), refAtt))
        b, m = np.linalg.lstsq(design_matrix, refPwr)[0]

        # Predict the actual power to this reference line
        att = float(self.metadata['laserPowerAttenuation']['elements']['IndexedValue'][0]['value'])
        self.norm_pwr = m * att + b
        transfer_function = get_transfer_function(self.date, self.laser)
        self.normalized = transfer_function(
            img=self.hyperstack, pwr=self.norm_pwr, gain=self.gain
        )

class LifetimeImage(PrairieViewImage):
    metadata_ext: str = '.xml'
    image_ext: str = '.sdt'
    calibration: Optional['InstrumentResponseFunction'] = None
    _parameters: Optional['DecayParameters'] = None
    _phasor_coordinates: Optional[np.ndarray] = None

    @cached_property
    def hyperstack(self):
        filename = glob.glob(os.path.join(self.image_path, f'*{self.image_ext}'))[0]
        # TODO: Check that the shape of the decay is as expected (C x H x W X T)
        self._hyperstack, self.metadata = open_sdt_file_with_json_metadata(filename)
        return self._hyperstack

    @cached_property
    def parameters(self) -> 'DecayParameters':
        self._parameters = DecayParameters.get_params(self)
        return self._parameters

    @property
    def decay(self):
        # Alias hyperstack for field consistency
        return self.hyperstack

    def load_irf(self,
                 irf: Optional[Union['InstrumentResponseFunction', str]] = None,
                 reference_lifetime: float = 0):
        # Load the IRF object
        irf = get_irf(irf)

        # Calibrate IRF
        irf.calculate_correction(reference_lifetime=reference_lifetime)

        # Save new IRF after creation
        self.calibration = irf
        irf.store()

    @cached_property
    def phasor_coordinates(self) -> np.ndarray:
        T = self.parameters.period
        dt = self.parameters.bin_width
        w = self.parameters.omega

        # Create an array of t (bin numbers) for each time frame of the decay
        t = np.expand_dims(np.arange(0.5, T, 1) * dt, axis=0)
        while t.ndim < self.hyperstack.ndim:
            t = np.expand_dims(t, axis=0)

        # Calculate raw phasor coordinates
        g = np.sum(self.decay * np.cos(w * t), axis=-1)
        s = np.sum(self.decay * np.sin(w * t), axis=-1)
        photons = np.sum(self.decay, axis=-1)

        # Normalize to counts
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(photons != 0, g / photons, 0)
            s = np.where(photons != 0, s / photons, 0)

        if self.calibration is not None:
            # Convert to polar coordinates
            phase, modulation = polar_from_cartesian(g, s)

            # Apply correction
            phase += self.calibration.phase_correction
            modulation *= self.calibration.modulation_correction

            # Convert to cartesian
            g, s = cartesian_from_polar(phase, modulation)

        self._phasor_coordinates = np.column_stack((g, s))
        return self._phasor_coordinates

    def phasor_plot(self, circ: bool = True, ax: Optional[plt.Axes] = None) -> plt.Axes:
        g, s = self.phasor_coordinates

        ax = plt.gca() if ax is None else ax

        if circ:
            x = np.linspace(0, 1, 0.1)
            y = np.sqrt((1 / 4) - (x - (1/2)) ** 2)
            ax.plot(x, y, 'k', label='Universal Circle')

        ax.scatter(g, s, label='Phasor Coordinates')

        return ax


@autoversion(major=1, minor=0)
class InstrumentResponseFunction(LifetimeImage, SerializableModel):
    modulation_correction: float = None
    phase_correction: float = None

    def calculate_correction(self, reference_lifetime):
        # Ensure self has no IRF
        self.calibration = None

        # Get raw coordinates
        g, s = self.phasor_coordinates

        # Convert to polar
        phase, modulation = polar_from_cartesian(g, s)

        # Calculate correction factors
        self.parameters.reference_modulation = 1 / np.sqrt(1 + (self.parameters.omega * reference_lifetime))
        self.parameters.reference_phase = np.arctan(self.parameters.omega / reference_lifetime)
        self.phase_correction = self.parameters.reference_phase - phase
        self.modulation_correction = self.parameters.reference_modulation / modulation

    def store(self, path: Optional[str] = None):
        if path is None:
            path = DATA_PATH
        if path.is_file():
            self.save_pickle(path)
        else:
            self.save_pickle(path / 'irf.pkl')


    @classmethod
    def load(cls, path: Optional[str] = None):
        if path is None:
            path = DATA_PATH
        if not path.is_file():
            file = path.glob('*irf.pkl')[-1]
            if not path:
                raise FileNotFoundError(f'No IRF file found at {path}.')
            warnings.warn(f'Loading default IRF file, {file}.')
        else:
            file = path
        ext = file.suffix.lower()
        if ext == '.sdt':
            return InstrumentResponseFunction(image_path=path)
        if ext == '.pkl':
            return cls.load_pickle(path)

class DecayParameters(BaseModel):
    frequency: float
    bin_width: float
    period: float
    omega: float
    reference_modulation: Optional[float] = None
    reference_phase: Optional[float] = None

    @classmethod
    def get_params(cls, decay: 'LifetimeImage',
                   reference_lifetime: Optional[float] = None
                   ) -> 'DecayParameters':
        frequency = decay.metadata['']
        bin_width = decay.metadata['']
        period = len(decay.hyperstack)
        omega = 2 * np.pi * frequency
        if reference_lifetime is not None:

        else:
            reference_modulation = None
            reference_phase = None

        return DecayParameters(
            frequency=frequency,
            bin_width=bin_width,
            period=period,
            omega=omega,
            reference_modulation=reference_modulation,
            reference_phase=reference_phase
        )
