import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Annotated

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from pydantic import model_validator, BaseModel, computed_field, AfterValidator


from .utils import get_transfer_function, truncate_colormap, get_cmap
from ..utils import read_xml, get_pvstate_values
from ...utils import is_file, ImageData


class AutofluorescenceImage(ImageData):
    metadata_ext: str = '.xml'
    image_ext: str = '.ome.tif'
    power_file_path: Annotated[str, AfterValidator(is_file)]

    @model_validator(mode='after')
    def set_metadata_path(self):
        # Validate metadata
        if self.metadata_path is None:
            matches = list(self.image_path.glob(f'*{self.metadata_ext}'))
            if not matches:
                raise ValueError(f"No file found in '{self.image_path}' with extension '{self.metadata_ext}'.")
            self.metadata_path = matches[0]

        self.metadata_path = Path(self.metadata_path)
        return self

        # Load metadata into attributes
        root = read_xml(self.metadata_path)
        for val in root.iter('PVScan'):
            self.date = datetime.strptime(val.get('date'), "%m/%d/%Y %I:%M:%S %p")
        self.metadata = get_pvstate_values(root)
        self.attenuation = float(self.metadata['laserPowerAttenuation']['elements']['IndexedValue'][0]['value'])
        self.wavelength = float(self.metadata['laserWavelength']['elements']['IndexedValue'][0]['value'])
        self.laser = self.metadata['laserWavelength']['elements']['IndexedValue'][0]['description']
        self.gain = np.array([float(pmt['value']) for pmt in self.metadata['pmtGain']['elements']['IndexedValue']])
        self.power = pd.read_excel(self.power_file_path)
        return self

    def normalize_to_fluorescein(self):
        # Get reference attenuation and measured power
        refAtt = self.power['Unnamed: 0'].values
        refPwr = self.power[self.wavelength].values

        # Fit a line to the measures
        design_matrix = np.column_stack((np.ones(len(refAtt)), refAtt))
        b, m = np.linalg.lstsq(design_matrix, refPwr)[0]

        # Predict the actual power to this reference line
        self.norm_pwr = m * self.attenuation + b
        transfer_function = get_transfer_function(self.date, self.laser)
        self.normalized = transfer_function(
            img=self.hyperstack, pwr=self.norm_pwr, gain=self.gain
        )

class OpticalRedoxRatio(BaseModel):
    ex755: Union[str, AutofluorescenceImage]
    ex855: Union[str, AutofluorescenceImage]
    power_file_path: Optional[str] = None
    mask: Optional[np.ndarray] = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def load_images(self):
        # Load power file if left implicit
        if self.power_file_path is None:
            try:
                self.power_file_path = self.ex755.power_file_path
            except AttributeError:
                try:
                    self.power_file_path = self.ex855.power_file_path
                except AttributeError:
                    try:
                        self.power_file_path = list(Path(self.ex755.image_path).glob('*power*.xlsx'))[0]
                    except IndexError:
                        try:
                            self.power_file_path = list(Path(self.ex855.image_path).glob('*power*.xlsx'))[0]
                        except IndexError:
                            try:
                                self.power_file_path = list(Path(self.ex755).glob('*power*.xlsx'))[0]
                            except IndexError:
                                self.power_file_path = list(Path(self.ex855).glob('*power*.xlsx'))[0]
            warnings.warn(f'Power file loaded implicitly from {self.power_file_path}.', Warning, stacklevel=2)

        if isinstance(self.ex755, (str, Path)):
            self.ex755 = AutofluorescenceImage(image_path=self.ex755, power_file_path=self.power_file_path)
        if isinstance(self.ex855, (str, Path)):
            self.ex855 = AutofluorescenceImage(image_path=self.ex855, power_file_path=self.power_file_path)
        self.fad = self.ex855[1] if len(self.ex855) == 4 else self.ex855[0]
        self.nadh = self.ex755[2] if len(self.ex755) == 4 else self.ex755[0]
        return self

    @computed_field
    @cached_property
    def map(self) -> np.ndarray:
        return self.fad / (self.nadh + self.fad)

    def colorize(self,
                 cmap: Optional[Union[Colormap, np.ndarray]] = None,
                 cmin: float = 0.0,
                 cmax: float = 1.0,
                 n: int = 100,
                 intmin: float = 0.1,
                 intmax: float = 0.95):
        intensity = (self.fad + self.nadh) / 2

        if cmap is None:
            # Default cmap if none is input
            cmap = truncate_colormap('jet', 0.13, 0.88, n)
        else:
            # Validate input
            cmap = get_cmap(cmap)

        # Map colors with normed space (cmin, cmax)
        norm = Normalize(vmin=cmin, vmax=cmax, clip=True)
        colored_orr = cmap(norm(self.map))[:, :, :3]

        # Normalize intensity within limits
        intmin = np.percentile(intensity, intmin * 100)
        intmax = np.percentile(intensity, intmax * 100)
        intensity = (intensity - intmin) / (intmax - intmin)
        intensity[intensity < 0] = 0
        intensity[intensity > 1] = 1

        return colored_orr * intensity[..., np.newaxis], cmap

    @computed_field
    @property
    def pixel_wise(self) -> np.float64:
        return np.average(self.map)

    @computed_field
    @property
    def fluorescence(self) -> np.float64:
        return np.average(self.fad) / (np.average(self.nadh) + np.average(self.fad))
