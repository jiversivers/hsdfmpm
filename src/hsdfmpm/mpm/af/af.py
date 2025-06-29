from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Annotated

import numpy as np
import pandas as pd
from hsdfmpm.mpm.af.utils import find_dated_power_file
from matplotlib.colors import Colormap
from pydantic import model_validator, BaseModel, computed_field, AfterValidator


from .utils import get_transfer_function
from ..utils import read_xml, get_pvstate_values
from ...utils import ImageData, ensure_path, colorize


class AutofluorescenceImage(ImageData):
    metadata_ext: str = ".xml"
    image_ext: str = ".ome.tif"
    power_file_path: Annotated[Union[str, Path], AfterValidator(ensure_path)]

    @model_validator(mode="after")
    def set_metadata_path(self):
        # Validate metadata
        if self.metadata_path is None:
            matches = list(self.image_path.glob(f"*{self.metadata_ext}"))
            if not matches:
                raise ValueError(
                    f"No file found in '{self.image_path}' with extension '{self.metadata_ext}'."
                )
            self.metadata_path = matches[0]

        self.metadata_path = Path(self.metadata_path)
        self._active = self.hyperstack

        # Load metadata into attributes
        root = read_xml(self.metadata_path)
        for val in root.iter("PVScan"):
            self.date = datetime.strptime(val.get("date"), "%m/%d/%Y %I:%M:%S %p")
        self.metadata = get_pvstate_values(root)
        self.attenuation = float(
            self.metadata["laserPower"]["elements"]["IndexedValue"][0]["value"]
        )
        self.wavelength = int(
            self.metadata["laserWavelength"]["elements"]["IndexedValue"][0]["value"]
        )
        self.laser = self.metadata["laserWavelength"]["elements"]["IndexedValue"][0][
            "description"
        ]
        self.gain = np.array(
            [
                float(pmt["value"])
                for pmt in self.metadata["pmtGain"]["elements"]["IndexedValue"]
            ]
        )
        if not self.power_file_path.is_file():
            self.power_file_path = find_dated_power_file(
                self.date, self.power_file_path
            )
        if self.power_file_path.suffix in [".xlsx", ".xls"]:
            self.power = pd.read_excel(self.power_file_path)
        elif self.power_file_path.suffix == ".csv":
            self.power = pd.read_csv(self.power_file_path, dtype=float)
        return self

    def normalize(self):
        # Get reference attenuation and measured power
        refAtt = self.power["Unnamed: 0"].values
        if self.power_file_path.suffix == ".csv":
            wl_key = str(self.wavelength)
        else:
            wl_key = self.wavelength
        refPwr = self.power[wl_key].values

        # Fit a line to the measures
        design_matrix = np.column_stack((np.ones(len(refAtt)), refAtt))
        b, m = np.linalg.lstsq(design_matrix, refPwr)[0]

        # Predict the actual power to this reference line
        self.norm_pwr = m * self.attenuation + b
        transfer_function = get_transfer_function(self.date, self.laser)
        self._active = transfer_function(
            img=self.raw,
            pwr=self.norm_pwr,
            gain=self.gain,
            pmt=np.arange(self.shape[0]),
        )


class OpticalRedoxRatio(BaseModel):
    ex755: Union[str, Path, AutofluorescenceImage]
    ex855: Union[str, Path, AutofluorescenceImage]
    power_file_path: Optional[Union[str, Path]] = None
    mask: Optional[np.ndarray] = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def load_images(self):
        # Make sure power info is available
        # If power is none, it has to be implemented in AF Images
        if self.power_file_path is None and (
            isinstance(self.ex755, (str, Path)) or isinstance(self.ex855, (str, Path))
        ):
            raise ValueError(
                "Must specify 'power_file_path' if both 'ex755' and 'ex855' are paths."
            )
        # If AF images are just paths, power file must be loaded from here. AF Image check for date-implicit loading
        elif isinstance(self.ex755, (str, Path)) or isinstance(self.ex855, (str, Path)):
            self.ex755 = AutofluorescenceImage(
                image_path=self.ex755, power_file_path=self.power_file_path
            )
            self.ex855 = AutofluorescenceImage(
                image_path=self.ex855, power_file_path=self.power_file_path
            )
        # Normalize
        self.ex855.normalize()
        self.ex755.normalize()
        return self

    @computed_field
    @property
    def fad(self) -> np.ndarray:
        return self.ex855[1].copy()

    @computed_field
    @property
    def nadh(self) -> np.ndarray:
        return self.ex755[2].copy()

    @computed_field
    @property
    def map(self) -> np.ndarray:
        return self.fad / (self.nadh + self.fad)

    def colorize(self, **kwargs) -> tuple[np.ndarray[float], Colormap]:
        intensity = (self.fad + self.nadh) / 2
        return colorize(self.map, intensity, **kwargs)

    @computed_field
    @property
    def pixel_wise(self) -> np.float64:
        return np.average(self.map)

    @computed_field
    @property
    def fluorescence(self) -> np.float64:
        return np.average(self.fad) / (np.average(self.nadh) + np.average(self.fad))

    def bin(self, **kwargs):
        self.ex755.bin(**kwargs)
        self.ex855.bin(**kwargs)

    def resize_to(self, h: int):
        self.ex755.resize_to(h)
        self.ex855.resize_to(h)
