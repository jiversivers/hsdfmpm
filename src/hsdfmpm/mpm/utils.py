import datetime
import importlib
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pydantic import model_validator

from hsdfmpm.utils import iterable_array, ImageData


class PrairieViewImage(ImageData):
    @model_validator(mode='after')
    def build_metadata_attr(self):
        # Load metadata
        root = read_xml(self.metadata_path)
        for val in root.iter('PVScan'):
            self.date = datetime.strptime(val.get('date'), "%m/%d/%Y %I:%M:%S %p")
        self.metadata = get_pvstate_values(root)
        self.power = pd.read_excel(self.power_file_path)
        self.wavelength = float(self.metadata['laserWavelength']['elements']['IndexedValue'][0]['value'])
        self.laser = self.metadata['laserWavelength']['elements']['IndexedValue'][0]['description']
        self.gain = np.array([float(pmt['value']) for pmt in self.metadata['pmtGain']['elements']['IndexedValue']])
        return self

def read_xml(file_path: str) -> pd.DataFrame:
    """Reads and parses an XML file, then prints the root element's tag and text of all elements."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

    except FileNotFoundError as e:
        raise Exception(f"Error: File not found at '{file_path}'") from e
    except ET.ParseError as e:
        raise Exception (f"Error: Failed to parse XML from '{file_path}'") from e

    return root


def get_pvstate_values(root: ET.Element) -> dict[dict[...]]:
    pvstate_values = {}

    for pvstate_value in root.iter('PVStateValue'):
        key = pvstate_value.get('key')
        value = pvstate_value.get('value')

        # Initialize a dictionary to store the elements
        elements = {}

        # Process 'IndexedValue' elements
        indexed_values = []
        for indexed_value in pvstate_value.findall('.//IndexedValue'):
            index = indexed_value.get('index')
            indexed_values.append(
                {'index': index, 'value': indexed_value.get('value'), 'description': indexed_value.get('description')})
        if indexed_values:
            elements['IndexedValue'] = indexed_values

        # Process 'SubindexedValues' elements
        subindexed_values = {}
        for subindexed_values_elem in pvstate_value.findall('.//SubindexedValues'):
            index = subindexed_values_elem.get('index')
            sub_values = []
            for subindexed_value in subindexed_values_elem.findall('.//SubindexedValue'):
                subindex = subindexed_value.get('subindex')
                sub_values.append({'subindex': subindex, 'value': subindexed_value.get('value'),
                                   'description': subindexed_value.get('description')})
            subindexed_values[index] = sub_values
        if subindexed_values:
            elements['SubindexedValues'] = subindexed_values

        # Store the main value and the elements in the dictionary
        pvstate_values[key] = {
            'value': value,
            'elements': elements
        }

    # Extract from AcquisitionData
    flim_time_bin = root.find(".//AcquisitionData").get("FLIMTimeBinNanoseconds")

    # Extract Sequence attributes
    sequence = root.find(".//Sequence")
    sequence_data = {
        "type": sequence.get("type"),
        "cycle": sequence.get("cycle"),
        "time": sequence.get("time"),
        "xYStageGridNumXPositions": sequence.get("xYStageGridNumXPositions"),
        "xYStageGridNumYPositions": sequence.get("xYStageGridNumYPositions"),
        "xYStageGridOverlapPercentage": sequence.get("xYStageGridOverlapPercentage"),
        "xYStageGridXOverlap": sequence.get("xYStageGridXOverlap"),
        "xYStageGridYOverlap": sequence.get("xYStageGridYOverlap")
    }

    # Extract Frame attributes
    frame = root.find(".//Frame")
    frame_data = {
        "relativeTime": frame.get("relativeTime"),
        "absoluteTime": frame.get("absoluteTime"),
        "index": frame.get("index"),
        "parameterSet": frame.get("parameterSet")
    }

    # Extract File information
    files = []
    for file in frame.findall(".//File"):
        file_data = {
            "channel": file.get("channel"),
            "channelName": file.get("channelName"),
            "filename": file.get("filename")
        }
        files.append(file_data)

    # Extract ExtraParameters
    extra_params = root.find(".//ExtraParameters")
    extra_params_data = {
        "lastGoodFrame": extra_params.get("lastGoodFrame")
    }

    # Print out the extracted data
    print("FLIMTimeBinNanoseconds:", flim_time_bin)
    print("Sequence Data:", sequence_data)
    print("Frame Data:", frame_data)
    print("Files:")
    for file in files:
        print(file)
    print("Extra Parameters:", extra_params_data)

    return pvstate_values

with importlib.resources.open_text('hsdfmpm.data', 'transfer_function.csv') as f:
    df = pd.read_csv(f)
df['startDate'] = pd.to_datetime(df['startDate'])
df['endDate'] = pd.to_datetime(df['endDate'])

class LaserFlag(Enum):
    UPRIGHT1 = "Upright1"
    U1 = "Upright1"  # Alias for UPRIGHT1
    InsightX3 = "Upright1"
    UPRIGHT2 = "Upright1"
    U2 = "Upright1"  # Alias for UPRIGHT1
    INVERTED1 = "Inverted1"
    I1 = "Inverted1"  # Alias for INVERTED1

def get_transfer_function(
        date: datetime.datetime, laser_flag: Union[str, LaserFlag]
) -> Callable[[ndarray, float, Union[float, list[float]], Union[int, list[int]] | None], ndarray]:
    # Get the laser input
    laser_flag = LaserFlag[laser_flag]

    # Filter rows based on the date and laserFlag
    valid_rows = df[(df['startDate'] <= date) & ~(df['endDate'] < date) & (df['laserFlag'] == laser_flag.value)]

    if not valid_rows.empty:
        # Get the last (mot recent) valid row (should be only one anyway)
        row = valid_rows.iloc[-1]
        offsets = np.array([val for key, val in row.items() if 'offset' in key])
        params = np.array([val for key, val in row.items() if 'param' in key])
    else:
        raise ValueError("Error: Could not get correct date or laserFlag. Please refer to the documentation.")

    def transfer(img: np.ndarray, pwr: float, gain: Union[float, list[float]], pmt: Optional[Union[int, list[int]]] = None) -> np.ndarray:
        """
        This function applies the appropriate adjustments to an input image of either 1 channel (shape: H x W). In this
         mode, a specific Index must be specified to tell the funciton which PMT offset to use. Alternatively, the image
         can have N channels (shape: N x H x W) where N is equal to the number of PMT offsets (typically, 4).

        :param gain: The PMT gain setting(s) of the image.
        :type gain: Union[float, list[float]]
        :param pwr: The calculated power at the objective for the image.
        :type pwr: float
        :param img: Image array to be transformed.
        :type img: np.ndarray
        :param pmt: Optional index of PMT offset to use. Required if img is 3D (channels != 0)
        :type pmt: int
        :return: Transformed image
        :rtype: np.ndarray
        """
        # Reshape the image to N x H x W,even if N = 1
        shape = img.shape
        img = img.reshape((-1, *shape[-2:]))

        # Check the size/pmt-index/gain match (see docstring)
        if img.shape[0] == 1 and pmt is None or img.shape[0] != len(offsets):
            raise RuntimeError(f'No PMT index given for image with {img.shape[0]} channels.')
        if len(gain) != img.shape[0]:
            raise RuntimeError(f'Ambiguous PMT gain list if length {len(gain)} for image with {img.shape[0]} channels.')

        # Prepare for matops
        gain = iterable_array(gain).reshape(-1, 1, 1)
        pmt = iterable_array(pmt) if pmt is not None else np.ones_like(gain, dtype=int)
        offset = offsets.reshape((-1, 1, 1))

        # Calculate transfer function
        g = params[0] * gain ** params[1]

        img = ((img - offset[pmt]) / (pwr ** 2)) / g

        return img

    return transfer

def polar_from_cartesian(x, y):
    return np.arctan(y, x), np.hypot(x, y)

def cartesian_from_polar(theta, r):
    return r * np.cos(theta), r * np.sin(theta)
