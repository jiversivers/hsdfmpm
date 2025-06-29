from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Callable

import numpy as np

from ..utils import LaserFlag, TRANSFER_FUNCTION_DATA
from ...utils import iterable_array, ensure_path


def get_transfer_function(
    date: datetime, laser_flag: Union[str, LaserFlag]
) -> Callable[
    [np.ndarray, float, Union[float, list[float]], Union[int, list[int]] | None],
    np.ndarray,
]:
    # Get the laser input
    laser_flag = LaserFlag[laser_flag]

    # Filter rows based on the date and laserFlag
    valid_rows = TRANSFER_FUNCTION_DATA[
        ~(TRANSFER_FUNCTION_DATA["startDate"] > date)
        & ~(TRANSFER_FUNCTION_DATA["endDate"] < date)
        & (TRANSFER_FUNCTION_DATA["laserFlag"] == laser_flag.value)
    ]

    if not valid_rows.empty:
        # Get the last (mot recent) valid row (should be only one anyway)
        row = valid_rows.iloc[-1]
        offsets = np.array([val for key, val in row.items() if "offset" in key])
        params = np.array([val for key, val in row.items() if "param" in key])
    else:
        raise ValueError(
            "Error: Could not get correct date or laserFlag. Please refer to the documentation."
        )

    def transfer(
        img: np.ndarray,
        pwr: float,
        gain: Union[float, list[float]],
        pmt: Optional[Union[int, list[int]]] = None,
    ) -> np.ndarray:
        """
        This function applies the appropriate adjustments to an input image of either 1 channel (shape: H x W). In this
         mode, a specific Index must be specified to tell the function which PMT offset to use. Alternatively, the image
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

        # Prepare for matops
        gain = iterable_array(gain)
        pmt = iterable_array(pmt) if pmt is not None else np.arange(img.shape[0])
        pmt_mask = np.array([i in pmt for i in range(len(offsets))])
        offset = offsets[pmt_mask].reshape((-1, 1, 1))
        gain = gain[pmt_mask].reshape((-1, 1, 1))

        # Check the size/pmt-index/gain match (see docstring)
        if img.shape[0] == 1 and pmt is None or img.shape[0] != len(offset):
            raise RuntimeError(
                f"No PMT index given for image with {img.shape[0]} channels."
            )
        if len(gain) != img.shape[0]:
            raise RuntimeError(
                f"Ambiguous PMT gain list if length {len(gain)} for image with {img.shape[0]} channels."
            )

        # Calculate transfer function
        g = params[0] * gain ** params[1]

        img = ((img - offset) / (pwr**2)) / g

        return img

    return transfer


def find_dated_power_file(date: str | datetime, power_directory: str | Path) -> Path:
    if isinstance(date, datetime):
        date = datetime.strftime(date, "%m%d%Y")
    power_directory = ensure_path(power_directory)
    power_files = sorted(power_directory.glob(f"*{date}*.xls*")) + sorted(
        power_directory.glob(f"*{date}*.csv")
    )
    if not power_files:
        raise FileNotFoundError(f"No power file found for date {date}.")
    return power_files[0]
