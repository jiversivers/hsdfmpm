import importlib.resources
import xml.etree.ElementTree as ET
from enum import Enum

import pandas as pd


def read_xml(file_path: str) -> pd.DataFrame:
    """Reads and parses an XML file, then prints the root element's tag and text of all elements."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

    except FileNotFoundError as e:
        raise Exception(f"Error: File not found at '{file_path}'") from e
    except ET.ParseError as e:
        raise Exception(f"Error: Failed to parse XML from '{file_path}'") from e

    return root


def get_pvstate_values(root: ET.Element) -> dict[dict[...]]:
    pvstate_values = {}

    for pvstate_value in root.iter("PVStateValue"):
        key = pvstate_value.get("key")
        value = pvstate_value.get("value")

        # Initialize a dictionary to store the elements
        elements = {}

        # Process 'IndexedValue' elements
        indexed_values = []
        for indexed_value in pvstate_value.findall(".//IndexedValue"):
            index = indexed_value.get("index")
            indexed_values.append(
                {
                    "index": index,
                    "value": indexed_value.get("value"),
                    "description": indexed_value.get("description"),
                }
            )
        if indexed_values:
            elements["IndexedValue"] = indexed_values

        # Process 'SubindexedValues' elements
        subindexed_values = {}
        for subindexed_values_elem in pvstate_value.findall(".//SubindexedValues"):
            index = subindexed_values_elem.get("index")
            sub_values = []
            for subindexed_value in subindexed_values_elem.findall(
                ".//SubindexedValue"
            ):
                subindex = subindexed_value.get("subindex")
                sub_values.append(
                    {
                        "subindex": subindex,
                        "value": subindexed_value.get("value"),
                        "description": subindexed_value.get("description"),
                    }
                )
            subindexed_values[index] = sub_values
        if subindexed_values:
            elements["SubindexedValues"] = subindexed_values

        # Store the main value and the elements in the dictionary
        pvstate_values[key] = {"value": value, "elements": elements}

    # Extract from AcquisitionData
    for flim_value in root.iter("AcquisitionData"):
        keys = flim_value.keys()
        for key in keys:
            pvstate_values[key] = flim_value.get(key)

    return pvstate_values


# Load transfer function ndata into dataframe
with importlib.resources.open_text("hsdfmpm.data", "transfer_function.csv") as f:
    TRANSFER_FUNCTION_DATA = pd.read_csv(f)
TRANSFER_FUNCTION_DATA["startDate"] = pd.to_datetime(
    TRANSFER_FUNCTION_DATA["startDate"]
)
TRANSFER_FUNCTION_DATA["endDate"] = pd.to_datetime(TRANSFER_FUNCTION_DATA["endDate"])


class LaserFlag(Enum):
    UPRIGHT1 = "Upright1"
    U1 = "Upright1"  # Alias for UPRIGHT1
    InsightX3 = "Upright1"
    UPRIGHT2 = "Upright1"
    U2 = "Upright1"  # Alias for UPRIGHT1
    INVERTED1 = "Inverted1"
    I1 = "Inverted1"  # Alias for INVERTED1
