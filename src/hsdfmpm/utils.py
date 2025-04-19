import json
import operator
import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path
from types import NoneType
from typing import Optional, Any, Union, Annotated, overload
from urllib.parse import urlparse

import cv2
import numpy as np
from poetry.console.commands import self
from pydantic import BaseModel, model_validator, field_validator, Field, computed_field, AfterValidator
from tqdm.contrib import itertools

DATA_PATH = Path.home() / ".hsdfmpm"
DATA_PATH.mkdir(exist_ok=True, parents=True)

# Define the class decorator
def add_arithmetic_methods(cls):
    # Mapping of dunder methods to their respective operator functions
    methods = {
        '__add__': operator.add,
        '__sub__': operator.sub,
        '__mul__': operator.mul,
        '__truediv__': operator.truediv,
        '__floordiv__': operator.floordiv,
        '__mod__': operator.mod,
        '__pow__': operator.pow,
        '__lt__': operator.lt,
        '__le__': operator.le,
        '__gt__': operator.gt,
        '__ge__': operator.ge,
        '__eq__': operator.eq,
    }

    # Dynamically add methods to the class
    for name, func in methods.items():
        # Define the function dynamically
        def method(self, other, op=func):
            # Ensure `other` is the compatible type or extract its value
            if isinstance(other, self.__class__):
                other = other._active
            return op(self._active, other)

        # Attach the dynamically created method to the class
        setattr(cls, name, method)

    return cls

def is_file(file_path: str) -> Path:
    file_path = ensure_path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return file_path

def is_dir(file_path: str) -> Path:
    file_path = ensure_path(file_path)
    if not file_path.is_dir():
        raise NotADirectoryError(file_path)
    return file_path

@overload
def channel_check(channels: NoneType) -> NoneType: ...
@overload
def channel_check(channels: tuple[int]) -> tuple[int]: ...
@overload
def channel_check(channels: Union[int, list[int]]) -> list[int]: ...

def channel_check(channels: Union[int, list[int], tuple[int]]) -> Union[tuple[int], list[int]]:
    channels = [channels] if isinstance(channels, int) else channels
    return channels

@add_arithmetic_methods
class ImageData(BaseModel):
    """
    This class holds several useful methods for loading and (pre-)processing HSDFM image cubes. To maintain
    computational efficiency, methods that return a new array are cached.

    :param image_path: str; The directory path of the image cube and metadata file.
    """
    image_path: Annotated[Union[str, Path], AfterValidator(is_dir)]
    image_ext: str = '.tiff'
    metadata_ext: str
    metadata_path: Annotated[Optional[Union[str, Path]], AfterValidator(is_file)] = None
    metadata: Optional[dict] = None
    channels: Annotated[Optional[Union[int, list[int], tuple[int]]], AfterValidator(channel_check)] = None
    _hyperstack: Optional[np.ndarray] = None
    _active: Optional[np.ndarray] = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    @computed_field
    @property
    def hyperstack(self) -> np.ndarray:
        if self._hyperstack is None:
            self._hyperstack = read_hyperstack(self.image_path, self.image_ext)
        # Select out channels
        hyperstack = self._hyperstack.copy()  # So channels can be selected after initialization
        if self.channels is not None:
            hyperstack = np.stack([self._hyperstack[ch] for ch in self.channels], axis=0)

        return hyperstack

    @computed_field
    @property
    def image(self) -> np.ndarray:
        return self._active

    @computed_field
    @property
    def raw(self) -> np.ndarray:
        return self.hyperstack

    def reset(self):
        self._active = self._hyperstack

    def __getitem__(self, item: int) -> np.ndarray:
        if item > len(self):
            raise IndexError(f'Index {item} out of range for ImageData with length {len(self)}')
        return self._active[item]

    def __len__(self) -> int:
        if self._active.ndim < 3:
            return 1
        else:
            return self._active.shape[-1]

    def __array__(self):
        return self._active

    def bin(self, bin_factor: int = 4):
        bands, h, w = self._active.shape
        h_binned = h // bin_factor
        w_binned = w // bin_factor

        # Crop to make divisible
        cube_cropped = self._active[:, :h_binned * bin_factor, :w_binned * bin_factor]

        # Reshape and bin
        cube_reshaped = cube_cropped.reshape(
            bands,
            h_binned, bin_factor,
            w_binned, bin_factor
        )

        # Average over binning axes
        self._active = cube_reshaped.mean(axis=(2, 4))

    def apply_kernel_bank(self, kernel_bank):
        self._active = np.nanmax([
            cv2.filter2D(self.hyperstack, -1, k) for (k,) in itertools.product(kernel_bank)
        ], axis=0)

# Helper for dtype conversion
def iterable_array(x: Any) -> Iterable[Any]:
    x = [x] if not isinstance(x, Iterable) else x
    return np.array(x)

# Helper function to handle reshaping input images
def prepare_src(src: np.ndarray, include_location: bool = False):
    if src.ndim <= 2:
        shape = src.shape
        X = src.reshape(-1,1)
    elif src.ndim == 3:
        shape = src.shape[1:]
        X = src.reshape(src.shape[0], -1).T
    if include_location:
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, shape[0]) * shape[0], np.linspace(-0.5, 0.5, shape[1]) * shape[1])
        X = np.column_stack([x.ravel(), y.ravel(), X])
    return X

def read_hyperstack(img_dir: str, ext: str = '.tif') -> np.ndarray:
    hs = []
    for img_path in Path(img_dir).glob(f'*{ext}'):
        hs.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
    hs = np.stack(hs, dtype=np.float64)
    while hs.ndim < 3:
        hs = np.expand_dims(hs, axis=0)
    return hs

def ensure_path(path_like: Union[str, Path]) -> Path:
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        if path_like.startswith("file://"):
            return Path(urlparse(path_like).path)
        return Path(path_like)
    if isinstance(path_like, NoneType):
        warnings.warn("'ensure_path' got NoneType, but expected str or Path object.", UserWarning, stacklevel=2)
        return None
    raise TypeError(f"Cannot convert {type(path_like)} to Path")

class SerializableModel(BaseModel):
    __version__ = "0.0.1"
    model_version: str = Field(default=__version__, exclude=True)

    def save_json(self, path: Union[str, Path], **kwargs):
        path = ensure_path(path)
        data = self.model_dump()
        data['model_version'] = self.__version__
        path.write_text(json.dumps(data, **kwargs))

    @classmethod
    def load_json(cls, path: Union[str, Path]):
        path = ensure_path(path)
        payload = json.loads(path.read_text())
        cls._report_version(payload.get("model_version"))
        return cls(**{k: v for k, v in payload.items() if k != "model_version"})

    def save_pickle(self, path: Union[str, Path]):
        path = ensure_path(path)
        with path.open("wb") as f:
            pickle.dump({"model": self, "model_version": self.__version__}, f)

    @classmethod
    def load_pickle(cls, path: Union[str, Path]):
        path = ensure_path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        cls._report_version(payload.get("model_version"))
        return payload["model"]

    @classmethod
    def _report_version(cls, version):
        if version != cls.__version__:
            warnings.warn(
                f"Loaded version {version} of {cls.__class__.__name__}",
                stacklevel=2
            )

#TODO: Fix this so it actually updates the saved irf model
def autoversion(major=1, minor=0):
    """
    Decorator to auto-increment the patch version of a model on each load.
    """

    def decorator(cls):
        # Find the current patch count (based on how many times class is defined)
        patch = getattr(cls, "_version_patch", 0) + 1
        version_str = f"{major}.{minor}.{patch}"
        cls.__version__ = version_str
        cls._version_patch = patch

        # Inject version into model field default
        if "model_version" not in cls.model_fields:
            cls.model_version = version_str  # fallback if not using Field()
        return cls

    return decorator
