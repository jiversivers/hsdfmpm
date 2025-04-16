import json
import operator
import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Any, Union, Annotated, overload
from urllib.parse import urlparse

import cv2
import numpy as np
from pydantic import BaseModel, model_validator, field_validator, Field, computed_field, AfterValidator
from tqdm.contrib import itertools

DATA_PATH = Path.home() / ".hsdfmpm"
DATA_PATH.mkdir(exist_ok=True, parents=True)

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
def channel_check(channels: tuple[int]) -> tuple[int]: ...
@overload
def channel_check(channels: Union[int, list[int]]) -> list[int]: ...

def channel_check(channels: Union[int, list[int], tuple[int]]) -> Union[tuple[int], list[int]]:
    channels = [channels] if isinstance(channels, int) else channels
    if np.any([ch > 3 for ch in channels]):
        raise IndexError('Channel index out of range.')
    return channels

def forward_arithmetics(attr):
    ops = {
        '__add__': operator.add,
        '__sub__': operator.sub,
        '__mul__': operator.mul,
        '__truediv__': operator.truediv,
        '__floordiv__': operator.floordiv,
        '__mod__': operator.mod,
        '__pow__': operator.pow,
        '__neg__': operator.neg,
        '__pos__': operator.pos,
        '__abs__': operator.abs,
        '__radd__': operator.add,
        '__rsub__': operator.sub,
        '__rmul__': operator.mul,
        '__rtruediv__': operator.truediv,
        '__rfloordiv__': operator.floordiv,
        '__rmod__': operator.mod,
        '__rpow__': operator.pow,
    }

    def wrapper(cls):
        for name, op in ops.items():
            def make_method(op, name):
                if name.startswith('__r'):  # right-hand operations
                    def method(self, other):
                        return cls(op(other, getattr(self, attr)))
                elif name.startswith('__') and name.endswith('__'):  # unary or left-hand
                    def method(self, other=None):
                        result = op(getattr(self, attr), other) if other is not None else op(getattr(self, attr))
                        return cls(result) if not isinstance(result, cls) else result
                return method

            setattr(cls, name, make_method(op, name))
        return cls

    return wrapper

@forward_arithmetics('normalized')
class ImageData(BaseModel):
    """
    This class holds several useful methods for loading and (pre-)processing HSDFM image cubes. To maintain
    computational efficiency, methods that return a new array are cached.

    :param image_path: str; The directory path of the HSDFM image cube and JSON metadata file.
    """
    image_path: Annotated[Union[str, Path], AfterValidator(is_dir)]
    metadata_ext: str
    metadata_path: Annotated[Optional[Union[str, Path]], AfterValidator(is_file)] = None
    image_ext: str = '.tiff'
    channels: Annotated[Union[int, list[int], tuple[int]], AfterValidator(channel_check)] = (0, 1, 2, 3)
    _hyperstack: Optional[np.ndarray] = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    @computed_field
    @property
    def hyperstack(self) -> np.ndarray:
        if self._hyperstack is None:
            self._hyperstack = read_hyperstack(self.image_path, self.image_ext)
        # Select out channels
        hyperstack = np.stack([self._hyperstack[ch] for ch in self.channels], axis=0)
        return hyperstack

    def __getitem__(self, item: int) -> np.ndarray:
        if item > len(self):
            raise IndexError(f'Index {item} out of range for ImageData with length {len(self)}')
        return self.hyperstack[item]

    def __len__(self) -> int:
        if self.hyperstack.ndim < 3:
            return 1
        else:
            return self.hyperstack.shape[-1]

    def bin(self, bin_factor: int = 4):
        bands, h, w = self.hyperstack.shape
        h_binned = h // bin_factor
        w_binned = w // bin_factor

        # Crop to make divisible
        cube_cropped = self.hyperstack[:, :h_binned * bin_factor, :w_binned * bin_factor]

        # Reshape and bin
        cube_reshaped = cube_cropped.reshape(
            bands,
            h_binned, bin_factor,
            w_binned, bin_factor
        )

        # Average over binning axes
        self._hyperstack = cube_reshaped.mean(axis=(2, 4))

    def apply_kernel_bank(self, kernel_bank):
        self.filtered = np.nanmax([
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
