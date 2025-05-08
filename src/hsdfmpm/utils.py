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
from matplotlib.colors import Colormap, Normalize
from matplotlib import cm
from matplotlib.colors import ListedColormap, Colormap, LinearSegmentedColormap
from pydantic import BaseModel, Field, computed_field, AfterValidator, model_validator
import itertools

DATA_PATH = Path.home() / ".hsdfmpm"
DATA_PATH.mkdir(exist_ok=True, parents=True)


# Path helpers
def ensure_path(path_like: Union[str, Path]) -> Path:
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        if path_like.startswith("file://"):
            return Path(urlparse(path_like).path)
        return Path(path_like)
    if isinstance(path_like, NoneType):
        warnings.warn(
            "'ensure_path' got NoneType, but expected str or Path object.",
            UserWarning,
            stacklevel=2,
        )
        return None
    raise TypeError(f"Cannot convert {type(path_like)} to Path")


def is_file(file_path: str) -> Path:
    file_path = ensure_path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    print("File passed")
    return file_path


def is_dir(file_path: str) -> Path:
    file_path = ensure_path(file_path)
    if not file_path.is_dir():
        raise NotADirectoryError(file_path)
    return file_path


# Helper for dtype conversion
def iterable_array(x: Any) -> Iterable[Any]:
    x = [x] if not isinstance(x, Iterable) else x
    return np.array(x)


# Helper function to handle reshaping input images
def vectorize_img(img: np.ndarray, include_location: bool = False) -> np.ndarray:
    if img.ndim <= 2:
        shape = img.shape
        X = img.reshape(-1, 1)
    elif img.ndim == 3:
        shape = img.shape[1:]
        X = img.reshape(img.shape[0], -1).T
    if include_location:
        x, y = np.meshgrid(
            np.linspace(-0.5, 0.5, shape[0]) * shape[0],
            np.linspace(-0.5, 0.5, shape[1]) * shape[1],
        )
        X = np.column_stack([x.ravel(), y.ravel(), X])
    return X


# Image reader
def read_hyperstack(img_dir: str, ext: str = ".tif") -> np.ndarray[float]:
    hs = []
    for img_path in Path(img_dir).glob(f"*{ext}"):
        hs.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
    if not img_path:
        raise FileNotFoundError(f"No image files found in {img_dir}.")
    hs = np.stack(hs, dtype=np.float64)
    while hs.ndim < 3:
        hs = np.expand_dims(hs, axis=0)
    return hs


def apply_kernel_bank(
    src: np.ndarray[float], bank: np.ndarray[float]
) -> np.ndarray[float]:
    return np.nanmax(
        [cv2.filter2D(src, -1, k) for (k,) in itertools.product(bank)], axis=0
    )


def truncate_colormap(
    cmap: Optional[Union[Colormap, np.ndarray[float], str]] = None,
    cmin: float = 0,
    cmax: float = 1,
    n: int = 100,
):
    cmap = get_cmap(cmap)
    new_colors = cmap(np.linspace(cmin, cmax, n))
    return LinearSegmentedColormap.from_list("truncated_cmap", new_colors, N=n)


def get_cmap(cmap: Optional[Union[Colormap, np.ndarray[float], str]] = None):
    if not isinstance(cmap, Colormap):
        if isinstance(cmap, np.ndarray):
            cmap = ListedColormap(cmap)
        elif isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        elif cmap is None:
            cmap = cm.get_cmap("jet")
        else:
            raise TypeError(
                f"cmap must be either a Colormap, str, or None to default to 'jet', but got type {type(cmap)}"
            )
    return cmap


def colorize(
    var_map: np.ndarray[float],
    intensity: Optional[np.ndarray[float]] = None,
    cmap: Optional[Union[Colormap, np.ndarray]] = None,
    cmin: float = 0.0,
    cmax: float = 1.0,
    n: int = 100,
    intmin: float = 0.1,
    intmax: float = 0.95,
) -> tuple[np.ndarray[float], Colormap]:

    if intensity is None:
        intensity = np.ones_like(var_map)

    if cmap is None:
        # Default cmap if none is input
        cmap = truncate_colormap("jet", 0.13, 0.88, n)
    else:
        # Validate input
        cmap = get_cmap(cmap)

    # Map colors with normed space (cmin, cmax)
    norm = Normalize(vmin=cmin, vmax=cmax, clip=True)
    colorized = cmap(norm(var_map))[:, :, :3]

    # Normalize intensity within limits
    intmin = np.percentile(intensity, intmin * 100)
    intmax = np.percentile(intensity, intmax * 100)
    intensity = (intensity - intmin) / (intmax - intmin)
    intensity[intensity < 0] = 0
    intensity[intensity > 1] = 1

    return colorized * intensity[..., np.newaxis], cmap


# Define the class decorator
def add_arithmetic_methods(cls):
    # Mapping of dunder methods to their respective operator functions
    methods = {
        "__add__": operator.add,
        "__sub__": operator.sub,
        "__mul__": operator.mul,
        "__truediv__": operator.truediv,
        "__floordiv__": operator.floordiv,
        "__mod__": operator.mod,
        "__pow__": operator.pow,
        "__lt__": operator.lt,
        "__le__": operator.le,
        "__gt__": operator.gt,
        "__ge__": operator.ge,
        "__eq__": operator.eq,
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


@overload
def channel_check(channels: NoneType) -> NoneType: ...
@overload
def channel_check(channels: tuple[int]) -> tuple[int]: ...
@overload
def channel_check(channels: Union[int, list[int]]) -> list[int]: ...


def channel_check(
    channels: Union[int, list[int], tuple[int]],
) -> Union[tuple[int], list[int]]:
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
    image_ext: str = ".tiff"
    metadata_ext: str
    metadata_path: Annotated[Optional[Union[str, Path]], AfterValidator(is_file)] = None
    metadata: Optional[dict] = None
    channels: Annotated[
        Optional[Union[int, list[int], tuple[int]]], AfterValidator(channel_check)
    ] = None
    _hyperstack: Optional[np.ndarray] = None
    _active: Optional[np.ndarray] = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    @computed_field
    @property
    def hyperstack(self) -> np.ndarray:
        if self._hyperstack is None:
            self._hyperstack = read_hyperstack(self.image_path, self.image_ext)
        # Select out channels
        hyperstack = (
            self._hyperstack.copy()
        )  # So channels can be selected after initialization
        if self.channels is not None:
            hyperstack = np.stack(
                [self._hyperstack[ch] for ch in self.channels], axis=0
            )
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
        self._hyperstack = None
        self._active = self.hyperstack

    def __getitem__(self, item: int) -> np.ndarray:
        return self._active[item]

    def __len__(self) -> int:
        if self._active.ndim < 3:
            return 1
        else:
            return self._active.shape[-1]

    @computed_field
    @property
    def shape(self) -> tuple[int, int, int]:
        return self._active.shape

    def __array__(self, dtype=None, copy=None):
        return self._active.astype(dtype, copy=copy)

    def bin(self, bin_factor: int = 4):
        bands, h, w = self.shape
        h_binned = h // bin_factor
        w_binned = w // bin_factor

        # Crop to make divisible
        cube_cropped = self._active[:, : h_binned * bin_factor, : w_binned * bin_factor]

        # Reshape and bin
        cube_reshaped = cube_cropped.reshape(
            bands, h_binned, bin_factor, w_binned, bin_factor
        )

        # Average over binning axes
        self._active = cube_reshaped.mean(axis=(2, 4))

    def resize_to(self, h: int):
        self.bin(bin_factor=self.shape[1] // h)

    def apply_kernel_bank(self, kernel_bank: np.ndarray) -> np.ndarray:
        return apply_kernel_bank(self, kernel)

    def apply_mask(self, mask):
        self._active[:, ~mask.astype(bool)] = np.nan

    def write(self, filename: str = "processed.tiff", **kwargs):
        filename = ensure_path(filename)
        cv2.imwrite(str(filename), self._active, **kwargs)


class SerializableModel(BaseModel):
    __version__ = "0.0.1"
    model_version: str = Field(default=__version__, exclude=True)

    def save_json(self, path: Union[str, Path], **kwargs):
        path = ensure_path(path)
        data = self.model_dump()
        data["model_version"] = self.__version__
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
                f"Loaded version {version} of {cls.__class__.__name__}", stacklevel=2
            )
