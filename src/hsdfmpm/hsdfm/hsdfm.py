from functools import cache
from typing import Optional

from pydantic import model_validator, BaseModel, SkipValidation
from tqdm.contrib import itertools
import scipy.optimize as opt

from .utils import *
from ..utils import ImageData, read_hyperstack, add_arithmetic_methods


class HyperspectralImage(ImageData):
    """
    This class holds several useful methods for loading and (pre-)processing HSDFM image cubes. To maintain
    computational efficiency, methods that return a new array are cached.

    :param image_path: str; The directory path of the HSDFM image cube and JSON metadata file.
    """
    model: ModelType = 'monte_carlo'
    metadata_ext: str = 'metadata.json'
    image_ext: str = '.tiff'
    standard: Optional[SkipValidation[Union['MergedHyperspectralImage', 'HyperspectralImage']]] = None
    background: Optional[SkipValidation[Union['MergedHyperspectralImage', 'HyperspectralImage']]] = None
    wavelengths: Optional[list[float]] = None
    scalar: Union[float, np.ndarray[float]] = 1

    @model_validator(mode='after')
    def load_data(self):
        metadata_path = list(self.image_path.glob(f'*{self.metadata_ext}'))[0]

        # Load image metadata
        self.metadata = read_metadata_json(metadata_path).copy()

        # Load hyperstack
        hyperstack = read_hyperstack(img_dir=self.image_path, ext=self.image_ext)
        self._hyperstack = hyperstack.copy()
        try:
            self._hyperstack /= self.scalar
            scale_after_masking = False
        except ValueError:
            scale_after_masking = True

        # Apply wavelength selection to image and metadata
        if self.wavelengths is not None:
            mask = np.isin(self.metadata['Wavelength'], self.wavelengths)
            self._hyperstack = self._hyperstack[mask]
            for key in self.metadata.keys():
                self.metadata[key] = [self.metadata[key][i] for i, is_in_mask in enumerate(mask) if is_in_mask]
        self._hyperstack /= self.scalar if scale_after_masking else 1
        self._active = self.hyperstack
        return self


    def normalize_integration_time(self):
        self._active = normalize_integration_time(self, self.metadata['ExpTime'])

    def normalize_to_standard(self):
        if self.standard is None or self.background is None:
            raise ValueError(f'"{self.image_path}" is missing "standard" and/or "background" attributes.\n'
                             f'Set them and try again.')
        self._active = normalize_to_standard(self, self.standard, self.background)

    def normalize(self):
        self.normalize_integration_time()
        if self.standard is not None and self.background is not None:
            self.normalize_to_standard()

    def get(self, wavelength: float) -> np.ndarray[float]:
        """Method to streamline selection of specific wavelengths from the image stack."""
        for wl in self.metadata['Wavelength']:
            if wl == wavelength:
                return self[self.metadata['Wavelength'].index(wl)]

    @cache
    def fit(self, model: Callable[[float, float, ...], np.ndarray[float]],
            guess: list[float], bounds: list[tuple[float, float]]) -> tuple[np.ndarray[float], np.ndarray[bool]]:

        # Get image shape
        image_shape = self.shape

        # Mask nans
        nan_mask = np.logical_or(np.isnan(self), np.isinf(self))

        # Create 0 output array
        out_image = np.zeros((len(bounds),) + image_shape[1:])
        params = guess
        for i, j in itertools.product(range(image_shape[1]), range(image_shape[2])):
            # Get current voxel
            r = self[:, i, j]
            mask = nan_mask[:, i, j]

            # Remove nan values
            if np.any(mask):
                r = r[~mask]
                wavelengths = self.wavelengths[~mask]
            try:
                params, _ = opt.curve_fit(model, wavelengths, r, p0=params, bounds=bounds)
            except RuntimeError:
                params = [np.nan] * 4

            # Store fitted out
            out_image[:, i, j] = params

            # Update guess to params for faster convergence
            params = guess if np.isnan(params).any() else params

        return out_image, nan_mask

    def k_cluster(self, k: Union[int, list[int]] = 2):
        if isinstance(k, int):
            k = [k]
        pass

@add_arithmetic_methods
class MergedHyperspectralImage(BaseModel):
    """
    This class is effectively a list of HyperspectralImages with easy, built-in iteration over that list, and
    implicit merging when accessed, so it can be easily used as a HyperspectralImage object for processing.
    """
    image_paths: list[Union[str, Path]]
    image_ext: str = '.tiff'
    channels: Optional[Union[int, list[int]]] = None
    metadata_path: Optional[Union[str, Path]] = None
    metadata_ext: str = 'metadata.json'
    listed_hyperstack: Optional[list[HyperspectralImage]] = None
    wavelengths: Optional[list[float]] = None
    scalar: float = 1

    @model_validator(mode='after')
    def load_data(self):
        self.listed_hyperstack = []
        for image_path in self.image_paths:
            kwargs = dict(image_path=image_path,
                          image_ext=self.image_ext,
                          metadata_ext=self.metadata_ext,
                          channels=self.channels,
                          scalar=self.scalar,
                          wavelengths=self.wavelengths)
            if self.metadata_path is not None:
                kwargs['metadata_path'] = self.metadata_path
            self.listed_hyperstack.append(HyperspectralImage(**kwargs))
        return self

    def __getattr__(self, attr):
        """Forward method calls to and get attributes from each instance of HyperspectralImage in the list."""
        first = self[0]
        val = getattr(first, attr)
        if callable(val):
            def delegated(*args, **kwargs):
                results = []
                for hs in self.listed_hyperstack:
                    method = getattr(hs, attr)
                    results.append(method(*args, **kwargs))
                return results

            return delegated
        else:
            out = [getattr(member, attr) for member in self.listed_hyperstack]
            try:
                return np.nanmean(out, axis=0)
            except TypeError:
                return out

    def __array__(self):
        return self._active

    def __getitem__(self, item):
        return self.listed_hyperstack[item]

    def __len__(self):
        return len(self.listed_hyperstack)

    def __iter__(self):
        return iter(self.listed_hyperstack)


