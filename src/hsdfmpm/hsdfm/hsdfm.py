from functools import cache
from typing import Optional

from pydantic import model_validator, BaseModel, SkipValidation

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
    scalar: Union[float, np.ndarray[float]] = 1

    @model_validator(mode='after')
    def load_data(self):
        metadata_path = list(self.image_path.glob(f'*{self.metadata_ext}'))[0]

        # Load image metadata
        self.metadata = read_metadata_json(metadata_path)

        # Load hyperstack
        hyperstack = read_hyperstack(img_dir=self.image_path, ext=self.image_ext)
        self._hyperstack = hyperstack if self.scalar is None else hyperstack / self.scalar
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

    @cache
    def fit(self, model: ModelType = 'monte_carlo', **kwargs):
        pass

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
    scalar: float = 1

    @model_validator(mode='after')
    def load_data(self):
        self.listed_hyperstack = []
        for image_path in self.image_paths:
            kwargs = dict(image_path=image_path,
                          image_ext=self.image_ext,
                          metadata_ext=self.metadata_ext,
                          channels=self.channels,
                          scalar=self.scalar)
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


