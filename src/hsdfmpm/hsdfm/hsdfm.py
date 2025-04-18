from functools import cache, cached_property
from pathlib import Path
from typing import Union, Optional

import numpy as np
from pydantic import model_validator, field_validator, BaseModel, computed_field

from .utils import *
from ..utils import ImageData, read_hyperstack, ensure_path


class HyperspectralImage(ImageData):
    """
    This class holds several useful methods for loading and (pre-)processing HSDFM image cubes. To maintain
    computational efficiency, methods that return a new array are cached.

    :param image_path: str; The directory path of the HSDFM image cube and JSON metadata file.
    """
    model: ModelType = 'monte_carlo'
    metadata_ext: str = 'metadata.json'
    image_ext: str = '.tiff'

    @model_validator(mode='after')
    def load_data(self):
        metadata_path = list(self.image_path.glob(f'*{self.metadata_ext}'))[0]

        # Load image metadata
        self.metadata = read_metadata_json(metadata_path)

        # Load hyperstack
        self._hyperstack = read_hyperstack(img_dir=self.image_path, ext=self.image_ext)

        # Store raw instance for back reference after mutating
        self.raw = self._hyperstack.copy()
        return self

    def normalize_integration_time(self):
        self._hyperstack = normalize_integration_time(self, self.metadata['ExpTime'])
        self._int_normed = True

    def normalize_to_standard(self):
        if not (hasattr(self, 'standard') and hasattr(self.standard, 'bg')):
            raise ValueError(f'"{self.image_path}" is missing "standard" and/or "background" attributes.\n'
                             f'Set them and try again.')
        self._hyperstack = normalize_to_standard(self, self.standard, self.bg)
        self._standard_normed = True

    def normalize(self):
        if not self._int_normed:
            self.normalize_integration_time()
        if not self._standard_normed:
            self.normalize_to_standard()
        return self._hyperstack

    @cache
    def fit(self):
        pass

    # def k_cluster

class MergedHyperspectralImage(ImageData):
    """
    This class is effectively a list of HyperspectralImages with easy, built-in iteration over that list, and
    implicit merging when accessed, so it can be easily used as a HyperspectralImage object for processing.
    """
    image_path: list[Union[str, Path]]
    metadata_ext: str = 'metadata.json'
    listed_hyperstack: Optional[list[HyperspectralImage]] = None

    @model_validator(mode='after')
    def load_data(self):
        self.listed_hyperstack = []
        for image_path in self.image_path:
            kwargs = dict(image_path=image_path,
                          image_ext=self.image_ext,
                          metadata_ext=self.metadata_ext,
                          channels=self.channels)
            if self.metadata_path is not None:
                kwargs['metadata_path'] = self.metadata_path
            self.listed_hyperstack.append(HyperspectralImage(**kwargs))

        return self

    @computed_field
    @property
    def hyperstack(self) -> np.ndarray:
        if self._hyperstack is None:
            self._hyperstack = np.mean([image.hyperstack for image in self.listed_hyperstack], axis=0)
        return self._hyperstack

    @computed_field
    @cached_property
    def raw(self) -> np.ndarray:
        # Get initial merged hyperstack for back reference after mutation
        return np.mean([image.hyperstack for image in self.listed_hyperstack], axis=0)

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
            return [getattr(member, attr) for member in self.listed_hyperstack]

    def __getitem__(self, item):
        return self.listed_hyperstack[item]

    def __len__(self):
        return len(self.listed_hyperstack)

    def __iter__(self):
        return iter(self.listed_hyperstack)


