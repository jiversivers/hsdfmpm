from functools import cache
from .utils import *
from ..utils import ImageData, read_hyperstack


class HyperspectralImage(ImageData):
    """
    This class holds several useful methods for loading and (pre-)processing HSDFM image cubes. To maintain
    computational efficiency, methods that return a new array are cached.

    :param image_path: str; The directory path of the HSDFM image cube and JSON metadata file.
    """
    model: ModelType = 'monte_carlo'
    metadata_ext: str = 'metadata.json'
    image_ext: str = '.tif'

    def __post_init__(self):
        # Load image metadata
        self.metadata = read_metadata_json(self.image_path)

        # Load hyperstack
        self.hyperstack = read_hyperstack(self.image_path)

        # Store raw instance for back reference after mutating
        self.raw = self.hyperstack.copy()

    def normalize_integration_time(self):
        self.hyperstack = normalize_integration_time(self, self.metadata['ExpTime'])

    def normalize_to_standard(self):
        if not (hasattr(self, 'standard') and hasattr(self.standard, 'bg')):
            raise ValueError(f'"{self.image_path}" is missing "standard" and/or "background" attributes.\n'
                             f'Set them and try again.')
        self.hyperstack = normalize_to_standard(self, self.standard, self.bg)

    @cache
    def fit(self):
        pass

    # def k_cluster

