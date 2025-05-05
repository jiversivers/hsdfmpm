import unittest
from unittest.mock import patch
from hsdfmpm.hsdfm.contrib.mie import (
    mie_scattering_coefficients,
    mie_efficiency,
    mie_scatter_and_anisotropy,
    compute_volume_distribution_by_scattering,
    compute_volume_distribution_by_bead_number,
    generate_phantom_profiles
)
import numpy as np


if __name__ == '__main__':
    unittest.main()
