import unittest

import numpy as np
import numpy.testing as npt
from hsdfmpm.hsdfm.fit import residual, jacobian, fit_voxel, volume_iter, fit_volume, make_jacobian, make_residual, \
    LossIsGoodEnough

N_WAVEL = 10
N_PARAM = 3

rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(rng.normal(size=(N_WAVEL, N_PARAM)))
BASIS = Q

def toy_model(*params):
    return BASIS @ np.array(params)

class TestFitFunctions(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(10)
        self.params = np.array([1, 2, 3], dtype=np.float32)
        self.model = toy_model
        self.y = self.model(*self.params)

    def test_residual(self):
        res = self.model(*self.params + 1) - self.y
        npt.assert_array_equal(residual(self.params + 1, self.y, model=self.model), res)

    def test_make_residual(self):
        res_fun = make_residual(voxel=self.y, model=self.model, loss_thresh=1e-3)
        npt.assert_array_equal(res_fun(self.params + 1), residual(self.params + 1, self.y, model=self.model, loss_thresh=1e-3))
        self.assertRaises(LossIsGoodEnough, res_fun, self.params)

    def test_jacobian(self):
        eps = 1e-4
        npt.assert_allclose(jacobian(self.params, self.y, eps=eps, model=self.model), BASIS, rtol=1e-2, atol=1e-2)

    def test_make_jacobian(self):
        jac_fun = make_jacobian(voxel=self.y, model=self.model, eps=1e-4)
        npt.assert_array_equal(jac_fun(self.params + 1), jacobian(self.params + 1, self.y, eps=1e-4, model=self.model))

    def test_volume_iter(self):
        vol = np.arange(0, 20, 0.5).reshape(2, 5, 4)
        vol[0, 1, 2] = np.nan
        vol[1, 2, 3] = np.nan
        i = 0
        for x, y, z in volume_iter(vol):
            self.assertTrue(np.all([x, y] != [1, 2]))
            self.assertTrue(np.all([x, y] != [2, 3]))
            i += 1
        self.assertEqual(i, 18)

    def test_fit_voxel(self):
        params = fit_voxel(
            voxel = self.y,
            model = self.model,
            x0 = [0, 0, 0],
            loss_thresh=1e-3)
        npt.assert_allclose(params, self.params, rtol=1e-7, atol=1e-9)

    def test_fit_volume(self):
        volume = np.column_stack([self.y, self.y])
        volume = np.stack([volume, volume], axis=-1)
        params = fit_volume(
            volume = volume,
            model = self.model,
            n_workers = 2,
            x0 = [0, 0, 0]
        )
        true_params = np.column_stack([self.params, self.params])
        true_params = np.stack([true_params, true_params], axis=-1)
        npt.assert_allclose(params, true_params, rtol=1e-7, atol=1e-9)