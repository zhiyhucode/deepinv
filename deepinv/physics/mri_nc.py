import numpy as np
import torch
from deepinv.physics.forward import LinearPhysics

try:
    from mrinufft import get_operator
    MRINUFFT_AVAILABLE = True
except ImportError:
    MRINUFFT_AVAILABLE = False


class MRI_NC(LinearPhysics):
    """MRI Non-Cartesian Multicoil operator.

    """

    def __init__(self, kspace_trajectory, shape, n_coils, smaps=None, density=True, backend="cufinufft", **kwargs):
        super().__init__(**kwargs)
        if MRINUFFT_AVAILABLE is False:
            raise RuntimeError("mri-nufft is not installed.")

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(kspace_trajectory, shape, density=density, n_coils=n_coils, smaps=smaps, keep_dims=True)

    def A(self, x):
        if x.dtype != torch.complex64:
            x = x.to(torch.complex64)
        x_np = np.complex64(x.cpu().numpy())
        y_np = self._operator.op(x_np)
        return torch.from_numpy(y_np).type(x.type())

    def A_adjoint(self, y):
        y_np = np.complex64(y.cpu().numpy())
        x_np = self._operator.adj_op(y_np)
        return torch.from_numpy(x_np).type(y.type())
