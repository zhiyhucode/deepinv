import numpy as np
import torch
from deepinv.physics.forward import LinearPhysics

try:
    from mrinufft import get_operator

    MRINUFFT_AVAILABLE = True
except ImportError:
    MRINUFFT_AVAILABLE = False


class MRI_NC(LinearPhysics):
    r"""
    Non-Cartesian Multicoil MRI imaging operator.

    This operator operates on batched 2D images and is defined as

    .. math::

        y = F_{\Omega}x

    where :math:`F_{\Omega}` is the 2D discrete Non-Uniform Fast-Fourier Transform (NUFFT).

    TODO: add doc on smaps, density compensation, backend...

    The images :math:`x` should be of size (B, C, H, W) and of type torch.complex64.

    :param numpy.ndarray kspace_trajectory: the k-space trajectory of shape (N, 2) where N is the number of samples.
    :param tuple shape: the shape of the image (H, W).
    :param int n_coils: the number of coils.
    :param numpy.ndarray smaps: the sensitivity maps of shape (H, W, n_coils).
    :param bool density: whether to apply density compensation.
    :param str backend: the backend to use. Can be "cufinufft" or "mri-nufft".
    """

    def __init__(
        self,
        kspace_trajectory,
        shape,
        n_coils,
        smaps=None,
        density=True,
        backend="cufinufft",
        **kwargs
    ):
        super().__init__(**kwargs)
        if MRINUFFT_AVAILABLE is False:
            raise RuntimeError("mri-nufft is not installed.")

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(
            kspace_trajectory,
            shape,
            density=density,
            n_coils=n_coils,
            smaps=smaps,
            keep_dims=True,
        )

    def A(self, x):
        # if x.dtype != torch.complex64:
        #     x = x.to(torch.complex64)
        x_np = np.complex128(x.cpu().numpy())
        y_np = self._operator.op(x_np)
        return torch.from_numpy(y_np).type(x.type())

    def A_adjoint(self, y):
        y_np = np.complex64(y.cpu().numpy())
        x_np = self._operator.adj_op(y_np)
        return torch.from_numpy(x_np).type(y.type())
