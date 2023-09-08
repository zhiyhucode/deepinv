# WARNING: this is a preliminary implementation of the NUFFT operator, which is a mere wrapper around the `mri-nufft`
# package, itself relying on the finufft package. This implementation is not optimized and should be used with caution.

import torch

import mrinufft
from mrinufft.trajectories.density import voronoi

from deepinv.physics.forward import LinearPhysics


class Nufft(LinearPhysics):
    r"""
    Non-uniformly sampled FFT operator.

    The linear operator on tensors of shape B x 1 x H x W operates in 2D slices and is defined as, for each frequency
    point :math:`\nu_k \in \mathbb{R}^2`:

    .. math::
        \begin{align*}
        \A\colon \mathbb{C}^2 \times \mathbb{R}^{2N} & \longrightarrow \mathbb{C}^{2}\\
        x, d &\longmapsto \sum_{n=0}^{N-1} x(p_n) e^{ -2\pi i d_n \nu_k},
        \end{align*}


    where :math:`x` is an image and :math:`d` are the locations of the samples.

    Following the nomenclature from `mri-nufft` we assume that the density is of shape (M, N, 2) where M is a number of
    shots and N is the number of samples per shot. See `mri-nufft` for more details.

    :param tuple img_size: size of the input images, e.g., (B, C, H, W).
    :param torch.tensor samples_loc: locations of the samples, of shape (M, N, 2).
    :param str density: density function, e.g., 'voronoi'. If None, the density is uniform. Default: None.
    """
    def __init__(
            self,
            img_size,
            samples_loc,
            density=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if density is not None:
            if density == 'voronoi':
                density = voronoi(samples_loc.reshape(-1, 2))

        NufftOperator = mrinufft.get_operator("finufft")
        self.nufft = NufftOperator(samples_loc.reshape(-1, 2), shape=img_size, density=density, n_coils=1)

    def forward(self, x):
        r"""
        Applies the forward operator to the input.
        """
        return self.A(x)

    def backward(self, k):
        r"""
        Applies the adjoint operator to the input.
        """
        return self.A_adjoint(k)

    def A(self, x):
        r"""
        Applies the forward operator to the input.
        """
        x_numpy = x.cpu().numpy()
        kspace = self.nufft.op(x_numpy)
        return torch.from_numpy(kspace)

    def A_adjoint(self, kspace):
        r"""
        Applies the adjoint operator to the input.
        """
        kspace_numpy = kspace.cpu().numpy()
        im = self.nufft.adj_op(kspace_numpy)
        return torch.from_numpy(im).type(torch.FloatTensor)
