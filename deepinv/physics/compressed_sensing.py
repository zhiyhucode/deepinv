import numpy as np
import torch

from deepinv.physics.forward import LinearPhysics
from deepinv.physics.functional import random_choice
from deepinv.physics.structured_random import StructuredRandom, generate_diagonal


class CompressedSensing(LinearPhysics):
    r"""
    Compressed Sensing forward operator. Creates a random sampling :math:`m \times n` matrix where :math:`n` is the
    number of elements of the signal, i.e., ``np.prod(img_shape)`` and ``m`` is the number of measurements.

    This class generates a random iid Gaussian matrix if ``fast=False``

    .. math::

        A_{i,j} \sim \mathcal{N}(0,\frac{1}{m})

    or a Subsampled Orthogonal with Random Signs matrix (SORS) if ``fast=True`` (see https://arxiv.org/abs/1506.03521)

    .. math::

        A = \text{diag}(m)D\text{diag}(s)

    where :math:`s\in\{-1,1\}^{n}` is a random sign flip with probability 0.5,
    :math:`D\in\mathbb{R}^{n\times n}` is a fast orthogonal transform (DST-1) and
    :math:`\text{diag}(m)\in\mathbb{R}^{m\times n}` is random subsampling matrix, which keeps :math:`m` out of :math:`n` entries.

    For image sizes bigger than 32 x 32, the forward computation can be prohibitively expensive due to its :math:`O(mn)` complexity.
    In this case, we recommend using :class:`deepinv.physics.StructuredRandom` instead.

    .. deprecated:: 0.2.2
       The ``fast`` option is deprecated and might be removed in future versions. Use :class:`deepinv.physics.StructuredRandom` instead.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    .. note::

        If ``fast=False``, the forward operator has a norm which tends to :math:`(1+\sqrt{n/m})^2` for large :math:`n`
        and :math:`m` due to the `Marcenko-Pastur law
        <https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>`_.
        If ``fast=True``, the forward operator has a unit norm.

    If ``dtype=torch.cfloat``, the forward operator will be generated as a random i.i.d. complex Gaussian matrix to be used with ``fast=False``

    .. math::

        A_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool fast: The operator is iid Gaussian if false, otherwise A is a SORS matrix with the Discrete Sine Transform (type I).
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param str mode: If ``unitary``, the forward operator is a Haar matrix. If ``circulant``, the forward operator is a circulant matrix. Default is None.
    :param int product: Number of Gaussian matrices to multiply. Default is 1.
    :param bool unit_mag: Normalize the forward matrix to have unit magnitude. Default is False.
    :param bool compute_inverse: Precompute the pseudo-inverse of the forward matrix (only for ``fast=False`` option). Precomputing the pseudoinverse can be slow if the matrix is large. Default is ``False``.
    :param torch.type dtype: Forward matrix is stored as a dtype. For complex matrices, use torch.cfloat. Default is torch.float.
    :param str device: Device to store the forward matrix.
    :param torch.Generator (Optional) rng: a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Compressed sensing operator with 100 measurements for a 3x3 image:

        >>> from deepinv.physics import CompressedSensing
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = CompressedSensing(m=10, img_shape=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[-1.7769,  0.6160, -0.8181, -0.5282, -1.2197,  0.9332, -0.1668,  1.5779,
                  0.6752, -1.5684]])

    """

    def __init__(
        self,
        m,
        img_shape,
        mode=None,
        product=1,
        unit_mag=False,
        compute_inverse=False,
        fast=False,
        channelwise=False,
        dtype=torch.float,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"CS_m{m}"
        self.img_shape = img_shape
        self.fast = fast
        self.channelwise = channelwise
        self.compute_inverse = compute_inverse
        self.dtype = dtype
        self.device = device

        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {self.device}."
            self.rng = rng
        self.initial_random_state = self.rng.get_state()

        if channelwise:
            n = int(np.prod(img_shape[1:]))
        else:
            n = int(np.prod(img_shape))

        if self.fast:
            print(
                "Warning: fast option is deprecated and might be removed in future versions."
            )
            # generate random subsampling matrix
            self.n = n
            self.mask = torch.zeros(self.n, device=device)
            idx = torch.sort(
                random_choice(self.n, size=m, replace=False, rng=self.rng)
            ).values
            self.mask[idx] = 1
            self.mask = self.mask.type(torch.bool)
            self.mask = torch.nn.Parameter(self.mask, requires_grad=False)

            # generate random sign matrix
            self.D = generate_diagonal(
                shape=(n,),
                mode="rademacher",
                dtype=torch.float,
                device=device,
                generator=self.rng,
            )
            self.D = torch.nn.Parameter(self.D, requires_grad=False)

            self.FD = StructuredRandom(
                input_shape=(n,),
                output_shape=(n,),
                device=device,
                diagonals=[self.D],
            )

        else:
            if mode == "unitary":
                print("Using Haar matrix")
                self._A = torch.randn((m, n), device=device, dtype=dtype) / np.sqrt(m)
                self._A, R = torch.linalg.qr(self._A)
                L = torch.sgn(torch.diag(R))
                self._A = self._A * L[None, :]
            elif mode == "circulant":
                print("Using circulant matrix")
                # generate a random row and use it to construct a circulant matrix
                dim = m if m > n else n
                v = torch.randn((dim,), device=device, dtype=dtype)
                self._A = torch.stack([torch.roll(v, shifts=-i) for i in range(dim)])
                # reshape to emulate oversampling/undersampling
                self._A = self._A[:m, :n] / np.sqrt(m)
            else:
                self._A = torch.randn((m, n), device=device, dtype=dtype)
                self._A = self._A / np.sqrt(m)
                # product of Gaussian matrices
                for i in range(product - 1):
                    gaussian = torch.randn(
                        (m, m), device=device, dtype=dtype
                    ) / np.sqrt(m)
                    self._A = gaussian @ self._A

            if unit_mag is True:
                self._A = self._A / self._A.abs()
            self._A = torch.nn.Parameter(self._A, requires_grad=False)

            if self.compute_inverse is True:
                self._A_dagger = torch.linalg.pinv(self._A)
                self._A_dagger = torch.nn.Parameter(self._A_dagger, requires_grad=False)

            self._A_adjoint = (
                torch.nn.Parameter(self._A.conj().T, requires_grad=False)
                .type(dtype)
                .to(device)
            )

    def A(self, x, **kwargs):
        N, C = x.shape[:2]
        if self.channelwise:
            x = x.reshape(N * C, -1)
        else:
            x = x.reshape(N, -1)

        if self.fast:
            y = self.FD(x)[:, self.mask]
        else:
            y = torch.einsum("in, mn->im", x, self._A)

        if self.channelwise:
            y = y.view(N, C, -1)

        return y

    def A_adjoint(self, y, **kwargs):
        y = y.type(self.dtype)
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        if self.channelwise:
            N2 = N * C
            y = y.view(N2, -1)
        else:
            N2 = N

        if self.fast:
            y2 = torch.zeros((N2, self.n), device=y.device)
            y2[:, self.mask] = y.type(y2.dtype)
            x = self.FD.A_adjoint(y2)
        else:
            x = torch.einsum("im, nm->in", y, self._A_adjoint)  # x:(N, n, 1)

        x = x.view(N, C, H, W)
        return x

    def A_dagger(self, y, **kwargs):
        y = y.type(self.dtype)
        if self.fast:
            return self.A_adjoint(y)
        else:
            N = y.shape[0]
            C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]

            if self.channelwise:
                y = y.reshape(N * C, -1)

            x = torch.einsum("im, nm->in", y, self._A_dagger)
            x = x.reshape(N, C, H, W)
        return x


# if __name__ == "__main__":
#     device = "cuda:0"
#
#     # for comparing fast=True and fast=False forward matrices.
#     for i in range(1):
#         n = 2 ** (i + 4)
#         im_size = (1, n, n)
#         m = int(np.prod(im_size))
#         x = torch.randn((1,) + im_size, device=device)
#
#         print((dst1(dst1(x)) - x).flatten().abs().sum())
#
#         physics = CompressedSensing(img_shape=im_size, m=m, fast=True, device=device)
#
#         print((physics.A_adjoint(physics.A(x)) - x).flatten().abs().sum())
#         print(f"adjointness: {physics.adjointness_test(x)}")
#         print(f"norm: {physics.power_method(x, verbose=False)}")
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         for j in range(100):
#             y = physics.A(x)
#             xhat = physics.A_dagger(y)
#         end.record()
#
#         # print((xhat-x).pow(2).flatten().mean())
#
#         # Waits for everything to finish running
#         torch.cuda.synchronize()
#         print(start.elapsed_time(end))
