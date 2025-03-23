from abc import ABC, abstractmethod
from functools import partial
import math
from typing import Optional

from fast_hadamard_transform import hadamard_transform
import numpy as np
import scipy as sp
from scipy.fft import dct, idct, fft
import torch

from deepinv.physics.forward import LinearPhysics
from deepinv.optim.phase_retrieval import generate_signal


class Distribution(ABC):
    def __init__(self):
        self.min_supp: float
        self.max_supp: float
        self.max_pdf = None

    @abstractmethod
    def pdf(self, x) -> np.ndarray:
        pass

    def sample(self, shape: int | tuple[int, ...]) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)

        # empirical maximum if not yet computed
        if self.max_pdf is None:
            self.max_pdf = np.max(
                self.pdf(np.linspace(self.min_supp + 1e-8, self.max_supp - 1e-8, 10000))
            )

        n_samples = np.prod(shape)
        samples = np.empty(0)
        while samples.shape[0] < n_samples:
            n_need = int(n_samples - samples.shape)
            x = np.random.uniform(self.min_supp, self.max_supp, size=n_need)
            y = np.random.uniform(0, self.max_pdf, size=n_need)
            accepted = x[np.where(y < self.pdf(x))]
            if accepted.shape[0] >= n_need:
                samples = np.append(samples, accepted[:n_need])
                break
            else:
                samples = np.append(samples, accepted)
        return samples.reshape(shape)


class MarchenkoPastur(Distribution):
    """
    Marchenko-Pastur distribution.

    It describes the asymptotic eigenvalue distribution of the matrix X = 1/sqrt(m) A^T A, where A is a matrix of shape m times n and sampled i.i.d. from a distribution with zero mean and variance sigma^2
    """

    def __init__(self, alpha: float, sigma: float = 1.0):
        """
        alpha: oversampling ratio
        sigma: standard deviation of the element distribution
        """
        assert alpha >= 0, "oversampling ratio must be nonnegative"
        assert sigma >= 0, "standard deviation must be nonnegative"
        self.alpha = np.array(alpha)  # oversampling ratio
        self.gamma = 1 / self.alpha
        self.sigma = np.array(sigma)
        self.min_supp = self.sigma**2 * (1 - math.sqrt(self.gamma)) ** 2
        self.max_supp = self.sigma**2 * (1 + math.sqrt(self.gamma)) ** 2
        super().__init__()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        assert (x >= self.min_supp).all() and (
            x <= self.max_supp
        ).all(), "x is out of the support of the distribution"
        x = np.array(x)

        return np.sqrt((self.max_supp - x) * (x - self.min_supp)) / (
            2 * np.pi * self.sigma**2 * self.gamma * x
        )

    def sample(
        self,
        shape,
        normalized=False,
        include_zero=True,
    ) -> np.ndarray:
        """using acceptance-rejection sampling if oversampling ratio is more than 1, otherwise using the eigenvalues sampled from a real matrix"""
        n_samples = np.prod(shape)
        if self.alpha < 1.0:
            # undersampling
            # there will be zero eigenvalues, the rest nonzero eigenvalues follow the pdf
            if include_zero is True:
                n_zeros = int(n_samples * (1 - self.alpha))
                nonzeros = super().sample((n_samples - n_zeros,))
                zeros = np.zeros(n_zeros)
                samples = np.random.permutation(np.concatenate((nonzeros, zeros)))
            else:
                samples = super().sample((n_samples,))
        elif self.alpha == 1.0:
            # equisampling
            #! The distribution has min support at 0, leading to a very high peak near 0 and difficulty to sample from acceptance-rejection sampling
            #! Instead, we directly eigenvalue decompose a matrix to get the eigenvalues
            X = (
                1
                / np.sqrt(n_samples)
                * torch.randn((n_samples, n_samples), dtype=torch.cfloat)
            )
            samples, _ = torch.linalg.eig(X.conj().T @ X)
        else:
            # oversampling
            samples = super().sample(shape)
        if normalized:
            # normalize the samples such that E[x^2] = 1
            samples = samples / np.sqrt(1 + self.gamma) / (self.sigma**2)
        return samples.reshape(shape)

    def mean(self):
        return self.sigma**2

    def var(self):
        return self.gamma * self.sigma**4


def padding(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Zero padding function for oversampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the zero-padded tensor.
    """
    assert (
        tensor.shape[-3:] == input_shape
    ), f"tensor doesn't have the correct shape {tensor.shape}, expected {input_shape}."
    assert (input_shape[-1] <= output_shape[-1]) and (
        input_shape[-2] <= output_shape[-2]
    ), f"Input shape {input_shape} should be smaller than output shape {output_shape} for padding."

    change_top = math.ceil(abs(input_shape[-2] - output_shape[-2]) / 2)
    change_bottom = math.floor(abs(input_shape[-2] - output_shape[-2]) / 2)
    change_left = math.ceil(abs(input_shape[-1] - output_shape[-1]) / 2)
    change_right = math.floor(abs(input_shape[-1] - output_shape[-1]) / 2)

    return torch.nn.ZeroPad2d((change_left, change_right, change_top, change_bottom))(
        tensor
    )


def trimming(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Trimming function for undersampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the trimmed tensor.
    """
    assert (
        tensor.shape[-3:] == input_shape
    ), f"tensor doesn't have the correct shape {tensor.shape}, expected {input_shape}."
    assert (input_shape[-1] >= output_shape[-1]) and (
        input_shape[-2] >= output_shape[-2]
    ), f"Input shape {input_shape} should be larger than output shape {output_shape} for trimming."

    change_top = math.ceil(abs(input_shape[-2] - output_shape[-2]) / 2)
    change_bottom = math.floor(abs(input_shape[-2] - output_shape[-2]) / 2)
    change_left = math.ceil(abs(input_shape[-1] - output_shape[-1]) / 2)
    change_right = math.floor(abs(input_shape[-1] - output_shape[-1]) / 2)

    if change_bottom == 0:
        tensor = tensor[..., change_top:, :]
    else:
        tensor = tensor[..., change_top:-change_bottom, :]
    if change_right == 0:
        tensor = tensor[..., change_left:]
    else:
        tensor = tensor[..., change_left:-change_right]
    return tensor


def generate_diagonal(
    shape: tuple[int, ...],
    mode,
    config: Optional[dict] = None,
    dtype=torch.complex64,
    device="cpu",
    generator=torch.Generator("cpu"),
):
    r"""
    Generate a random tensor as the diagonal matrix.
    """

    if isinstance(mode, str):
        if mode == "rademacher":
            diag = torch.where(
                torch.rand(shape, device=device, generator=generator) > 0.5, -1.0, 1.0
            )
        elif mode == "gaussian":
            diag = torch.randn(shape, dtype=dtype)
        elif mode == "student-t":
            #! variance = df/(df-2) if df > 2
            #! variance of complex numbers is doubled
            student_t_dist = torch.distributions.studentT.StudentT(
                config["degree_of_freedom"], 0, 1
            )
            scale = torch.sqrt(
                (torch.tensor(config["degree_of_freedom"]) - 2)
                / torch.tensor(config["degree_of_freedom"])
                / 2
            )
            diag = scale * (
                student_t_dist.sample(shape) + 1j * student_t_dist.sample(shape)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    elif isinstance(mode, list):
        assert (
            len(mode) == 2
        ), "mode must be a list of two elements to specify the magnitude and phase distributions"
        mag, phase = mode
        #! should be normalized to have E[|x|^2] = 1 if energy conservation
        if mode[0] == "unit":
            mag = torch.ones(shape, dtype=dtype)
        elif mode[0] == "marchenko":
            mag = torch.from_numpy(
                MarchenkoPastur(alpha=config["alpha"]).sample(
                    shape, normalized=True, include_zero=config["include_zero"]
                )
            ).to(dtype)
            mag = torch.sqrt(mag)
        elif mode[0] == "custom":
            mag = torch.sqrt(config["diagonal"])
        else:
            raise ValueError(f"Unsupported magnitude: {mode[0]}")

        if mode[1] == "uniform":
            phase = 2 * np.pi * torch.rand(shape)
            phase = torch.exp(1j * phase)
        elif mode[1] == "zero":
            phase = torch.ones(shape, dtype=dtype)
        elif mode[1] == "laplace":
            #! variance = 2*scale^2
            #! variance of complex numbers is doubled
            laplace_dist = torch.distributions.laplace.Laplace(0, 0.5)
            phase = laplace_dist.sample(shape) + 1j * laplace_dist.sample(shape)
            phase = phase / phase.abs()
        elif mode[1] == "quadrant":
            # generate random phase 1, -1, j, -j
            values = torch.tensor([1, -1, 1j, -1j])
            # Randomly select elements from the values with equal probability
            phase = values[torch.randint(0, len(values), shape)]
        else:
            raise ValueError(f"Unsupported phase: {mode[1]}")

        diag = mag * phase
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return diag.to(device)


def generate_spectrum(
    shape: tuple[int, ...],
    mode: str,
    config: Optional[dict] = None,
    dtype=torch.complex64,
    device="cpu",
    generator: Optional[torch.Generator] = torch.Generator("cpu"),
):
    if mode == "unit":
        spectrum = torch.ones(shape, dtype=dtype)
    elif mode == "marchenko":
        spectrum = torch.from_numpy(
            MarchenkoPastur(alpha=config["alpha"]).sample(
                shape, normalized=True, include_zero=config["include_zero"]
            )
        ).to(dtype)
        spectrum = torch.sqrt(spectrum)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return spectrum.to(device)


def dst1(x):
    r"""
    Orthogonal Discrete Sine Transform, Type I
    The transform is performed across the last dimension of the input signal
    Due to orthogonality we have ``dst1(dst1(x)) = x``.

    :param torch.Tensor x: the input signal
    :return: (torch.tensor) the DST-I of the signal over the last dimension

    """
    x_shape = x.shape

    b = int(np.prod(x_shape[:-1]))
    n = x_shape[-1]
    x = x.view(-1, n)

    z = torch.zeros(b, 1, device=x.device)
    x = torch.cat([z, x, z, -x.flip([1])], dim=1)
    x = torch.view_as_real(torch.fft.rfft(x, norm="ortho"))
    x = x[:, 1:-1, 1]
    return x.view(*x_shape)


def fft1(x: torch.Tensor, device) -> torch.Tensor:
    r"""1D FFT

    FFT is performed on the flattened input tensor and then reshape to the original shape.
    """
    shape = x.shape
    x = x.flatten()
    x = torch.fft.fft(x, norm="ortho")
    return x.reshape(shape)


def ifft1(x: torch.Tensor, device) -> torch.Tensor:
    r"""1D IFFT

    IFFT is performed on the flattened input tensor and then reshape to the original shape.
    """
    shape = x.shape
    x = x.flatten()
    x = torch.fft.ifft(x, norm="ortho")
    return x.reshape(shape)


def fft2(x: torch.Tensor, device) -> torch.Tensor:
    r"""2D FFT

    FFT is performed along the last two dimensions of the input tensor.
    """
    return torch.fft.fft2(x, norm="ortho")


def ifft2(x: torch.Tensor, device) -> torch.Tensor:
    r"""2D IFFT

    IFFT is performed along the last two dimensions of the input tensor.
    """
    return torch.fft.ifft2(x, norm="ortho")


def dct1(x: torch.Tensor, device) -> torch.Tensor:
    r"""1D DCT

    DCT is performed on the flattened input tensor and then reshape to the original shape.
    """
    shape = x.shape
    x = x.flatten()
    x = torch.from_numpy(dct(x.cpu().numpy(), norm="ortho")).to(device)
    return x.reshape(shape)


def idct1(x: torch.Tensor, device) -> torch.Tensor:
    r"""1D IDCT

    IDCT is performed on the flattened input tensor and then reshape to the original shape.
    """
    shape = x.shape
    x = x.flatten()
    x = torch.from_numpy(idct(x.cpu().numpy(), norm="ortho")).to(device)
    return x.reshape(shape)


def dct2(x: torch.Tensor, device):
    r"""2D DCT

    DCT is performed along the last two dimensions of the input tensor.
    """
    return torch.from_numpy(
        dct(dct(x.cpu().numpy(), axis=-1, norm="ortho"), axis=-2, norm="ortho")
    ).to(device)


def idct2(x: torch.Tensor, device):
    r"""2D IDCT

    IDCT is performed along the last two dimensions of the input tensor.
    """
    return torch.from_numpy(
        idct(idct(x.cpu().numpy(), axis=-2, norm="ortho"), axis=-1, norm="ortho")
    ).to(device)


def hadamard1(x):
    shape = x.shape

    x = x.flatten()
    real = x.real
    imag = x.imag
    real = hadamard_transform(real, scale=1 / np.sqrt(x.shape[0]))
    imag = hadamard_transform(imag, scale=1 / np.sqrt(x.shape[0]))

    x = real + 1j * imag
    x = torch.reshape(x, shape)

    return x


def hadamard2(x):
    assert x.dim() >= 2, "Input tensor must have shape (..., H, W)"
    *_, h, w = x.shape

    real = x.real
    imag = x.imag
    real = hadamard_transform(
        hadamard_transform(real, scale=1 / np.sqrt(w)).transpose(-2, -1),
        scale=1 / np.sqrt(h),
    ).transpose(-2, -1)
    imag = hadamard_transform(
        hadamard_transform(imag, scale=1 / np.sqrt(w)).transpose(-2, -1),
        scale=1 / np.sqrt(h),
    ).transpose(-2, -1)

    x = real + 1j * imag

    return x


def oversampling_matrix(
    m: int, n: int, pos="center", dtype=torch.complex64, device="cpu"
) -> torch.Tensor:
    """Generate an oversampling matrix of shape (m, n) with its upper part being identity and the rest being zero"""
    assert m >= n, "m should be larger than or equal to n"

    if pos == "first":
        return (
            torch.cat((torch.eye(n), torch.zeros(m - n, n)), dim=0).to(dtype).to(device)
        )
    # alternative way, make the center of the matrix identity
    # dimension is still m x n
    elif pos == "center":
        return (
            torch.cat(
                (
                    torch.zeros(math.ceil((m - n) / 2), n),
                    torch.eye(n),
                    torch.zeros(math.floor((m - n) / 2), n),
                ),
                dim=0,
            )
            .to(dtype)
            .to(device)
        )
    elif pos == "last":
        return (
            torch.cat((torch.zeros(m - n, n), torch.eye(n)), dim=0).to(dtype).to(device)
        )
    else:
        raise ValueError(f"Unsupported position: {pos}")


def subsampling_matrix(
    m: int, n: int, pos="center", dtype=torch.complex64, device="cpu"
) -> torch.Tensor:
    """Generate a subsampling matrix of shape (m, n) with its left part being identity and the rest being zero"""
    assert m <= n, "m should be smaller than or equal to n"

    if pos == "first":
        return (
            torch.cat((torch.eye(m), torch.zeros(m, n - m)), dim=1).to(dtype).to(device)
        )
    # alternative way, make the center of the matrix identity
    elif pos == "center":
        return (
            torch.cat(
                (
                    torch.zeros(m, math.ceil((n - m) / 2)),
                    torch.eye(m),
                    torch.zeros(m, math.floor((n - m) / 2)),
                ),
                dim=1,
            )
            .to(dtype)
            .to(device)
        )
    elif pos == "last":
        return (
            torch.cat((torch.zeros(m, n - m), torch.eye(m)), dim=1).to(dtype).to(device)
        )
    else:
        raise ValueError(f"Unsupported position: {pos}")


def diagonal_matrix(
    diag: torch.Tensor, dtype=torch.complex64, device="cpu"
) -> torch.Tensor:
    """Given a torch tensor, construct a diagonal matrix with the tensor as the diagonal"""

    return torch.diag(diag.flatten()).to(dtype).to(device)


def dft_matrix(n: int, dtype=torch.complex64, device="cpu"):
    """Generate the DFT matrix of size n"""
    T = fft(np.eye(n), axis=0, norm="ortho")
    return torch.tensor(T).to(device).to(dtype)


def dct_matrix(n: int, dtype=torch.complex64, device="cpu"):
    """Generate the DCT matrix of size n"""
    T = dct(np.eye(n), axis=0, norm="ortho")
    return torch.tensor(T).to(device).to(dtype)


def hadamard_matrix(n: int, dtype=torch.complex64, device="cpu"):
    """Generate the Hadamard matrix of size n"""
    # assert n is a power of 2
    assert n & (n - 1) == 0, "n should be a power of 2"

    T = sp.linalg.hadamard(n)
    # scale to be orthogonal
    T = torch.tensor(T) / torch.sqrt(torch.tensor(n))
    return T.to(dtype).to(device)


class StructuredRandom(LinearPhysics):
    r"""
    Structured random linear operator model corresponding to the operator

    .. math::

        A(x) = \prod_{i=1}^N (F D_i) x,

    where :math:`F` is a matrix representing a structured transform, :math:`D_i` are diagonal matrices, and :math:`N` refers to the number of layers. It is also possible to replace :math:`x` with :math:`Fx` as an additional 0.5 layer.

    :param tuple input_shape: input shape. If (C, H, W), i.e., the input is a 2D signal with C channels, then zero-padding will be used for oversampling and cropping will be used for undersampling.
    :param tuple output_shape: shape of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math`F` transform is included, ie :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`. Default is 1.
    :param list transforms: transform functions for each layer.
    :param list transform_invs: inverse transform for each layer.
    :param list diagonals: list of diagonal matrices. If None, a random :math:`{-1,+1}` mask matrix will be used. Default is None.
    :param str device: device of the physics. Default is 'cpu'.
    :param torch.Generator rng: Random number generator. Default is None.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        middle_shape: Optional[tuple] = None,
        n_layers=1,
        spectrum=None,
        transforms=["dst1"],
        diagonals=None,
        diagonal_config=None,
        shared_weights=False,
        explicit_matrix=False,
        dtype=torch.complex64,
        device="cpu",
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ):
        self.input_shape = input_shape
        self.middle_shape = middle_shape
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.dtype = dtype
        self.device = device

        # * generate spectrum
        if spectrum is None:
            # default setting for fast compressed sensing
            spectrum = generate_spectrum(
                shape=input_shape,
                mode="uniform",
                dtype=torch.float,
                generator=rng,
                device=device,
            )
        if isinstance(spectrum, str):
            #! spectrum shape will always be the input shape
            self.spectrum = generate_spectrum(
                shape=input_shape,
                mode=spectrum,
                config=diagonal_config,
                dtype=self.dtype,
                device=self.device,
                generator=torch.Generator(device=device),
            )
        else:
            self.spectrum = spectrum

        # * generate diagonal matrices
        if diagonals is None:
            # default settings for fast compressed sensing
            diagonals = [
                generate_diagonal(
                    shape=input_shape,
                    mode="rademacher",
                    dtype=torch.float,
                    generator=rng,
                    device=device,
                )
            ]
        self.diagonals = []
        if not shared_weights:
            for i in range(math.floor(self.n_layers)):
                diagonal = generate_diagonal(
                    self.middle_shape,
                    mode=diagonals[i],
                    dtype=self.dtype,
                    device=self.device,
                    config=diagonal_config,
                )
                self.diagonals.append(diagonal)
        else:
            diagonal = generate_diagonal(
                self.middle_shape,
                mode=diagonals[0],
                dtype=self.dtype,
                device=self.device,
                config=diagonal_config,
            )
            self.diagonals = self.diagonals + [diagonal] * math.floor(self.n_layers)

        # * generate transform functions
        self.transforms = []
        self.transform_invs = []
        for transform in transforms:
            if transform == "dst1":
                self.transforms.append(dst1)
                self.transform_invs.append(dst1)
            if transform == "fourier1":
                self.transforms.append(partial(fft1, device=self.device))
                self.transform_invs.append(partial(ifft1, device=self.device))
            elif transform == "fourier2":
                self.transforms.append(partial(fft2, device=self.device))
                self.transform_invs.append(partial(ifft2, device=self.device))
            elif transform == "cosine1":
                self.transforms.append(partial(dct1, device=self.device))
                self.transform_invs.append(partial(idct1, device=self.device))
            elif transform == "cosine2":
                self.transforms.append(partial(dct2, device=self.device))
                self.transform_invs.append(partial(idct2, device=self.device))
            elif transform == "hadamard1":
                self.transforms.append(hadamard1)
                self.transform_invs.append(hadamard1)
            elif transform == "hadamard2":
                self.transforms.append(hadamard2)
                self.transform_invs.append(hadamard2)
            else:
                raise ValueError(f"Unimplemented transform: {transform}")

        # * construct forward matrix
        if explicit_matrix:
            self.get_forward_matrix(transforms)

        # forward operator
        def A(x):
            assert (
                x.shape[1:] == self.input_shape
            ), f"x doesn't have the correct shape {x.shape[1:]} != {self.input_shape}"
            x = x * self.spectrum

            if len(self.input_shape) == 3:
                x = padding(x, self.input_shape, self.middle_shape)

            # position of the transform
            p = 0
            if self.n_layers - math.floor(self.n_layers) == 0.5:
                x = self.transforms[p](x)
                p += 1
            for i in range(math.floor(self.n_layers)):
                x = self.diagonals[i] * x
                x = self.transforms[p](x)
                p += 1

            if len(self.input_shape) == 3:
                x = trimming(x, self.middle_shape, self.output_shape)

            return x

        def A_adjoint(y):
            assert (
                y.shape[1:] == self.output_shape
            ), f"y doesn't have the correct shape {y.shape[1:]} != {self.output_shape}"

            if len(self.input_shape) == 3:
                y = padding(y, self.output_shape, self.middle_shape)

            for i in range(math.floor(self.n_layers)):
                y = self.transform_invs[-i - 1](y)
                y = torch.conj(self.diagonals[-i - 1]) * y
            if self.n_layers - math.floor(self.n_layers) == 0.5:
                y = self.transform_invs[0](y)

            if len(self.input_shape) == 3:
                y = trimming(y, self.middle_shape, self.input_shape)

            y = y * torch.conj(self.spectrum)

            return y

        super().__init__(A=A, A_adjoint=A_adjoint, **kwargs)

    def get_forward_matrix(self, transforms, verbose=False):
        """Given the structure of the operator, return the forward matrix."""
        if not hasattr(self, "forward_matrix"):
            m = np.prod(self.output_shape)
            p = np.prod(self.middle_shape)
            n = np.prod(self.input_shape)

            if verbose:
                print("computing transform matrix")
            transform_matrices = []
            for transform in transforms:
                if "fourier" in transform:
                    transform_matrices.append(dft_matrix(p, self.dtype, self.device))
                elif "cosine" in transform:
                    transform_matrices.append(dct_matrix(p, self.dtype, self.device))
                elif "hadamard" in transform:
                    transform_matrices.append(
                        hadamard_matrix(p, self.dtype, self.device)
                    )
                else:
                    raise ValueError(f"Unsupported transform: {transform}")

            mat = torch.eye(n).to(self.dtype).to(self.device)
            if verbose:
                print("computing oversampling")
            mat = oversampling_matrix(p, n, dtype=self.dtype, device=self.device) @ mat
            if verbose:
                print("computing transform")
            counter = 0
            if self.n_layers - math.floor(self.n_layers) == 0.5:
                mat = transform_matrices[counter] @ mat
                counter += 1
            for i in range(math.floor(self.n_layers)):
                mat = (
                    diagonal_matrix(self.diagonals[i].flatten(), device=self.device)
                    @ mat
                )
                mat = transform_matrices[counter] @ mat
                counter += 1
            if verbose:
                print("computing undersampling")
            mat = subsampling_matrix(m, p, dtype=self.dtype, device=self.device) @ mat

            self.forward_matrix = mat
            return mat
        else:
            return self.forward_matrix

    def get_singular_values(self):
        """Compute the singular values of the forward matrix"""
        if not hasattr(self, "forward_matrix"):
            self.get_forward_matrix()
        s = sp.linalg.svdvals(self.forward_matrix.cpu().numpy())
        return s

    def partial_forward(self, x, n_layers) -> torch.Tensor:
        """Compute the forward operator until n_layers.

        x and return both have the middle shape"""

        assert (
            n_layers <= self.n_layers
        ), f"n_layers should be no greater than to {self.n_layers}"
        assert (
            self.n_layers - math.floor(self.n_layers) == 0.0
        ), "currently only support integer number of layers"

        x = padding(x, self.input_shape, self.middle_shape)

        for i in range(math.floor(n_layers)):
            x = self.diagonals[i] * x
            x = self.transforms[i](x)

        return x

    def partial_inverse(self, y, n_layers) -> torch.Tensor:
        """Compute the inverse operator from n_layers to the start.

        y and return both have the middle shape.
        """

        assert (
            n_layers <= self.n_layers
        ), f"n_layers should be no greater than {self.n_layers}"
        assert (
            self.n_layers - math.floor(self.n_layers) == 0.0
        ), "currently only support integer number of layers"

        for i in range(math.floor(n_layers)):
            y = self.transform_invs[-(self.n_layers - n_layers) - i - 1](y)
            y = y / self.diagonals[-(self.n_layers - n_layers) - i - 1]

        return y

    def get_adversarial(self, n_layers=None, trimmed=True) -> torch.Tensor:
        """returns the closest input to the signal which yields the measuremnts as a delta signal"""

        if n_layers is None:
            n_layers = self.n_layers

        delta = generate_signal(
            (1,) + self.middle_shape,
            mode=["delta", "constant"],
            dtype=self.dtype,
            device=self.device,
        )
        adver = self.partial_inverse(delta, n_layers)

        if trimmed is True:
            return trimming(adver, self.middle_shape, self.input_shape)
        else:
            return adver
