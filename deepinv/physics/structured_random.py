from abc import ABC, abstractmethod
import math

from fast_hadamard_transform import hadamard_transform
import numpy as np
import scipy as sp
from scipy.fft import dct, idct, fft
import torch

from deepinv.physics.forward import LinearPhysics


class Distribution(ABC):
    def __init__(self):
        self.max_pdf = None

    @abstractmethod
    def pdf(self, x):
        pass

    def sample(self, shape: tuple[int, ...]) -> np.ndarray:
        # compute the maximum value of the pdf if not yet computed
        if self.max_pdf is None:
            self.max_pdf = np.max(
                self.pdf(np.linspace(self.min_supp + 1e-8, self.max_supp - 1e-8, 10000))
            )

        samples = []
        while len(samples) < np.prod(shape):
            x = np.random.uniform(self.min_supp, self.max_supp, size=1)
            y = np.random.uniform(0, self.max_pdf, size=1)
            if y < self.pdf(x):
                samples.append(x)
        return np.array(samples).reshape(shape)


def triangular_distribution(a, size):
    u = torch.rand(size)  # Sample from uniform distribution [0, 1]

    # Apply inverse transform method for triangular distribution
    condition = u < 0.5
    samples = torch.zeros(size)

    # Left part of the triangular distribution
    samples[condition] = -a + torch.sqrt(u[condition] * 2 * a**2)

    # Right part of the triangular distribution
    samples[~condition] = a - torch.sqrt((1 - u[~condition]) * 2 * a**2)

    return samples


class MarchenkoPastur(Distribution):
    def __init__(self, m, n, sigma=None):
        self.m = np.array(m)
        self.n = np.array(n)
        # when oversampling ratio is 1, the distribution has min support at 0, leading to a very high peak near 0 and numerical issues.
        self.gamma = np.array(n / m)
        if sigma is not None:
            self.sigma = np.array(sigma)
        else:
            # automatically set sigma to make E[|x|^2] = 1
            # self.sigma = (1+self.gamma)**(-0.25)
            self.sigma = 1
        self.lamb = m / n
        self.min_supp = np.array(self.sigma**2 * (1 - np.sqrt(self.gamma)) ** 2)
        self.max_supp = np.array(self.sigma**2 * (1 + np.sqrt(self.gamma)) ** 2)
        super().__init__()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        assert (x >= self.min_supp).all() and (
            x <= self.max_supp
        ).all(), "x is out of the support of the distribution"
        return np.sqrt((self.max_supp - x) * (x - self.min_supp)) / (
            2 * np.pi * self.sigma**2 * self.gamma * x
        )

    def sample(self, shape: tuple[int, ...], include_zero=False) -> np.ndarray:
        """using acceptance-rejection sampling if oversampling ratio is more than 1, otherwise using the eigenvalues sampled from a real matrix"""
        if self.m < self.n:
            # there will be n - m zero eigenvalues, the rest nonzero eigenvalues follow the Marchenko-Pastur distribution
            if include_zero is True:
                n_zeros = int(np.product(shape) / self.n * (self.n - self.m))
                nonzeros = super().sample((np.product(shape) - n_zeros,))
                zeros = np.zeros(n_zeros)
                return np.random.permutation(np.concatenate((nonzeros, zeros))).reshape(
                    shape
                )
            else:
                return super().sample(shape)
        elif self.m == self.n:
            # compute the eigenvalues from a real matrix and use it as the samples
            X = 1 / np.sqrt(self.m) * torch.randn((self.m, self.n), dtype=torch.cfloat)
            eigenvalues_X, _ = torch.linalg.eig(X.conj().T @ X)
            return np.array(eigenvalues_X).reshape(shape)
        else:
            return super().sample(shape)

    def mean(self):
        return self.sigma**2

    def var(self):
        return self.gamma * self.sigma**4


class SemiCircle(Distribution):
    def __init__(self, radius):
        self.radius = radius
        self.min_supp = 1 - radius
        self.max_supp = 1 + radius
        super().__init__()

    def pdf(self, x):
        return 2 / (np.pi * self.radius**2) * np.sqrt(self.radius**2 - (x - 1) ** 2)


def compare(input_shape: tuple, output_shape: tuple) -> str:
    r"""
    Compare input and output shape to determine the sampling mode.

    :param tuple input_shape: Input shape (C, H, W).
    :param tuple output_shape: Output shape (C, H, W).

    :return: The sampling mode in ["oversampling","undersampling","equisampling].
    """
    if input_shape[1] == output_shape[1] and input_shape[2] == output_shape[2]:
        return "equisampling"
    elif input_shape[1] <= output_shape[1] and input_shape[2] <= output_shape[2]:
        return "oversampling"
    elif input_shape[1] >= output_shape[1] and input_shape[2] >= output_shape[2]:
        return "undersampling"
    else:
        raise ValueError(
            "Does not support different sampling schemes on height and width."
        )


def padding(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Zero padding function for oversampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (torch.Tensor) the zero-padded tensor.
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

    :return: (torch.Tensor) the trimmed tensor.
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
    config: dict = None,
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
        elif mode == "triangular":
            #! variance = a^2/6 for real numbers
            real = triangular_distribution(torch.sqrt(torch.tensor(3)), shape)
            imag = triangular_distribution(torch.sqrt(torch.tensor(3)), shape)
            diag = real + 1j * imag
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
        elif mode[0] == "uniform":
            # ensure E[|x|^2] = 1
            mag = torch.sqrt(torch.tensor(3.0)) * torch.rand(shape)
            mag = mag.to(dtype)
        elif mode[0] == "marchenko":
            mag = torch.from_numpy(
                MarchenkoPastur(config["m"], config["n"]).sample(shape)
            ).to(dtype)
            mag = torch.sqrt(mag)
        elif mode[0] == "semicircle":
            mag = torch.from_numpy(SemiCircle(config["radius"]).sample(shape)).to(dtype)
        elif mode[0] == "custom":
            mag = torch.sqrt(config["diagonal"])
        elif mode[0] == "cauchy":
            mag = torch.distributions.cauchy.Cauchy(0, 1).sample(shape).to(dtype)
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
    config: dict = None,
    dtype=torch.complex64,
    device="cpu",
    generator=torch.Generator("cpu"),
):
    if mode == "unit":
        spectrum = torch.ones(shape, dtype=dtype)
    elif mode == "marchenko":
        spectrum = torch.from_numpy(
            MarchenkoPastur(config["m"], config["n"]).sample(shape)
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


def hadamard2(x):

    assert x.dim() == 4, "Input tensor must have shape (N, C, H, W)"
    n, c, h, w = x.shape

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

    # x = x.flatten()
    # real = x.real
    # imag = x.imag
    # real = hadamard_transform(real, scale=1 / np.sqrt(x.shape[0]))
    # imag = hadamard_transform(imag, scale=1 / np.sqrt(x.shape[0]))

    # x = real + 1j * imag
    # x = torch.reshape(x, (n, c, h, w))

    return x


def oversampling_matrix(m, n, dtype=torch.complex64, device="cpu"):
    """Generate an oversampling matrix of shape (m, n) with its upper part being identity and the rest being zero"""
    assert m >= n, "m should be larger than or equal to n"
    # return torch.cat((torch.eye(n), torch.zeros(m - n, n)), dim=0).to(dtype).to(device)
    # alternative way, make the center of the matrix identity
    # dimension is still m x n
    return (
        torch.cat(
            (torch.zeros((m - n) // 2, n), torch.eye(n), torch.zeros((m - n) // 2, n)),
            dim=0,
        )
        .to(dtype)
        .to(device)
    )


def subsampling_matrix(m, n, dtype=torch.complex64, device="cpu"):
    """Generate a subsampling matrix of shape (m, n) with its left part being identity and the rest being zero"""
    assert m <= n, "m should be smaller than or equal to n"
    # return torch.cat((torch.eye(m), torch.zeros(m, n - m)), dim=1).to(dtype).to(device)
    # alternative way, make the center of the matrix identity
    # dimension is still m x n
    return (
        torch.cat(
            (torch.zeros(m, (n - m) // 2), torch.eye(m), torch.zeros(m, (n - m) // 2)),
            dim=1,
        )
        .to(dtype)
        .to(device)
    )


def diagonal_matrix(diag: torch.tensor, dtype=torch.complex64, device="cpu"):
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
    :param function transform_func: structured transform function. Default is :meth:`deepinv.physics.structured_random.dst1`.
    :param function transform_func_inv: structured inverse transform function. Default is :meth:`deepinv.physics.structured_random.dst1`.
    :param list diagonals: list of diagonal matrices. If None, a random :math:`{-1,+1}` mask matrix will be used. Default is None.
    :param str device: device of the physics. Default is 'cpu'.
    :param torch.Generator rng: Random number generator. Default is None.
    """

    def __init__(
        self,
        mode: str,
        input_shape: tuple,
        output_shape: tuple,
        middle_shape: tuple = None,
        n_layers=1,
        transform="dst1",
        transform_func=dst1,
        transform_func_inv=dst1,
        diagonals=None,
        spectrum=None,
        dtype=torch.complex64,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):

        self.dtype = dtype
        self.device = device

        self.input_shape = input_shape
        self.middle_shape = middle_shape
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.transform = transform
        self.transform_func = transform_func
        self.transform_func_inv = transform_func_inv
        self.diagonals = diagonals
        # default settings for fast compressed sensing
        if diagonals is None:
            diagonals = [
                generate_diagonal(
                    shape=input_shape,
                    mode="rademacher",
                    dtype=torch.float,
                    generator=rng,
                    device=device,
                )
            ]

        if spectrum is None:
            spectrum = generate_spectrum(
                shape=input_shape,
                mode="uniform",
                dtype=torch.float,
                generator=rng,
                device=device,
            )

        # forward operator
        def A(x):

            assert (
                x.shape[1:] == input_shape
            ), f"x doesn't have the correct shape {x.shape[1:]} != {input_shape}"

            if mode == "oversampling" or mode == "equisampling":
                x = x * spectrum

            if len(input_shape) == 3:
                x = padding(x, input_shape, middle_shape)

            if n_layers - math.floor(n_layers) == 0.5:
                x = transform_func(x)
            for i in range(math.floor(n_layers)):
                diagonal = diagonals[i]
                x = diagonal * x
                x = transform_func(x)

            if len(input_shape) == 3:
                x = trimming(x, middle_shape, output_shape)

            if mode == "undersampling":
                x = x * spectrum

            return x

        def A_adjoint(y):

            assert (
                y.shape[1:] == output_shape
            ), f"y doesn't have the correct shape {y.shape[1:]} != {output_shape}"

            if mode == "undersampling":
                y = y * torch.conj(spectrum)

            if len(input_shape) == 3:
                y = padding(y, output_shape, middle_shape)

            for i in range(math.floor(n_layers)):
                diagonal = diagonals[-i - 1]
                y = transform_func_inv(y)
                y = torch.conj(diagonal) * y
            if n_layers - math.floor(n_layers) == 0.5:
                y = transform_func_inv(y)

            if len(input_shape) == 3:
                y = trimming(y, middle_shape, input_shape)

            if mode == "oversampling" or mode == "equisampling":
                y = y * torch.conj(spectrum)

            return y

        super().__init__(A=A, A_adjoint=A_adjoint, **kwargs)

    def forward_matrix(self):
        """Given the structure of the operator, return the forward matrix."""
        m = np.prod(self.output_shape)
        p = np.prod(self.middle_shape)
        n = np.prod(self.input_shape)

        print("computing transform matrix")
        if self.transform == "fft":
            transform_matrix = dft_matrix(p, self.dtype, self.device)
        elif self.transform == "dct":
            transform_matrix = dct_matrix(p, self.dtype, self.device)
        elif self.transform == "hadamard":
            transform_matrix = hadamard_matrix(p, self.dtype, self.device)
        else:
            raise ValueError(f"Unsupported transform: {self.transform}")

        forward_matrix = torch.eye(n).to(self.dtype).to(self.device)
        print("computing oversampling")
        forward_matrix = (
            oversampling_matrix(p, n, self.dtype, self.device) @ forward_matrix
        )
        print("computing transform")
        if self.n_layers - math.floor(self.n_layers) == 0.5:
            forward_matrix = transform_matrix @ forward_matrix
        for i in range(math.floor(self.n_layers)):
            forward_matrix = (
                diagonal_matrix(self.diagonals[i].flatten()) @ forward_matrix
            )
            forward_matrix = transform_matrix @ forward_matrix
        print("computing undersampling")
        forward_matrix = (
            subsampling_matrix(m, p, self.dtype, self.device) @ forward_matrix
        )

        self.matrix = forward_matrix
        return forward_matrix

    def singular_values(self):
        """Compute the singular values of the forward matrix"""
        if self.forward_matrix is None:
            self.forward_matrix()
        s = sp.linalg.svdvals(self.matrix.cpu().numpy())
        return s
