r"""
Unfolded Chambolle-Pock for constrained image inpainting
====================================================================================================

Image inpainting consists in solving :math:`y = Ax` where :math:`A` is a mask operator.
This problem can be reformulated as the following minimization problem:

.. math::

    \begin{equation*}
    \underset{x}{\operatorname{min}} \,\, \iota_{\mathcal{B}_2(y, r)}(Ax) + \regname(x)
    \end{equation*}


where :math:`\iota_{\mathcal{B}_2(y, r)}` is the indicator function of the ball of radius :math:`r` centered at
:math:`y` for the :math:`\ell_2` norm, and :math:`\regname` is a regularisation. Recall that the indicator function of
a convex set :math:`\mathcal{C}` is defined as :math:`\iota_{\mathcal{C}}(x) = 0` if :math:`x \in \mathcal{C}` and
:math:`\iota_{\mathcal{C}}(x) = +\infty` otherwise.

In this example, we unfold the Chambolle-Pock algorithm to solve this problem, and learn the thresholding parameters of
a wavelet denoiser in a LISTA fashion.

"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import deepinv as dinv
from deepinv.utils.demo import load_dataset
from deepinv.optim.data_fidelity import IndicatorL2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from deepinv.training_utils import train, test

# %%
# Setup paths for data loading and results.
# --------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset for training and the set3c dataset for testing.
# We work with images of size 32x32 if no GPU is available, else 128x128.


operation = "inpainting"
train_dataset_name = "CBSD68"
test_dataset_name = "set3c"
img_size = 128 if torch.cuda.is_available() else 32

test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)

train_base_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform=train_transform
)
test_base_dataset = load_dataset(
    test_dataset_name, ORIGINAL_DATA_DIR, transform=test_transform
)


# %%
# Define forward operator and generate dataset
# --------------------------------------------------------------------------------------------
# We define an inpainting operator that randomly masks pixels with probability 0.5.
#
# A dataset of pairs of measurements and ground truth images is then generated using the
# :meth:`dinv.datasets.generate_dataset` function.
#
# Once the dataset is generated, we can load it using the :class:`dinv.datasets.HDF5Dataset` class.

n_channels = 3  # 3 for color images, 1 for gray-scale images
probability_mask = 0.5  # probability to mask pixel

# Generate inpainting operator
physics = dinv.physics.Inpainting(
    (n_channels, img_size, img_size), mask=probability_mask, device=device
)


# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    100 if torch.cuda.is_available() else 50
)  # maximal number of images used for training
my_dataset_name = "demo_training_inpainting"
measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)


train_batch_size = 32 if torch.cuda.is_available() else 3
test_batch_size = 32 if torch.cuda.is_available() else 3

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Set up the reconstruction network
# --------------------------------------------------------
# We unfold the Chambolle-Pock algorithm as follows:
#
#      .. math::
#          \begin{equation*}
#          \begin{aligned}
#          u_{k+1} &= \operatorname{prox}_{\sigma d^*}(u_k + \sigma A z_k) \\
#          x_{k+1} &= \operatorname{D_{\theta}}(x_k-\tau A^\top u_{k+1}) \\
#          z_{k+1} &= 2x_{k+1} -x_k \\
#          \end{aligned}
#          \end{equation*}
#
# where :math:`\operatorname{D_{\sigma}}` is a wavelet denoiser with thresholding parameters :math:`\sigma`.
#
# The learnable parameters of our network are :math:`\tau` and :math:`\theta`.

# Select the data fidelity term
data_fidelity = IndicatorL2(radius=0.0)

# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each CP iteration.
# For fixed trained model prior across iterations, initialize with a single model.
max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
level = 3
prior = [
    PnP(denoiser=dinv.models.WaveletPrior(wv="db8", level=level, device=device))
    for i in range(max_iter)
]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.
sigma_denoiser = [0.01 * torch.ones(level, 3)] * max_iter

stepsize_dual = 1.0  # dual stepsize for Chambolle-Pock

# Define the parameters of the unfolded Primal-Dual Chambolle-Pock algorithm
# The CP algorithm requires to specify `params_algo`` the linear operator and its adjoint on which splitting is performed.
# See the documentation of the CP algorithm :meth:`deepinv.optim.optim_iterators.CPIteration` for more details.
params_algo = {
    "stepsize": stepsize,  # Stepsize for the primal update.
    "g_param": sigma_denoiser,  # prior parameter.
    "lambda": lamb,  # Regularization parameter.
    "stepsize_dual": stepsize_dual,  # The CP algorithm requires a second stepsize ``sigma`` for the dual update.
    "K": physics.A,
    "K_adjoint": physics.A_adjoint,
}

# define which parameters from 'params_algo' are trainable
trainable_params = ["g_param", "stepsize"]


# Define the unfolded trainable model.
model = unfolded_builder(
    iteration="CP",
    trainable_params=trainable_params,
    params_algo=params_algo.copy(),
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    g_first=False
)

# %%
# Train the model
# ---------------
# We train the model using the :meth:`dinv.training_utils.train` function.
#
# We perform supervised learning and use the mean squared error as loss function. This can be easily done using the
# :class:`dinv.loss.SupLoss` class.
#
# .. note::
#
#       In this example, we only train for a few epochs to keep the training time short on CPU.
#       For a good reconstruction quality, we recommend to train for at least 50 epochs.
#

epochs = 10 if torch.cuda.is_available() else 5  # choose training epochs
learning_rate = 1e-3

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

# choose training losses
losses = dinv.loss.SupLoss(metric=dinv.metric.mse())

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

train(
    model=model,
    train_dataloader=train_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
)

# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :meth:`dinv.training_utils.test` function.
#
# The testing function will compute test_psnr metrics and plot and save the results.

plot_images = True
method = "artifact_removal"

test_psnr, test_std_psnr, init_psnr, init_std_psnr = test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_folder=RESULTS_DIR / method / operation / test_dataset_name,
    verbose=verbose,
    wandb_vis=wandb_vis,  # training vialisations can be done in Weight&Bias
)

# %%
# Saving the model
# ----------------
# We can save the trained model following the standard PyTorch procedure.

# Save the model
torch.save(model.state_dict(), CKPT_DIR / operation / "model.pth")

# %%
# Loading the model
# -----------------
# Similarly, we can load our trained unfolded architecture following the standard PyTorch procedure.
# To check that the loading is performed correctly, we use new variables for the initialization of the model.

# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
level = 3
model_spec = {
    "name": "waveletprior",
    "args": {"wv": "db8", "level": level, "device": device},
}
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each PGD iteration.
# For fixed trained model prior across iterations, initialize with a single model.
max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
prior_new = [
    PnP(denoiser=dinv.models.WaveletPrior(wv="db8", level=level, device=device))
    for i in range(max_iter)
]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.
sigma_denoiser = [0.01 * torch.ones(level, 3)] * max_iter

stepsize_dual = 1.0  # stepsize for Chambolle-Pock

params_algo_new = {
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
    "stepsize_dual": stepsize_dual,
    "K": physics.A,
    "K_adjoint": physics.A_adjoint,
}

model_new = unfolded_builder(
    "CP",
    trainable_params=trainable_params,
    params_algo=params_algo_new,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior_new,
    g_first=False
)
model_new.load_state_dict(torch.load(CKPT_DIR / operation / "model.pth"))
model_new.eval()

# Test the model and check that the results are the same as before saving
test_psnr, test_std_psnr, init_psnr, init_std_psnr = test(
    model=model_new,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_folder=RESULTS_DIR / method / operation / test_dataset_name,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
