import sys
import os

sys.path.append("/home/zhhu/workspaces/deepinv/")

from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import torch
from tqdm import trange
import yaml

import deepinv as dinv
from deepinv.optim.data_fidelity import L2, AmplitudeLoss
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.phase_retrieval import (
    compute_lipschitz_constant,
    cosine_similarity,
    generate_signal,
)
from deepinv.optim.prior import Zero
from deepinv.physics import RandomPhaseRetrieval


def init_with(x_init):
    def func(y, physics):
        return {"est": (x_init, x_init)}

    return func


# load config
config_path = "../config/full_gd_rand.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# general
model_name = config["general"]["name"]
recon = config["general"]["recon"]
save = config["general"]["save"]
verbose = config["general"]["verbose"]
img_size = config["signal"]["shape"][-1]

# recon
n_repeats = config["recon"]["n_repeats"]

if config["recon"]["series"] == "arange":
    start = config["recon"]["start"]
    end = config["recon"]["end"]
    step = config["recon"]["step"]
    oversampling_ratios = torch.arange(start, end, step)
elif config["recon"]["series"] == "list":
    oversampling_ratios = torch.tensor(config["recon"]["list"])
else:
    raise ValueError("Invalid series type.")

loss = config["recon"]["gd"]["loss"]
if loss == "intensity":
    data_fidelity = L2()
elif loss == "amplitude":
    data_fidelity = AmplitudeLoss()
else:
    raise ValueError(f"Invalid data fidelity: {config['recon']['gd']['loss']}")
if config["recon"]["gd"]["prior"] == "zero":
    prior = Zero()
else:
    raise ValueError(f"Invalid prior: {config['recon']['gd']['prior']}")
early_stop = config["recon"]["gd"]["early_stop"]
max_iter = config["recon"]["gd"]["max_iter"]

# save
if save:
    res_name = config["save"]["name"].format(
        model_name=model_name,
        recon=recon,
    )
    print("res_name:", res_name)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    SAVE_DIR = Path(config["save"]["path"])
    SAVE_DIR = SAVE_DIR / current_time
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print("save directory:", SAVE_DIR)

    # save config
    shutil.copy(config_path, SAVE_DIR / "config.yaml")
    # read-only
    os.chmod(SAVE_DIR / "config.yaml", 0o444)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Set up the signal to be reconstructed.
x = generate_signal(
    shape=config["signal"]["shape"],
    mode=config["signal"]["mode"],
    config=config["signal"]["config"],
    dtype=torch.complex64,
    device=device,
)

df_res = pd.DataFrame(
    {
        "oversampling_ratio": oversampling_ratios,
        **{f"repeat{i}": None for i in range(n_repeats)},
    }
)

for i in trange(oversampling_ratios.shape[0]):
    oversampling_ratio = oversampling_ratios[i]
    print(f"oversampling_ratio: {oversampling_ratio}")
    for j in range(n_repeats):
        physics = RandomPhaseRetrieval(
            m=int(oversampling_ratio * img_size**2),
            img_shape=(1, img_size, img_size),
            mode=config["model"]["mode"],
            product=config["model"]["product"],
            unit_mag=config["model"]["unit_mag"],
            dtype=torch.complex64,
            device=device,
        )
        y = physics(x)

        x_init = torch.randn_like(x)

        step_size = compute_lipschitz_constant(
            x_init, y, physics, config["recon"]["gd"]["spectrum"], loss
        )
        params_algo = {"stepsize": 2 / step_size.item(), "g_params": 0.00}
        model = optim_builder(
            iteration="PGD",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=verbose,
            params_algo=params_algo,
            custom_init=init_with(x_init),
        )
        x_gd_spec = model(y, physics, x_gt=x)

        df_res.loc[i, f"repeat{j}"] = cosine_similarity(x, x_gd_spec).item()
        print(f"cosine similarity: {df_res.loc[i, f'repeat{j}']}")

        if save:
            df_res.to_csv(SAVE_DIR / res_name)

        physics.release_memory()

if save:
    print(f"Experiment {res_name} finished. Results saved at {SAVE_DIR}.")
