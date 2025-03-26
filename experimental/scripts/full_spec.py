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
from deepinv.optim.phase_retrieval import (
    cosine_similarity,
    spectral_methods,
    generate_signal,
)
from deepinv.physics import RandomPhaseRetrieval

# load config
config_path = "../config/full_spec.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# general
model_name = config["general"]["name"]
recon = config["general"]["recon"]
save = config["general"]["save"]
img_size = config["signal"]["shape"][-1]

# recon
n_repeats = config["recon"]["n_repeats"]
max_iter = config["recon"]["max_iter"]

if config["recon"]["series"] == "arange":
    start = config["recon"]["start"]
    end = config["recon"]["end"]
    step = config["recon"]["step"]
    oversampling_ratios = torch.arange(start, end, step)
elif config["recon"]["series"] == "list":
    oversampling_ratios = torch.tensor(config["recon"]["list"])
else:
    raise ValueError("Invalid series type.")

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

        x_spec = spectral_methods(y, physics, n_iter=max_iter)
        df_res.loc[i, f"repeat{j}"] = cosine_similarity(x, x_spec).item()
        # print the cosine similarity
        print(f"cosine similarity: {df_res.loc[i, f'repeat{j}']}")

        physics.release_memory()

# save results
if save:
    df_res.to_csv(SAVE_DIR / res_name)
