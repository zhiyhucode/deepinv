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
from deepinv.physics import RandomPhaseRetrieval, StructuredRandomPhaseRetrieval

# load config
config_path = "../config/structured_spec.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# general
model_name = config["general"]["name"]
recon = config["general"]["recon"]
save = config["general"]["save"]
verbose = config["general"]["verbose"]

# signal
shape = config["signal"]["shape"]
signal_mode = config["signal"]["mode"]
signal_config = config["signal"]["config"]
if signal_config is None:
    signal_config = {}

# model
img_size = config["signal"]["shape"][-1]
n_layers = config["model"]["n_layers"]
structure = StructuredRandomPhaseRetrieval.get_structure(n_layers)
transforms = config["model"]["transforms"]
diagonals = config["model"]["diagonals"]["mode"]
diagonal_config = config["model"]["diagonals"]["config"]
if diagonal_config is None:
    diagonal_config = {}
shared_weights = config["model"]["shared_weights"]
spectrum = config["model"]["spectrum"]["mode"]
pad_powers_of_two = config["model"]["pad_powers_of_two"]
include_zero = config["model"]["include_zero"]

# recon
n_repeats = config["recon"]["n_repeats"]
max_iter = config["recon"]["max_iter"]
## oversampling ratios
if config["recon"]["series"] == "arange":
    start = config["recon"]["start"]
    end = config["recon"]["end"]
    output_sizes = torch.arange(start, end, 2)
elif config["recon"]["series"] == "list":
    output_sizes = torch.tensor(config["recon"]["list"])
else:
    raise ValueError("Invalid series type.")
oversampling_ratios = output_sizes**2 / img_size**2
n_oversampling = oversampling_ratios.shape[0]

if save:
    res_name = config["save"]["name"].format(
        model_name=model_name,
        structure=structure,
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
if signal_mode == ["adversarial"]:
    pass
else:
    x = generate_signal(
        shape=shape,
        mode=signal_mode,
        config=signal_config,
        phase_range=(0, 2 * torch.pi),
        dtype=torch.complex64,
        device=device,
    )


df_res = pd.DataFrame(
    {
        "oversampling_ratio": oversampling_ratios,
        **{f"repeat{i}": None for i in range(n_repeats)},
    }
)

last_oversampling_ratio = -0.1

for i in trange(n_oversampling):
    oversampling_ratio = oversampling_ratios[i]

    if oversampling_ratio - last_oversampling_ratio < 0.05:
        continue
    # skip oversampling 1 as it takes too much time to sample
    if oversampling_ratio > 0.99 and oversampling_ratio < 1.01:
        continue

    output_size = output_sizes[i]
    print(f"output_size: {output_size}")
    print(f"oversampling_ratio: {oversampling_ratio}")
    for j in range(n_repeats):
        # * use spectrum from a full matrix
        if spectrum == "custom":
            example = RandomPhaseRetrieval(
                m=output_size**2,
                img_shape=(1, img_size, img_size),
                product=config["model"]["spectrum"]["product"],
                dtype=torch.complex64,
                device=device,
            )
            spectrum = torch.linalg.svdvals(example.B._A)
            # bootstrap the spectrum to have the dimension of img_size**2
            extra = img_size**2 - output_size**2
            if extra > 0:
                extra_indices = torch.randint(0, spectrum.numel(), (extra,))
                extra_spectrum = spectrum[extra_indices]
                spectrum = torch.cat((spectrum, extra_spectrum))
            # permute the spectrum
            spectrum = spectrum[torch.randperm(spectrum.numel())]
            spectrum = spectrum.reshape(1, img_size, img_size)
        # * model setup
        physics = StructuredRandomPhaseRetrieval(
            input_shape=(1, img_size, img_size),
            output_shape=(1, output_size, output_size),
            n_layers=n_layers,
            transforms=transforms,
            diagonals=diagonals,
            diagonal_config=diagonal_config,
            manual_spectrum=spectrum,
            pad_powers_of_two=pad_powers_of_two,
            shared_weights=shared_weights,
            include_zero=include_zero,
            dtype=torch.complex64,
            device=device,
            verbose=verbose,
        )
        # * adversarial signal
        if signal_mode == ["adversarial"]:
            signal_config["physics"] = physics
            x = generate_signal(
                shape=shape,
                mode=signal_mode,
                config=signal_config,
                dtype=torch.complex64,
            )

        y = physics(x)

        x_spec = spectral_methods(y, physics, n_iter=max_iter)
        df_res.loc[i, f"repeat{j}"] = cosine_similarity(x, x_spec).item()
        print(f"cosine similarity: {df_res.loc[i, f'repeat{j}']}")
        if save:
            df_res.to_csv(SAVE_DIR / res_name)

        physics.release_memory()
    last_oversampling_ratio = oversampling_ratio

if save:
    print(f"Experiment {res_name} finished. Results saved at {SAVE_DIR}.")
