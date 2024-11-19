import sys
import os

sys.path.append("/home/zhhu/workspaces/deepinv/")

from functools import partial

import numpy as np
import pandas as pd
import scipy as sp
import torch

import deepinv as dinv
from deepinv.optim.phase_retrieval import (
    cosine_similarity,
    generate_signal,
    spectral_methods,
)
from deepinv.physics.phase_retrieval import StructuredRandomPhaseRetrieval
from deepinv.physics.structured_random import MarchenkoPastur


def find_specturm_frequencies(input_shape, output_shape, n_bins=20, n_samples=100000):
    # given an oversampling ratio, get the histgram of the marchenko pastur distribution by sampling.
    # return the edges of the bins and the frequencies normalized to sum to 1
    distribution = MarchenkoPastur(m=np.prod(output_shape), n=np.prod(input_shape))
    samples = distribution.sample(n_samples)
    hist, bin_edges = np.histogram(samples, bins=n_bins, density=False)
    hist = hist / hist.sum()
    hist = np.log(hist)
    return hist, bin_edges


def sample_diagonal(hist, bin_edges, shape):
    # given the unsoftmaxed histogram and the bin edges, sample from the distribution assuming every bin is a uniform distribution
    assert (
        hist.shape[0] == bin_edges.shape[0] - 1
    ), "shapes of histogram and bin edges don't match"
    probabilities = sp.special.softmax(hist)
    # Step 1: Choose bins based on probabilities
    chosen_bins = np.random.choice(
        len(probabilities), size=np.prod(shape), p=probabilities
    )
    # Step 2: Sample uniformly within the chosen bin
    samples = np.empty(np.prod(shape))
    for i, bin_index in enumerate(chosen_bins):
        left_edge = bin_edges[bin_index]
        right_edge = bin_edges[bin_index + 1]
        samples[i] = np.random.uniform(left_edge, right_edge)

    return samples.reshape(shape)


def loss(x, bin_edges, input_shape, output_shape, n_repeats=10, device="cpu"):
    # given a decision x representing the histogram, sample from the distribution and compute the loss as average cosine similarity obstained by spectral methods
    score = 0

    img = generate_signal(
        img_size=input_shape[-1],
        mode="shepp-logan",
        transform=None,
        config={"unit_mag": True},
        device=device,
    )

    for _ in range(n_repeats):
        samples = sample_diagonal(x, bin_edges, output_shape)
        diagonal = torch.tensor(samples, dtype=torch.complex64)
        physics = StructuredRandomPhaseRetrieval(
            input_shape=input_shape,
            output_shape=output_shape,
            n_layers=2,
            diagonal_mode=[["custom", "uniform"], ["unit", "uniform"]],
            distri_config={"diagonal": diagonal},
            device=device,
        )
        y = physics(img)
        x_spec = spectral_methods(y, physics, verbose=False)
        score += cosine_similarity(x_spec, img)

    # score larger better
    print("current average cosine similarity:", (score / n_repeats).item())
    return (-score / n_repeats).item()


def main():

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    input_shape = (1, 64, 64)
    output_shape = (1, 78, 78)
    n_bins = 30

    hist, bin_edges = find_specturm_frequencies(
        input_shape=input_shape,
        output_shape=output_shape,
        n_bins=n_bins,
        n_samples=100000,
    )

    samples = sample_diagonal(hist, bin_edges, shape=output_shape)

    loss_func = partial(
        loss,
        bin_edges=bin_edges,
        input_shape=input_shape,
        output_shape=output_shape,
        n_repeats=10,
        device=device,
    )

    # Initialize a list to store x history
    x_history = []
    loss_history = []

    # Define a callback function to store the value of x at each iteration
    def callback(xk):
        x_history.append(np.copy(xk))  # Copy to avoid referencing the same array
        loss_history.append(loss_func(xk))

    solution = sp.optimize.minimize(
        fun=loss_func,
        x0=hist,
        method="Nelder-Mead",
        options={"maxiter": 1500},
        callback=callback,
    )

    loss_history = pd.DataFrame(loss_history, columns=["loss"])
    np.save("x_history.npy", x_history)
    loss_history.to_csv("loss_history.csv")

    print("solution:", solution.x)
    print("loss:", solution.fun)


if __name__ == "__main__":
    main()
