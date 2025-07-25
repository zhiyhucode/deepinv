from __future__ import annotations
from typing import Union
import torch
import math
import warnings
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.noise import NoiseModel, GaussianNoise, PoissonNoise, GammaNoise


class R2RLoss(Loss):
    r"""
    Generalized Recorrupted-to-Recorrupted (GR2R) Loss

    This self-supervised loss can be used when the noise model is
    Gaussian, Poisson, Gamma or Binomial. The GR2R loss is defined as:

    .. math::

        y_1 \sim p(y_1 \vert y, \alpha),

    where

    .. list-table::
       :header-rows: 1
       :widths: 40 60

       * - Noise Model
         - :math:`p(y_1 \vert y,\alpha)`
       * - :math:`y \sim \mathcal{N}(x, I\sigma^2)`
         - :math:`y_1 = y + \sqrt{\frac{\alpha}{1-\alpha}} \boldsymbol{\omega}, \quad \boldsymbol{\omega} \sim \mathcal{N}(0, I\sigma^2)`
       * - :math:`z \sim \mathcal{P}(x/\gamma), \quad y = \gamma z`
         - :math:`y_1 = \frac{y - \gamma \boldsymbol{\omega}}{1 - \alpha}, \quad \boldsymbol{\omega} \sim \mathrm{Bin}(z, \alpha)`
       * - :math:`y \sim \mathcal{G}(\ell, \ell / x)`
         - :math:`y_1 = y \circ (\mathbf{1} - \boldsymbol{\omega}) / (1 - \alpha), \quad \boldsymbol{\omega} \sim \mathrm{Beta}(\ell\alpha, \ell(1 - \alpha))`
       * - :math:`z \sim \mathrm{Bin}(\ell, x), \quad y = z / \ell`
         - :math:`y_1 = \frac{y - \boldsymbol{\omega} / \ell}{1 - \alpha}, \quad \boldsymbol{\omega} \sim \mathrm{HypGeo}(\ell, \ell\alpha, z)`

    and

    .. math::

        y_2 = \frac{1}{\alpha} \left( y - y_1(1-\alpha) \right),

    then, the loss is computed as:

    .. math::

        \| AR(y_1) - y_2 \|_2^2,

    where, :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement, and :math:`\alpha` is a scaling factor.

    The loss was first introduced by :footcite:t:`pang2021recorrupted`
    for the specific case of Gaussian noise, formalizing the Noise2Noisier loss from :footcite:t:`moran2020noisier2noise`,
    such that it is statistically equivalent to the supervised loss function defined on noisy/clean image pairs.
    The loss was later extended to other exponential family noise distributions by :footcite:t:`monroy2025generalized`, including Poisson,
    Gamma and Binomial noise distributions.

    .. warning::

        The model should be adapted before training using the method :meth:`adapt_model` to include the additional noise at the input.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the added noise, i.e. :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N R(y_1^{(i)})`,
        where :math:`N>1`. This can be achieved using :meth:`adapt_model`.

    .. note::

        If the ``noise_model`` parameter is not provided, the noise model from the physics module will be used.

    .. deprecated:: 0.2.3

        The ``sigma`` paramater is deprecated and will be removed in future versions. Use ``noise_model=deepinv.physics.GaussianNoise(sigma=sigma)`` parameter instead.

    :param Metric, torch.nn.Module metric: Metric for calculating loss, defaults to MSE.
    :param NoiseModel noise_model: Noise model of the natural exponential family, defaults to None. Implemented options are :class:`deepinv.physics.GaussianNoise`, :class:`deepinv.physics.PoissonNoise` and :class:`deepinv.physics.GammaNoise`
    :param float alpha: Scaling factor of the corruption.
    :param int eval_n_samples: Number of samples used for the Monte Carlo approximation.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> sigma = 0.1
    >>> noise_model = dinv.physics.GaussianNoise(sigma)
    >>> physics = dinv.physics.Denoising(noise_model)
    >>> model = dinv.models.MedianFilter()
    >>> loss = dinv.loss.R2RLoss(noise_model=noise_model, eval_n_samples=2)
    >>> model = loss.adapt_model(model) # important step!
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics, update_parameters=True) # save extra noise in forward pass
    >>> l = loss(x_net, y, physics, model)
    >>> print(l.item() > 0)
    True

    """

    def __init__(
        self,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        noise_model: NoiseModel = None,
        alpha=0.15,
        sigma=None,
        eval_n_samples=5,
    ):
        super(R2RLoss, self).__init__()
        self.name = "gr2r"
        self.metric = metric
        self.alpha = alpha
        self.eval_n_samples = eval_n_samples
        self.noise_model = noise_model

        if sigma is not None:

            warnings.warn(
                "The sigma parameter is deprecated and will be removed in future versions. "
                "Please use the noise_model parameter instead."
            )

            self.noise_model = GaussianNoise(sigma)

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the GR2R Loss.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction model.
        :return: (:class:`torch.Tensor`) R2R loss.
        """

        y1 = model.get_corruption()
        y2 = (1 / self.alpha) * (y - y1 * (1 - self.alpha))
        return self.metric(physics.A(x_net), y2)

    def adapt_model(self, model, **kwargs):
        r"""
        Adds noise to model input.


        This method modifies a reconstruction
        model :math:`R` to include the re-corruption mechanism at the input:

        .. math::

            \hat{R}(y) = \frac{1}{N}\sum_{i=1}^N R(y_1^{(i)}),

        where :math:`y_1^{(i)} \sim p(y_1 \vert y, \alpha)` are i.i.d samples, and :math:`N\geq 1` are the number of samples used for the Monte Carlo approximation.
        During training (i.e. when ``model.train()``), we use only one sample, i.e. :math:`N=1`
        for computational efficiency, whereas at test time, we use multiple samples for better performance.

        :param torch.nn.Module model: Reconstruction model.
        :param NoiseModel noise_model: Noise model of the natural exponential family.
            Implemented options are :class:`deepinv.physics.GaussianNoise`, :class:`deepinv.physics.PoissonNoise` and :class:`deepinv.physics.GammaNoise`
        :param float alpha: Scaling factor of the corruption.
        :return: (:class:`torch.nn.Module`) Modified model.
        """

        if isinstance(model, R2RModel):
            return model
        else:
            return R2RModel(
                model, self.noise_model, self.alpha, self.eval_n_samples, **kwargs
            )


def set_gaussian_corruptor(y, alpha, sigma):
    mu = torch.ones_like(y) * 0.0
    sigma = torch.ones_like(y) * sigma
    sampler = torch.distributions.Normal(mu, sigma)
    corruptor = lambda: y + sampler.sample() * (math.sqrt(alpha / (1 - alpha)))
    return corruptor


def set_binomial_corruptor(y, alpha, gamma):
    z = y / gamma
    sampler = torch.distributions.Binomial(torch.round(z), alpha)
    corruptor = lambda: gamma * (z - sampler.sample()) / (1 - alpha)
    return corruptor


def set_beta_corruptor(y, alpha, l):
    tmp = torch.ones_like(y)
    concentration1 = tmp * l * alpha
    concentration0 = tmp * l * (1 - alpha)
    sampler = torch.distributions.Beta(concentration1, concentration0)
    corruptor = lambda: y * (1 - sampler.sample()) / (1 - alpha)
    return corruptor


class R2RModel(torch.nn.Module):
    r"""
    Generalized Recorrupted-to-Recorrupted (GR2R) Model

    """

    def __init__(self, model, noise_model, alpha, eval_n_samples):
        super(R2RModel, self).__init__()
        self.model = model
        self.noise_model = noise_model
        self.eval_n_samples = eval_n_samples
        self.alpha = alpha

    def forward(self, y, physics, update_parameters=False):

        if self.noise_model is None:

            if isinstance(physics.noise_model, NoiseModel):
                self.curr_noise_model = physics.noise_model
            else:
                raise ValueError(
                    "Noise model not found in physics module. Please provide noise model in the constructor."
                )

        elif isinstance(self.noise_model, NoiseModel):
            self.curr_noise_model = self.noise_model
        else:
            raise ValueError(
                "Noise model not found in the constructor or physics module. Please provide noise model in the constructor."
            )

        eval_n_samples = 1 if self.training else self.eval_n_samples
        out = 0
        corruptor = self.get_corruptor(y)
        self.curr_noise_model = None

        with torch.set_grad_enabled(self.training):
            for i in range(eval_n_samples):
                y1 = corruptor()
                out += self.model(y1, physics)

            if self.training and update_parameters:
                self.corruption = y1

            out = out / eval_n_samples
        return out

    def get_corruptor(self, y):
        alpha = self.alpha

        if isinstance(self.curr_noise_model, GaussianNoise):

            sigma = self.curr_noise_model.sigma
            return set_gaussian_corruptor(y, alpha, sigma)

        elif isinstance(self.curr_noise_model, PoissonNoise):

            gain = self.curr_noise_model.gain
            return set_binomial_corruptor(y, alpha, gain)

        elif isinstance(self.curr_noise_model, GammaNoise):

            l = self.curr_noise_model.l
            return set_beta_corruptor(y, alpha, l)

        else:
            raise ValueError(
                f"Noise model {self.curr_noise_model} not supported, available options are Gaussian, Poisson and Gamma."
            )

    def get_corruption(self):
        return self.corruption
