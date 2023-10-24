import torch

from .optim_iterator import OptimIterator, fStep, gStep


class DRSIteration(OptimIterator):
    r"""
    Single iteration of DRS.

    Class for a single iteration of the Douglas-Rachford Splitting (DRS) algorithm for minimising
    :math:`\lambda f(x) + g(x)`.

    If the attribute ``g_first`` is set to False (by default), the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma \lambda f}(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(2*u_{k+1}-z_k) \\
        z_{k+1} &= z_k + \beta (x_{k+1} - u_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.
    Here, :math:`z_k` is the iterate i.e. the fixed point variable iterated by the algorithm and :math:`u_k` is the estimate i.e. the estimation of the solution of the minimization problem.

    If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`g` are inverted in the previous iteration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)
        self.requires_prox_g = True

    def get_minimizer_from_FP(self, x, cur_data_fidelity, cur_prior, cur_params, y, physics):
        """
        Get the minimizer of F from the fixed point variable x.

        :param torch.Tensor x: Fixed point variable iterated by the algorithm.
        :return: Minimizer of F.
        """
        if self.g_first:
            return self.g_step(x, x, cur_prior, cur_params, y, physics)
        else:
            return self.f_step(x, x, cur_data_fidelity, cur_params, y, physics)
    

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the DRS algorithm.

        :param dict X: Dictionary containing the current iterate, current estimate and cost at the current estimate.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{'fp' : x,  'est': z , 'cost': F}` containing the updated iterate, estimate and cost value.
        """
        z = X["fp"]
        if self.g_first:
            u = self.g_step(z, z, cur_prior, cur_params)
            x = self.f_step(u, z, cur_data_fidelity, cur_params, y, physics)
        else:
            u = self.f_step(z, z, cur_data_fidelity, cur_params, y, physics)
            x = self.g_step(u, z, cur_prior, cur_params)
        z = z + cur_params["beta"] * (x - u)
        est = self.get_minimizer_from_FP(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
        F = (
            self.F_fn(est, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"fp" : x, "est": est, "cost": F}


class fStepDRS(fStep):
    r"""
    DRS fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current first variable.
        :param torch.Tensor z: Current second variable.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        if self.g_first:
            p = 2 * x - z
        else:
            p = z
        return cur_data_fidelity.prox(
            p, y, physics, cur_params["lambda"] * cur_params["stepsize"]
        )


class gStepDRS(gStep):
    r"""
    DRS gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x:  Current first variable.
        :param torch.Tensor z: Current second variable.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        if self.g_first:
            p = z
        else:
            p = 2 * x - z
        return cur_prior.prox(p, cur_params["stepsize"], cur_params["g_param"])
