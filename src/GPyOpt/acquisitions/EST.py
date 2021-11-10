# Written by Shuhei Horiguchi

from .base import AcquisitionBase
from ..util.general import get_quantiles
import scipy.stats
import numpy as np
from ..experiment_design import LatinDesign

class AcquisitionEST(AcquisitionBase):
    """
    GP-EST acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, N_points=10000, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcquisitionEST, self).__init__(model, space, optimizer)

        self.N_points = N_points
        self.exploration_weight = 1.0 # update only when model.X is changed
        self._cached_X = np.zeros(0)

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        if not np.array_equal(self._cached_X, self.model.model.X):
            self.exploration_weight = compute_beta_EST(self.model, self.space, self.N_points)
            self._cached_X = self.model.model.X.copy()

        m, s = self.model.predict(x)
        f_acqu = -m + self.exploration_weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        if not np.array_equal(self._cached_X, self.model.model.X):
            self.exploration_weight = compute_beta_EST(self.model, self.space, self.N_points)
            self._cached_X = self.model.model.X.copy()

        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu




def compute_beta_EST(model, space, N_points=10000, points=None, numerical_integration=True):
    if points is None:
        points = LatinDesign(space).get_samples(N_points)

    if numerical_integration:
        min_est = estimate_min_numerical_integration(model, points)
    else:
        min_est = estimate_min_laplace_approximation(model, points)
    m, s = model.predict(points)
    beta = np.min((m - min_est) / s)
    assert beta > 0
    return beta

def estimate_min_laplace_approximation(model, points):
    raise NotImplementedError

def estimate_min_numerical_integration(model, points, binwidth=None, max_count=10000, verbose=False):
    means,stds = model.predict(points)

    if not binwidth:
        binwidth = max(0.01 * np.mean(stds), 1e-3)
        #print("binwidth:", binwidth)
    m0 = np.min(model.model.Y)
    m = m0
    logprodphi = []
    count = 0
    while count == 0 or logprodphi[-1] < -1e-9:
        logprodphi.append(np.sum(np.log(scipy.stats.norm.sf((m0-means)/stds))))
        m -= (1 - np.exp(logprodphi[-1])) * binwidth
        m0 -= binwidth
        count += 1
        if count > max_count:
            print("@EST (count > {})".format(max_count))
            if verbose:
                print("m0:",m0)
                print("m:",m)
                print("binwidth:",binwidth)
                print("logprodphi:",logprodphi)
            break
    if not verbose:
        return m
    else:
        return m, logprodphi
