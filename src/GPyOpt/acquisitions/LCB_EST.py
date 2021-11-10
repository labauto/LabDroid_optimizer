from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionLCB_EST(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=1,est_max=40):
        self.optimizer = optimizer
        super(AcquisitionLCB_EST, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        self.est_min = -1*est_max

        if cost_withGradients is not None:
            print('The set cost function is ignored! UCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Uower Confidence Bound 
        """
        m, s = self.model.predict(x)
        #est = np.maximum(0.1,(m - self.est_min)/s)
        weight = self.exploration_weight*(m - self.est_min)/s
        f_acqu = -1*m + weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Uower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = m + self.exploration_weight * s       
        df_acqu = dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu

