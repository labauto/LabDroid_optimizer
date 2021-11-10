from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionMV(AcquisitionBase):
    """
    GP Maximum Variance acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=1):
        self.optimizer = optimizer
        super(AcquisitionMV, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        m, s = self.model.predict(x)   
        f_acqu = self.exploration_weight * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu =  self.exploration_weight * s       
        df_acqu =  self.exploration_weight * dsdx
        return f_acqu, df_acqu