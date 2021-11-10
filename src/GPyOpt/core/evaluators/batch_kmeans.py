# written by Shuhei Horiguchi

from .base import EvaluatorBase
from ...util.general import samples_multidimensional_uniform
import sampyl as smp
import numpy as np
import scipy.optimize
from sklearn.cluster import KMeans

list_sampler = [
    "slice",
    "nuts",
]

class KMBBO(EvaluatorBase):
    """
    Class for the batch method on 'Efficient and Scalable Batch Bayesian Optimization Using K-Means' (Groves et al., 2018).

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size, sampler="slice", N_sample=200, warmup=100, epsilon=1e-5, N_chain=1, max_resample=10, kmeans_after_expand=True, verbose=True):
        super(KMBBO, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

        assert sampler in list_sampler
        self.sampler = sampler
        self.n_sample = N_sample
        self.warmup = warmup
        self.epsilon = epsilon
        self.n_chains = N_chain
        self.max_resample = max_resample
        self.kmeans_after_expand = kmeans_after_expand
        self.verbose = verbose

    def compute_batch(self, duplicate_manager=None, context_manager=None, batch_context_manager=None):
        """
        Computes the elements of the batch.
        """
        assert not batch_context_manager or len(batch_context_manager) == self.batch_size
        if batch_context_manager:
            self.acquisition.optimizer.context_manager = batch_context_manager[0]
            raise NotImplementedError("batch_context is not supported")

        if not context_manager or context_manager.A_reduce is None:
            # not reduce dimension
            _expand = lambda x: x
            _reduce_d = lambda x: x
            f = lambda x: -self.acquisition.acquisition_function(x)[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(self.acquisition.space.get_bounds(), 1)[0,:]
            dimension = self.acquisition.space.dimensionality
            #print("not reduce: {} D".format(dimension))
        else:
            # reduce dimension
            _expand = lambda x: context_manager._expand_vector(x)
            _reduce_d = lambda x: context_manager._reduce_derivative(x)
            f = lambda x: -self.acquisition.acquisition_function(context_manager._expand_vector(x))[0,0]
            uniform_x = lambda : samples_multidimensional_uniform(context_manager.reduced_bounds, 1)[0,:]
            dimension = context_manager.space_reduced.dimensionality
            #print("do reduce: {} D".format(dimension))

        def is_valid(x):
            #print(x)
            #print(np.array(context_manager.noncontext_bounds))
            lower = np.alltrue(x > np.array(context_manager.noncontext_bounds)[:,0])
            upper = np.alltrue(x < np.array(context_manager.noncontext_bounds)[:,1])
            return lower and upper

        def _logp(x, fmin):
            x_ = _expand(x)
            p = -self.acquisition.acquisition_function(x_)[0,0]-fmin
            if not is_valid(x_):
                p = 0
            #print("p(", x, x_, ") =", p)
            lower_barrier = np.sum(np.log(
                np.clip(x_-np.array(context_manager.noncontext_bounds)[:,0], a_min=0, a_max=None)
            ))
            upper_barrier = np.sum(np.log(
                np.clip(np.array(context_manager.noncontext_bounds)[:,1] - x_, a_min=0, a_max=None)
            ))
            #print("lower_barrier:", lower_barrier)
            #print("upper_barrier:", upper_barrier)
            #logp = np.log(np.clip(p+lower_barrier+upper_barrier, a_min=0, a_max=None))
            logp = np.log(np.clip(p, a_min=0, a_max=None))
            #print("logp(", x, ") =", logp)
            return logp#p+lower_barrier+upper_barrier

        def _dlogp(x, fmin):
            x_ = _expand(x)
            p,dp = self.acquisition.acquisition_function_withGradients(x_)
            p = -p
            dp = _reduce_d(-dp)[0]
            if not is_valid(x_):
                dp *= 0
            #print("dp", x, x_, ") =", dp)
            lower_barrier = np.sum(np.log(
                np.clip(x_-np.array(context_manager.noncontext_bounds)[:,0], a_min=0, a_max=None)
            ))
            upper_barrier = np.sum(np.log(
                np.clip(np.array(context_manager.noncontext_bounds)[:,1] - x_, a_min=0, a_max=None)
            ))
            dlower_barrier = np.sum(1./np.clip(x_-np.array(context_manager.noncontext_bounds)[:,0], a_min=0, a_max=None))
            dupper_barrier = np.sum(1./np.clip(np.array(context_manager.noncontext_bounds)[:,1] - x_, a_min=0, a_max=None))
            #print("lower_barrier:", lower_barrier)
            #print("upper_barrier:", upper_barrier)
            #logp = np.log(np.clip(p[0]-fmin+lower_barrier+upper_barrier, a_min=0, a_max=None))
            logp = np.log(np.clip(p[0]-fmin, a_min=0, a_max=None))
            #dlogp = (dp+dlower_barrier+dupper_barrier) / logp
            dlogp = (dp) / logp
            #print("dlogp(", x, ") =", dlogp)
            return dlogp#dp+lower_barrier+upper_barrier#dlogp

        # first sample
        s0 = uniform_x()

        res = scipy.optimize.basinhopping(f, x0=s0, niter=100)
        acq_min = res.fun - self.epsilon
        #print("acq_min:",acq_min)

        # Now sample from x ~ p(x) = max(f(x) - acq_min, 0)
        # using No-U-Turn Sampler or Slice Sampler
        logp = lambda x: _logp(x, acq_min)
        dlogp = lambda x: _dlogp(x, acq_min)
        ok = False
        count = 0
        while not ok and count < self.max_resample:
            try:
                s0 = uniform_x()
                start = smp.find_MAP(logp, {'x': s0}, grad_logp=dlogp)
                #print("start:",start)
                if self.sampler == "slice":
                    s = smp.Slice(logp, start)#, grad_logp=dlogp)
                elif self.sampler == "nuts":
                    s = smp.NUTS(logp, start, grad_logp=dlogp)
                chain = s.sample(self.n_sample, burn=self.warmup, n_chains=self.n_chains, progress_bar=self.verbose)
                ok = True
            except Exception as e:
                #print("Exception:", e.args)
                ok = False
                count += 1
        if count == self.max_resample:
            if self.verbose: print("Maximum number of resample exceeded!")
            self.samples = np.array([uniform_x() for i in range(self.n_sample)])
        else:
            self.samples = chain.x

        # K-Means
        if self.kmeans_after_expand:
            km = KMeans(n_clusters=self.batch_size)
            km.fit(_expand(self.samples))
            self.km = km
            return km.cluster_centers_

        else:
            km = KMeans(n_clusters=self.batch_size)
            km.fit(self.samples)
            self.km = km

            return _expand(km.cluster_centers_)
