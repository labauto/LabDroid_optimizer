# Copyright (c) 2019, Shuhei Horiguchi

from .base import EvaluatorBase
import numpy as np
from ...experiment_design import initial_design
from ...acquisitions.EST import compute_beta_EST
from dppy.finite_dpps import FiniteDPP
from ...core.task.space import Design_space
from ...optimization.acquisition_optimizer import ContextManager

class DPP(EvaluatorBase):
    """
    Class for the batch method on 'Efficient and Scalable Batch Bayesian Optimization Using K-Means' (Groves et al., 2018).

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size, base_points=None, N_points=10000, design_name="random", suppress_emb=False, randomize=False, verbose=False):
        super(DPP, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.space = acquisition.space
        self.design_name = design_name
        self.suppress_emb = suppress_emb
        self.randomize = randomize
        self.verbose = verbose

        if base_points is None:
            self.N_points = N_points
            self.base_points = None
        else:
            assert base_points.shape[1] == acquisition.model.input_dim
            self.base_points = base_points
            self.N_points = base_points.shape[0]

    def sample_base_points(self, N_points=None, context_manager=None,
                           randomize=None, random_bound_axes=False,
                           max_rejection_low2high=10):
        if N_points is None:
            N_points = self.N_points
        if randomize is None:
            randomize = self.randomize

        kern = self.acquisition.model.model.kern

        if self.base_points is None:
            """
            if not self.suppress_emb and hasattr(kern, "emb_min") and hasattr(kern, "emb_max"):
                # if kernel is LinEmbKern, base_points are placed in reduced subspace
                base_points = kern.sample_X_uniform_on_emb(N_points, randomize_low2high=randomize,
                                                           random_bound_axes=random_bound_axes,
                                                           max_rejection_low2high=max_rejection_low2high,
                                                           max_rejection_Z=1000)
            """
            
            #elif
            if not self.suppress_emb and context_manager and context_manager.A_reduce is not None:
                # if context is specified, base_points are placed in reduced subspace
                base_points_emb = initial_design(self.design_name, context_manager.space_reduced, N_points)
                base_points = context_manager._expand_vector(base_points_emb)
            else:
                # base_points are placed in original space
                base_points = initial_design(self.design_name, self.space, N_points)
        else:
            base_points = self.base_points
        return base_points

    def compute_batch(self, duplicate_manager=None, context_manager=None, batch_context_manager=None):
        """
        Computes the elements of the batch.
        """
        assert not batch_context_manager or len(batch_context_manager) == self.batch_size
        if batch_context_manager:
            self.acquisition.optimizer.context_manager = batch_context_manager[0]
            raise NotImplementedError("batch_context is not supported")


        model = self.acquisition.model

        N_points = self.N_points
        base_points = np.empty((0, model.input_dim))
        beta = None
        # get points in relevance region as many as batch size
        while True:
            # sample base points
            base_points = np.concatenate([
                base_points,
                self.sample_base_points(N_points, context_manager)
            ])

            # first point is greedy
            X_batch = np.empty((self.batch_size, model.input_dim))
            acq_on_points = self.acquisition.acquisition_function(base_points)
            X_batch[0] = base_points[np.argmin(acq_on_points)] #self.acquisition.optimize()[0]
            if self.verbose: print("first point:", X_batch[0])
            if self.batch_size == 1:
                return X_batch

            # to get posterior covariance after first point selection
            model_post = model.model.copy()
            X_ = np.vstack([model_post.X, X_batch[0]])
            Y_ = np.vstack([model_post.Y, [0]]) #0 is arbitrary
            model_post.set_XY(X_, Y_)

            # using beta from EST
            if beta is None:
                beta = compute_beta_EST(model=model,
                                        space=self.acquisition.space,
                                        points=base_points)
                if self.verbose: print("beta:", beta)
            m, s = model.predict(base_points)
            ucb  = (m +   beta*s).flatten()
            lcb2 = (m - 2*beta*s).flatten()
            in_relevance_region = lcb2 < np.min(ucb)
            num_relevance_points = np.count_nonzero(in_relevance_region)
            if self.verbose: print("num_points:", num_relevance_points, "/", N_points)
            if num_relevance_points < self.batch_size:
                N_points *= self.batch_size
                continue

            _, cov_post = model_post.predict(
                base_points[in_relevance_region], full_cov=True)

            # sample rest of points by DPP
            noise = model.model.Gaussian_noise.variance.values[0]
            dpp_kernel = np.eye(num_relevance_points)+noise**(-2)*cov_post
            if self.verbose: print("rank:",np.linalg.matrix_rank(dpp_kernel))
            try:
                samples_dpp = sample_dpp(dpp_kernel, self.batch_size-1, T=1)
                break
            except ValueError:
                if self.verbose: print("size k={} > rank".format(self.batch_size-1))
                N_points *= self.batch_size
                continue


        X_batch[1:] = base_points[samples_dpp]
        return X_batch


def sample_dpp(L, B, T=1):
    dpp = FiniteDPP("likelihood", L=L)

    dpp.flush_samples()
    for _ in range(T):
        dpp.sample_exact_k_dpp(size=B)

    if T == 1:
        return dpp.list_of_samples[0]
    else:
        return dpp.list_of_samples
