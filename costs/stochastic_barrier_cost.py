import numpy as np
import jax
import jax.numpy as jnp
from bas_functions.bas_dynamics import (
    sigmoid_jax,
    softplus_jax,
    dsigmoid_jax,
    dsoftplus_jax,
)
from functools import partial
from costs.quadratic_cost import QuadraticCost

pad = lambda A: jnp.vstack((A, jnp.zeros((1,) + A.shape[1:])))


class StochasticBarrierCost:
    def __init__(
        self,
        hs: list,
        Q,
        R,
        S,
        T,
        n_state,
        barrier_params=[1, 100, 500, 500],
        risk_threshold=0.1,
    ):
        self.n_state = n_state
        self.Q = Q[: self.n_state, : self.n_state]
        self.R = R
        self.S = S[: self.n_state, : self.n_state]
        self.quad_cost = jax.vmap(
            QuadraticCost(self.Q, self.R, self.S, T).cost_jax, in_axes=(0, 0, 0, 0)
        )
        self.timesteps = np.arange(T + 1)
        self.T = T
        assert (
            risk_threshold < 1 and risk_threshold > 0
        ), "Your risk_threshold must be between 0 and 1"
        self.risk_threshold = risk_threshold
        self.p, self.m, self.c1, self.c2 = barrier_params
        self.barrier = lambda h: self.p * sigmoid_jax(
            h, self.c1
        ) + self.m * softplus_jax(
            h, self.c2
        )  # barrier func on total risk < threshold
        self.barrier_derivative = lambda h: self.p * dsigmoid_jax(
            h, self.c1
        ) + self.m * dsoftplus_jax(h, self.c2)
        self.getCostDerivs = jax.grad(self.cost, argnums=(0, 1))

        # self.norm_cdf = partial(norm_cdf, n)
        self.safety_functions = hs  # Stochastic safety function
        self.n_h = len(hs)

    @partial(jax.jit, static_argnums=(0,))
    def cost(self, X, U, xd):
        risk_traj = self.risk_of_traj(X) * 100
        bas_cost = jnp.sum(self.barrier(self.risk_threshold * 100 - risk_traj))
        # with jax.disable_jit(True):
        l_traj = jnp.sum(
            self.quad_cost(
                X[:, : self.n_state], pad(U), self.timesteps, xd[:, : self.n_state]
            )
        )
        return l_traj + bas_cost

    def risk_of_traj(self, X):
        # mu_h, sigma_h = jnp.hsplit(self.safety_function(X, self.timesteps), 2)
        mu_sigmas = [
            safety_function(X, self.timesteps)
            for safety_function in self.safety_functions
        ]
        mu_h, sigma_h = jnp.hstack([ms[0] for ms in mu_sigmas]), jnp.hstack(
            [ms[1] for ms in mu_sigmas]
        )
        cdfs = jax.scipy.stats.norm.cdf((-mu_h) / sigma_h) / (self.T)
        return jnp.sum(cdfs, axis=0)


# def norm_cdf(n, x):
#     ns = jnp.arange(n)
#     numerator = jnp.power(-1, ns)*jnp.power(x, 2*ns+1)
#     denominator = jnp.power(2,ns)*jnp.factor
#     return 0.5 + (1/jnp.sqrt(2*jnp.pi))*jnp.sum(numerator/denominator)

# l is either dependent on x[model.n:] or the args
