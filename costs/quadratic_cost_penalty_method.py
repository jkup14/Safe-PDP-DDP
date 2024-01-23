import numpy as np
import jax.numpy as jnp
import jax


class QuadraticCostPenaltyMethod:
    """
    Class for penalty method cost. Same cost as barrier state method but allows for unembedded state.
    args:
    Q: (n,n) running cost matrix, n is embedded state length
    R: (m,m) control cost matrix
    S: (n,n) terminal cost matrix, n is embedded state length
    n_state: unembedded system state dimension
    T: time horizon
    embed_func: embed function from EmbedDynamics object
    """

    def __init__(self, Q, R, S, n_state, T, embed_func):
        self.Q = Q
        self.R = R
        self.S = S
        self.T = T
        # embed function from embed_dynamics
        self.embed_func = embed_func
        self.n_state = n_state
        # For pdp
        self.cost_pdp = self.parameterized_cost

    # Regular cost
    def cost(self, x, u, k, xd):
        x = self.embed_func(x[: self.n_state], k)
        xd = xd.at[self.n_state :].set(0)
        e = x - xd
        l = 0.5 * jax.numpy.where(
            k == self.T, e.T @ self.S @ e, e.T @ self.Q @ e + u.T @ self.R @ u
        )
        return l

    # Parameterized version of above cost
    def parameterized_cost(self, x, u, k, xd, args):
        # Hacky way to get l to be a function of params, even though barrier state was already calculated in x
        x = self.embed_func(
            x[: self.n_state], k, args
        )  # throw out tdbas and make new one so that param tracers go through
        xd = xd.at[self.n_state :].set(0)
        e = x - xd
        l = 0.5 * jax.numpy.where(
            k == self.T, e.T @ self.S @ e, e.T @ self.Q @ e + u.T @ self.R @ u
        )
        return l
