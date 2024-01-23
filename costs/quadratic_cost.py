import numpy as np
import jax.numpy as jnp


class QuadraticCost:
    """
    Traditional quadratic cost
    args:
    Q: (n,n) running cost matrix
    R: (m,m) control cost matrix
    S: (n,n) terminal cost matrix
    T: time horizon
    """

    def __init__(self, Q, R, S, T):
        self.Q = Q
        self.R = R
        self.S = S
        self.T = T

    def cost_jax(self, x, u, k, xd):
        e = x - xd
        l = 0.5 * jnp.where(
            k == self.T, e @ self.S @ e.T, e @ self.Q @ e.T + u @ self.R @ u.T
        )
        return l

    def parameterized_cost_jax(self, x, u, k, xd, args):
        e = x - xd
        l = 0.5 * jnp.where(
            k == self.T, e @ self.S @ e.T, e @ self.Q @ e.T + u @ self.R @ u.T
        )
        return l

    def run_cost_grad(self, x, u, xd):
        e = x - xd
        l_x = self.Q @ e
        l_xx = self.Q
        l_u = self.R @ u
        l_uu = self.R
        l_ux = np.zeros([np.shape(self.R)[0], np.shape(self.Q)[0]])
        l_xu = l_ux.T
        return l_x, l_u, l_xx, l_xu, l_ux, l_uu

    # Running cost
    def run_cost(self, x, u, xd):
        e = x - xd
        l = 0.5 * (u.T @ self.R @ u + e.T @ self.Q @ e)
        return l

    # Terminal cost
    def term_cost(self, x, xf):
        e = x - xf
        phi = 0.5 * (e.T @ self.S @ e)
        return phi
