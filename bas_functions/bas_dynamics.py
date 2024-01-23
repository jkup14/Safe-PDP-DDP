import numpy as np
import jax.numpy as jnp
import jax


class BaSDynamics:
    """
    Discrete Barrier State dynamics. Implemented in jax. To be embedded into dynamics using EmbedDynamics class.
    Only tolerant barrier function is available here
    args:
    dynamics: model dynamics f(x, u, t)
    safety_function: h(x, k) -> array of dimension = num of constraints
    N: time horizon
    tol_params: tuple (p,m,c1,c2) for tolerant barrier function. Replace param with None to flag to PDP to optimize it.
    parameterized: boolean for if the barrier function is parameterized or not
    """

    def __init__(
        self,
        dynamics,
        safety_function,
        N,
        tol_params=[10, 10, 5, 5],
    ):
        self.dynamics = dynamics
        self.safety_function = safety_function
        self.N = N
        self.tol_params = tol_params

        self.parameterized = None in tol_params
        if self.parameterized:
            self.barrier = self.parameterized_tolerant_barrier_jax
        else:
            self.barrier = self.tolerant_barrier_jax
            self.p, self.m, self.c1, self.c2 = tol_params
            # BaS initial and final states
            self.bas_initial = self.barrier(jnp.array(np.squeeze(self.dynamics.x0)), 0)
            self.bas_terminal = self.barrier(
                jnp.array(np.squeeze(self.dynamics.xf)), N - 1
            )

    # tolerant barrier function, k optional
    def tolerant_barrier_jax(self, x, k=None):
        h = self.safety_function(x, k)
        beta = jnp.sum(
            self.p * sigmoid_jax(h, self.c1) + self.m * softplus_jax(h, self.c2)
        ).reshape(1)
        return beta

    # parameterized version of above function
    def parameterized_tolerant_barrier_jax(self, x, k, tol_params):
        p, m, c1, c2 = tol_params
        h = self.safety_function(x, k)
        beta = jnp.sum(p * sigmoid_jax(h, c1) + m * softplus_jax(h, c2)).reshape(1)
        return beta

    # Returns barrier of next state, unused
    def discrete_bas_dyn_jax(self, x, u, k):
        x_next = self.dynamics.system_propagate_jax(x[0 : self.dynamics.n], u, k)
        return self.barrier(x_next, k)

    # parameterized version of above function
    def parameterized_discrete_bas_dyn_jax(self, x, u, k, args):
        x_next = self.dynamics.system_propagate_jax(x[0 : self.dynamics.n], u, k)
        return self.barrier(x_next, k, args)


# sigmoid function, uses tanh to avoid numerical issues
def sigmoid_jax(x, c1):
    return 0.5 * (jnp.tanh(-x * c1 / 2) + 1)


# derivative of sigmoid
def dsigmoid_jax(x, c1):
    sig = sigmoid_jax(x, c1)
    return -sig * (1 - sig)


# softplus function, with numerical trick as well
def softplus_jax(x, c2):
    return (1 / c2) * jnp.log(1 + jnp.exp(0 - c2 * abs(x))) + jax.nn.relu(-x)


# derivative of softplus function
def dsoftplus_jax(x, c2):
    return -1 / (1 + jnp.exp(c2 * x))
