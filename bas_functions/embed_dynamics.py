import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import vmap


class EmbedDynamics:
    """Class for embedding a dynamics model with barrier states
    args:
    dynamics: dynamics object to embed with BaS, must have n, m, x0, xf, system_propagate_jax
    bas_dynamics:  list of BasDynamics objects to be embedded into dynamics
    N: time horizon
    """

    def __init__(self, dynamics, bas_dynamics, N):
        self.dynamics = dynamics
        self.bas_dynamics = bas_dynamics

        # Time Horizon
        self.N = self.bas_dynamics[0].N

        # Number of barrier states
        self.n_bas = len(bas_dynamics)

        # Collect parameter lists from each barrier state object, (n_bas, 4)
        self.parameter_list = [b.tol_params for b in bas_dynamics]
        # If any are None, then this is used for pdp, if not, this is just for ddp
        self.parameterized = any(b.parameterized for b in bas_dynamics)
        if self.parameterized:
            # How many parameters to optimize and where they are
            self.n_parameters, self.parameter_inds = self.parameterize_bas(
                self.parameter_list
            )
            self.system_propagate = self.parameterized_system_propagate_jax
            self.embed = self.parameterized_embed
        else:
            self.x0 = self.embed(dynamics.x0, 0)
            self.xf = self.embed(dynamics.xf, N)
            self.system_propagate = self.system_propagate_jax

        self.createEmbedTrajectory()
        self.n_model = dynamics.x0.shape[0]
        self.n = self.n_model + self.n_bas
        self.m = dynamics.m

    # Embed state x with barrier state(s)
    def embed(self, x, k):
        embedded_state = x
        for i in range(self.n_bas):
            bas_state = self.bas_dynamics[i].barrier(x, k)
            embedded_state = jnp.concatenate((embedded_state, bas_state), axis=0)
        return embedded_state

    # Create embed function for entire trajectory
    def createEmbedTrajectory(self):
        self.embed_traj = vmap(
            self.embed, in_axes=(0, 0) + (None,) if self.parameterized else ()
        )

    # Embed state x with barrier states, given parameters in args
    def parameterized_embed(self, x, k, args):
        assert (
            len(args) == self.n_parameters
        ), "Provided args should be a list of len=" + str(self.n_parameters)

        embedded_state = x
        for i in range(self.n_bas):
            # extracts parameters from args vec that are for this barrier state
            parameters_i = args[self.parameter_inds[i] : self.parameter_inds[i + 1]]
            # inputs them into wrapped barrier function that knows where to put them
            bas_state = self.bas_dynamics[i].barrier(x, k, parameters_i)
            embedded_state = jnp.concatenate((embedded_state, bas_state), axis=0)
        return embedded_state

    # Get number of parameters that are being optimized and where they are
    # Also replace barrier function so that parameters are placed in the right order
    def parameterize_bas(self, parameter_list):
        n_parameter_list = [0]  # so that cumsum gives the right indices
        assert (
            len(parameter_list) == self.n_bas
        ), "Length of parameter list must be equal to number of barrier states"
        self.n_bas_parameterized = self.n_bas
        for i, parameter_set in enumerate(parameter_list):
            sum_parameters = 0
            given_inds = []
            givens = []
            for ind, parameter in enumerate(parameter_set):
                if parameter is None:
                    sum_parameters += 1
                else:
                    given_inds += [ind]
                    givens += [parameter]
            # replace barrier function
            self.bas_dynamics[i].barrier = self.array_partial(
                self.bas_dynamics[i].barrier, given_inds, givens
            )
            n_parameter_list += [sum_parameters]
        return np.sum(n_parameter_list), np.cumsum(n_parameter_list)

    # Return function that takes in parameters being optimized and puts them into the correct order with parameters not being optimized
    def array_partial(self, func, given_inds, givens):
        def packed_func(x, k, params):
            for given_ind, given in zip(given_inds, givens):
                params = jnp.insert(params, given_ind, given)
            return func(x, k, params)

        return packed_func

    # Propogate x using u and embed barrier state
    def system_propagate_jax(self, x, u, k):
        system_state = self.dynamics.system_propagate_jax(x[0 : self.dynamics.n], u, k)
        embedded_state_next = self.embed(system_state, k)
        return embedded_state_next

    # Propogate x using u, embed barrier state
    def parameterized_system_propagate_jax(self, x, u, k, args):
        state_next = self.dynamics.system_propagate_jax(x[0 : self.dynamics.n], u, k)
        embedded_state_next = self.parameterized_embed(state_next, k, args)
        return embedded_state_next

    # # WIP, need a hmap
    # def checkCollision(self, X):
    #     h_traj = self.hmap(X[:, :self.n_model], )
    #     collided = jnp.any(h_traj <= 0)
    #     return collided
