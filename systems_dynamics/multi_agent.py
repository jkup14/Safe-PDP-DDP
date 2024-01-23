import numpy as np
import random as r
import math
import operator
import jax.numpy as jnp


class Multi_Agent:
    def __init__(
        self,
        dynamics,
        n_agents,
        x0=None,
        xf=None,
        arrange=False,
        width=None,
        height=None,
        sd=None,
        seed=None,
        pointed=False,
        space_dim="2D",
    ):
        self.dynamics = dynamics
        self.n_agents = n_agents
        self.n = dynamics.n * self.n_agents
        self.m = dynamics.m * self.n_agents
        self.n1 = dynamics.n
        self.m1 = dynamics.m
        self.dt = dynamics.dt
        assert (
            x0 is not None and xf is not None
        ) ^ arrange, "Either provide x0 and xf or set arrange to true for a elliptical arrangement"

        if x0 is not None:
            self.x0 = x0
            self.xf = xf
        else:
            if space_dim == "2D":
                if pointed:
                    self.x0, self.xf = arrange_agents_ellipse_pointed(
                        n_agents, self.n1, width, height, sd, seed
                    )
                else:
                    self.x0, self.xf = arrange_agents_ellipse(
                        n_agents, self.n1, width, height, sd, seed
                    )

    def system_dyn(self, state, control):
        f = np.zeros((self.n, 1))
        for ii in range(self.n_agents):
            f[ii * self.n1 : (ii + 1) * self.n1] = self.dynamics.system_dyn(
                state[ii * self.n1 : (ii + 1) * self.n1],
                control[ii * self.m1 : (ii + 1) * self.m1],
            )
        return f

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control, k)
        state_next = state + self.dt * f
        return state_next

    def system_propagate_jax(self, state, control, k):
        f = self.system_dyn_jax(state, control)
        state_next = state + self.dt * f.T
        return state_next

    def system_dyn_jax(self, state, control):
        f = jnp.zeros(self.n)
        for ii in range(self.n_agents):
            f = f.at[ii * self.n1 : (ii + 1) * self.n1].set(
                self.dynamics.system_dyn_jax(
                    state[ii * self.n1 : (ii + 1) * self.n1],
                    control[ii * self.m1 : (ii + 1) * self.m1],
                )
            )
        return f

    def system_grad(self, state, control, k):
        fx = np.zeros((self.n, self.n))
        fu = np.zeros((self.n, self.m))
        for ii in range(self.n_agents):
            fxi, fui = self.dynamics.system_grad(
                state[ii * self.n1 : (ii + 1) * self.n1],
                control[ii * self.m1 : (ii + 1) * self.m1],
                k,
            )
            fx[
                ii * self.n1 : (ii + 1) * self.n1, ii * self.n1 : (ii + 1) * self.n1
            ] = fxi
            fu[
                ii * self.n1 : (ii + 1) * self.n1, ii * self.m1 : (ii + 1) * self.m1
            ] = fui
        return fx, fu


def arrange_agents_ellipse(n_agents, n1, width, height, sd, seed):
    x0 = np.zeros((n_agents * n1, 1))
    xf = np.zeros((n_agents * n1, 1))
    for ii in range(n_agents):
        x0[ii * n1 : ii * n1 + 2] = np.array(
            [
                [r.random() * sd + width * math.cos(ii * 2 * math.pi / n_agents)],
                [r.random() * sd + height * math.sin(ii * 2 * math.pi / n_agents)],
            ]
        )
        xf[ii * n1 : ii * n1 + 2] = np.array(
            [[r.random() * sd - x0[ii * n1, 0]], [r.random() * sd - x0[ii * n1 + 1, 0]]]
        )
    return x0, xf


def arrange_agents_ellipse_pointed(n_agents, n1, width, height, sd, seed):
    x0 = np.zeros((n_agents * n1, 1))
    xf = np.zeros((n_agents * n1, 1))
    for ii in range(n_agents):
        x0[ii * n1 : ii * n1 + 2] = np.array(
            [
                [r.random() * sd + width * math.cos(ii * 2 * math.pi / n_agents)],
                [r.random() * sd + height * math.sin(ii * 2 * math.pi / n_agents)],
            ]
        )
        xf[ii * n1 : ii * n1 + 2] = np.array(
            [[r.random() * sd - x0[ii * n1, 0]], [r.random() * sd - x0[ii * n1 + 1, 0]]]
        )
        x0[ii * n1 + 2] = np.arctan2(
            xf[ii * n1 + 1] - x0[ii * n1 + 1], xf[ii * n1] - x0[ii * n1]
        )
        xf[ii * n1 + 2] = x0[ii * n1 + 2]
    return x0, xf


def arrange_agents_rectangle(n_agents, n1, agent_radius, sd, seed):
    r.seed(seed)
    x0 = np.array(
        [-0.9],
        [-0.9],
        [-0.5],
        [-0.9],
        [0],
        [-0.9],
        [0.5],
        [-0.9],
        [0.9],
        [-0.9],
        [-0.9],
        [0.9],
        [-0.5],
        [0.9],
        [0],
        [0.9],
        [0.5],
        [0.9],
        [0.9],
        [0.9],
    )
    xf = np.array(
        [0.5],
        [0.9],
        [0.9],
        [0.9],
        [-0.5],
        [0.9],
        [0],
        [0.9],
        [-0.9],
        [0.9],
        [0.5],
        [-0.9],
        [0.9],
        [-0.9],
        [-0.9],
        [-0.9],
        [0],
        [-0.9],
        [-0.5],
        [-0.9],
    )


def check_collision(n_agents, X, n1, delta):
    collided = False
    for ii in range(n_agents):
        for jj in range(ii + 1, n_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            # ax[1].plot(T, dists_ij)
            if (np.array(dists_ij) < delta).any():
                collided = True
    print("Collided?:", collided)
    return collided
