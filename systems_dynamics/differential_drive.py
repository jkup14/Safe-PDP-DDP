import numpy as np

# from jax import jit
from functools import partial
import jax.numpy as jnp


class DifferentialDrive:
    def __init__(self, dt):
        self.n = 3
        self.m = 2
        self.dt = dt  # time discretization
        self.rad = 0.2  # radius of wheels
        self.width = 0.2  # distance between wheels

    def system_dyn(self, state, control):
        x, y, theta = state.item(0), state.item(1), state.item(2)
        u_r, u_l = control.item(0), control.item(1)

        f = np.array(
            [
                [self.rad * (u_r + u_l) / 2 * np.cos(theta)],
                [self.rad * (u_r + u_l) / 2 * np.sin(theta)],
                [self.rad * (u_r - u_l) / (2 * self.width)],
            ]
        )
        return f

    def system_dyn_jax(self, state, control):
        theta = state[2]
        u_r, u_l = control

        dxdt = jnp.array(
            [
                self.rad * (u_r + u_l) / 2 * jnp.cos(theta),
                self.rad * (u_r + u_l) / 2 * jnp.sin(theta),
                self.rad * (u_r - u_l) / (2 * self.width),
            ]
        )
        return dxdt

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f  # euler integration
        return state_next

    def system_propagate_jax(self, state, control, k):
        f = self.system_dyn_jax(state, control)
        state_next = state + self.dt * f.T  # euler integration
        return state_next

    def system_grad(self, state, control, k):
        x, y, theta = state.item(0), state.item(1), state.item(2)
        u_r, u_l = control.item(0), control.item(1)

        fx = np.eye(self.n) + self.dt * np.array(
            [
                [0.0, 0.0, -np.sin(theta) * self.rad * (u_r + u_l) / 2],
                [0.0, 0.0, np.cos(theta) * self.rad * (u_r + u_l) / 2],
                [0.0, 0.0, 0.0],
            ]
        )

        fu = self.dt * np.array(
            [
                [(self.rad * np.cos(theta) / 2), self.rad * np.cos(theta) / 2],
                [self.rad * np.sin(theta) / 2, self.rad * np.sin(theta) / 2],
                [self.rad / (2 * self.width), -self.rad / (2 * self.width)],
            ]
        )
        return fx, fu
