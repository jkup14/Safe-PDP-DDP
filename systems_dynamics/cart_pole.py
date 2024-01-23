import numpy as np
import jax.numpy as jnp


class CartPole:
    def __init__(self, dt):
        self.n = 4
        self.m = 1
        self.dt = dt
        self.mp = 0.05  # pole mass (kg)
        self.mc = 1  # cart mass (kg)
        self.g = 9.8  # gravitational force (m/s^2)
        self.l = 2  # pole length (m)
        self.x0 = np.array([[0], [0], [0], [0]])
        self.xf = np.array([[0], [3.14], [0], [0]])

    def system_dyn(self, state, control):
        x = state.item(0)  # cart position
        theta = state.item(1)  # pole angle
        xdot = state.item(2)  # cart velocity
        thetadot = state.item(3)  # pole angular velocity
        u = control.item()  # force

        f = np.array(
            [
                [xdot],
                [thetadot],
                [
                    (
                        self.mp
                        * np.sin(theta)
                        * (self.l * thetadot**2 + self.g * np.cos(theta))
                        + u
                    )
                    / (self.mc + self.mp * (np.sin(theta)) ** 2)
                ],
                [
                    (
                        -self.mp
                        * self.l
                        * thetadot**2
                        * np.cos(theta)
                        * np.sin(theta)
                        - (self.mc + self.mp) * self.g * np.sin(theta)
                        - np.cos(theta) * u
                    )
                    / (self.l * (self.mc + self.mp * (np.sin(theta)) ** 2))
                ],
            ]
        )
        return f

    def system_propagate(self, state, control):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f
        return state_next

    def system_propogate_jax(self, state, control, k):
        return _system_propagate_jax(
            self.dt, self.mp, self.l, self.mc, self.g, state, control, k
        )

    def system_grad(self, state, control):
        x = state.item(0)  # cart position
        theta = state.item(1)  # pole angle
        xdot = state.item(2)  # cart velocity
        thetadot = state.item(3)  # pole angular velocity
        u = control.item()  # force

        fx = np.eye(self.n) + self.dt * np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [
                    0.0,
                    (
                        self.mp
                        * np.cos(theta)
                        * (self.l * thetadot**2 + self.g * np.cos(theta))
                        - self.g * self.mp * np.sin(theta) ** 2
                    )
                    / (self.mp * np.sin(theta) ** 2 + self.mc)
                    - (
                        2
                        * self.mp
                        * np.cos(theta)
                        * np.sin(theta)
                        * (
                            u
                            + self.mp
                            * np.sin(theta)
                            * (self.l * thetadot**2 + self.g * np.cos(theta))
                        )
                    )
                    / (self.mp * np.sin(theta) ** 2 + self.mc) ** 2,
                    0.0,
                    (2 * self.l * self.mp * thetadot * np.sin(theta))
                    / (self.mp * np.sin(theta) ** 2 + self.mc),
                ],
                [
                    0.0,
                    (
                        -self.l * self.mp * thetadot**2 * np.cos(theta) ** 2
                        + self.l * self.mp * thetadot**2 * np.sin(theta) ** 2
                        - self.g * (self.mc + self.mp) * np.cos(theta)
                        + u * np.sin(theta)
                    )
                    / (self.l * (self.mp * np.sin(theta) ** 2 + self.mc))
                    + (
                        2
                        * self.mp
                        * np.cos(theta)
                        * np.sin(theta)
                        * (
                            self.l
                            * self.mp
                            * np.cos(theta)
                            * np.sin(theta)
                            * thetadot**2
                            + u * np.cos(theta)
                            + self.g * np.sin(theta) * (self.mc + self.mp)
                        )
                    )
                    / (self.l * (self.mp * np.sin(theta) ** 2 + self.mc) ** 2),
                    0.0,
                    -(2 * self.mp * thetadot * np.cos(theta) * np.sin(theta))
                    / (self.mp * np.sin(theta) ** 2 + self.mc),
                ],
            ]
        )

        fu = self.dt * np.array(
            [
                [0.0],
                [0.0],
                [1 / (self.mp * np.sin(theta) ** 2 + self.mc)],
                [-np.cos(theta) / (self.l * (self.mp * np.sin(theta) ** 2 + self.mc))],
            ]
        )
        return fx, fu


def _system_propagate_jax(dt, l, mc, g, mp, state, control, k):
    f = _system_dyn_jax(l, mc, g, mp, state, control, k)
    state_next = state + dt * f
    return state_next


def _system_dyn_jax(l, mc, g, mp, state, control, k):
    _, theta, xdot, thetadot = state
    u = control[0]

    f = jnp.array(
        [
            xdot,
            thetadot,
            (mp * jnp.sin(theta) * (l * thetadot**2 + g * jnp.cos(theta)) + u)
            / (mc + mp * (jnp.sin(theta)) ** 2),
            (
                -mp * l * thetadot**2 * jnp.cos(theta) * jnp.sin(theta)
                - (mc + mp) * g * jnp.sin(theta)
                - jnp.cos(theta) * u
            )
            / (l * (mc + mp * (jnp.sin(theta)) ** 2)),
        ]
    )
    return f
