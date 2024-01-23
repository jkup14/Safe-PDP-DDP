import numpy as np
import jax.numpy as jnp


class DoubleIntegrator:
    def __init__(self, dt):
        self.n = 4
        self.m = 2
        self.dt = dt

    def system_dyn(self, state, control):
        x = state.item(0)
        y = state.item(1)
        vx = state.item(2)
        vy = state.item(3)
        u1, u2 = control.item(0), control.item(1)

        dxdt = np.array([[vx], [vy], [u1], [u2]])
        return dxdt

    def system_dyn_jax(self, state, control):
        x, y, vx, vy = state
        u1, u2 = control

        dxdt = jnp.array([vx, vy, u1, u2])
        return dxdt

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f  # euler integration
        return state_next

    def system_propagate_jax(self, state, control, k):
        f = self.system_dyn_jax(state[: self.n], control)
        state_next = state[: self.n] + self.dt * f  # euler integration
        return state_next

    def system_grad(self, state, control, k):
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        u1, u2 = control

        fx = np.eye(self.n) + self.dt * np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        fu = self.dt * np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        return fx, fu
