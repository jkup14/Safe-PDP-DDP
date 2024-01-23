import numpy as np
import jax.numpy as jnp


class SingleIntegrator:
    def __init__(self, dt):
        self.n = 2
        self.m = 2
        self.dt = dt
        self.x0 = np.array([[3], [0], [0], [0]])
        self.xf = np.array([[-3], [-0.85], [0], [0]])

    def system_dyn(self, state, control):
        u1, u2 = control.item(0), control.item(1)

        dxdt = np.array([[u1], [u2]])
        return dxdt

    def system_dyn_jax(self, state, control):
        u1, u2 = control

        dxdt = jnp.array([u1, u2])
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
        fx = np.eye(self.n) + self.dt * np.array([[0.0, 0.0], [0.0, 0.0]])

        fu = self.dt * np.array([[1.0, 0.0], [0.0, 1.0]])
        return fx, fu
