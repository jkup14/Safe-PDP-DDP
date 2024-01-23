""" This is a differential drive model but with inputs as trans velocity and ang velocity """
"called it unicycle as the robotarium which we use it for"
import numpy as np


class UnicycleRobot:
    def __init__(self, dt):
        self.n = 3
        self.m = 2
        self.dt = dt
        # self.rad = 0.2      # radius of wheels
        # self.width = 0.2     # distance between wheels
        self.x0 = np.array([[3], [0], [0]])
        self.xf = np.array([[-2], [-0.5], [0]])

    def system_dyn(self, state, control):
        x, y, theta = state.item(0), state.item(1), state.item(2)
        v, w = control.item(0), control.item(1)

        f = np.array([[v * np.cos(theta)], [v * np.sin(theta)], [w]])
        return f

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f  # euler integration
        return state_next

    def system_grad(self, state, control, k):
        x, y, theta = state.item(0), state.item(1), state.item(2)
        v, w = control.item(0), control.item(1)

        fx = np.eye(self.n) + self.dt * np.array(
            [
                [0.0, 0.0, -v * np.sin(theta)],
                [0.0, 0.0, v * np.cos(theta)],
                [0.0, 0.0, 0.0],
            ]
        )

        fu = self.dt * np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        return fx, fu
