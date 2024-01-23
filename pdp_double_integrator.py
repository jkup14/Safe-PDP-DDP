from systems_dynamics.double_integrator import DoubleIntegrator
from costs.quadratic_cost_penalty_method import QuadraticCostPenaltyMethod
from costs.quadratic_cost import QuadraticCost
from ddp_algorithms.tdbas_pdp import TDBAS_PDP
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial

"""Specify horizon, sampling time, and generate time vector"""
N = 250
dt = 0.02
times = np.linspace(0, dt * N - dt, N)

"""choose system's dynamics"""
model = DoubleIntegrator(dt)

"""Start and Goal states"""
model.x0 = np.array([-4, 0, 0, 0])
model.xf = np.array([4, 0.1, 0, 0])

""" Define safety constraints (function h), generate BaS and embed into dynamics"""


# Takes circular obstacle params and state, gives safety function h
def h_(x, k, obs_array):
    px, py = x[0:2]
    return (
        (px - obs_array[:, 0]) ** 2 + (py - obs_array[:, 1]) ** 2 - obs_array[:, 2] ** 2
    ).reshape((-1))


# Define Obstacles, two circles
obs_array = np.array([[1.5, 0.5, 1], [-1.5, -0.5, 1]])

h = partial(h_, **{"obs_array": obs_array})

# Define which parameters to optimize, ordered by [p,m,c1,c2]. Replace param with None to flag to PDP to optimize it. Second dimension is for multiple barrier states.
parameter_list = [[None, None, None, None]]

# Define BaS dynamics given system's dynamics and safety function
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, h, N, tol_params=parameter_list[0])
]

# Mask to take abs of parameter list
take_abs = [True, True, True, True]

# Embed parmaeterized barrier state into dynamics to make parameterized dynamics
embedded_dynamics_parameterized = embed_dynamics.EmbedDynamics(
    model, bas_parameterized, N=N
)
n_bas = embedded_dynamics_parameterized.n_bas

""" Define Inner Cost """
# State Cost matrices
Q = [0] * model.n  # state running cost
Q += [1e-1]  # BaS running cost
Q = np.diag(Q)

# Control Cost matrix
R = np.diag([0.5 * 1e-3] * model.m)  # input running cost
# Terminal Cost matrix
S = [50] * model.n  # state terminal cost
S += [0.05]  # BaS terminal cost
S = np.diag(S)

# Cost for pdp, to be used in hamiltonian calculation
cost_pdp = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_parameterized.embed
).cost_pdp
# Cost for bas-embedded ddp
cost_ddp = QuadraticCost(Q, R, S, N).parameterized_cost_jax


"""Define Outer Cost"""
# Barrier functions for outer cost penalty, tolerant barrier function is designed so that there is no penalty in the safe set
bas_outer = [bas_dynamics.BaSDynamics(model, h, N, tol_params=[1, 100, 500, 500])]
embedded_dynamics_outer = embed_dynamics.EmbedDynamics(model, bas_outer, N)
outer_cost = QuadraticCostPenaltyMethod(
    Q, R * 100, S, model.n, N, embedded_dynamics_outer.embed
).cost

""" Initialize PDP Solver """
# DDP Iterations and convergence threshold, learning rate schedule for linesearch
max_iters_ilqr = 200
max_iters_pdp = 50
conv_threshold = 1e-3
alphas = np.power(5, np.linspace(1, -5, 10))
# Choose linesearch to be iterative (traditional) or in parallel (slower but more optimal)
ls_option = ["iterative", "parallel"][0]
PDP_solver = TDBAS_PDP(
    embedded_dynamics_parameterized,
    cost_pdp,
    cost_ddp,
    outer_cost,
    take_abs=take_abs,
    ls_option=ls_option,
    max_iters_ilqr=max_iters_ilqr,
    max_iters_pdp=max_iters_pdp,
    time_flag=False,
    alphas=alphas,
    conv_threshold_pdp=conv_threshold,
)

# Initial guess of control trajectory and parameters
ubar = np.zeros((N, model.m))
init_params = np.ones(PDP_solver.n_auxvar) * 5
x0 = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
xd = embedded_dynamics_parameterized.embed(model.xf, 0, init_params)
# Compute Trajectory
start = time.time()
sol = PDP_solver.PDP_solve(x0, ubar, xd, init_params)
X = sol["Xs"][-1]
end = time.time()
print("elapsed time=", end - start)

""" Plot and print data """
fig, ax = plt.subplots(1)
ax.set_aspect("equal")
plt.scatter(x0[0], x0[1], color="g")
plt.scatter(xd[0], xd[1], color="r", marker="x")
plt.plot(X[:, 0], X[:, 1])
for ox, oy, r in obs_array:
    ax.add_patch(plt.Circle((ox, oy), r, color="r"))
print("collided: ", np.any(h(X[:, :2], 0) <= 0))
plt.show()
