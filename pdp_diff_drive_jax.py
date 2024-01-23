from systems_dynamics.differential_drive import DifferentialDrive
from costs.quadratic_cost_penalty_method import QuadraticCostPenaltyMethod
from costs.quadratic_cost import QuadraticCost
from ddp_algorithms.tdbas_pdp import TDBAS_PDP
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from functools import partial
import jax
import math
from plottings.plot_2dvehicle import visualize_safety_function

"""Specify horizon, sampling time, and generate time vector"""
N = 250
dt = 0.02
times = jnp.linspace(0, dt * N - dt, N)

"""choose system's dynamics"""
model = DifferentialDrive(dt)

"""Start and Goal states"""
model.x0 = jnp.array([0, -4, np.pi / 2])
model.xf = jnp.array([0, 4, np.pi / 2])

""" Define safety constraints (function h), generate BaS and embed into dynamics"""


# Rectangle safety function give agent state x,y, rect center cx,cy, params s,r, and rotation theta
def rectangle(x, y, cx, cy, s, r, theta):
    stheta, ctheta = (math.sin(theta), math.cos(theta))
    px, py = (
        (x - cx) * ctheta + (y - cy) * stheta,
        (x - cx) * stheta - (y - cy) * ctheta,
    )
    terms = (r * px + py, r * px - py)
    return abs(terms[0]) + abs(terms[1]) - s


def h_generic(params, x, k):
    h = jnp.zeros(params.shape[0])
    for i in range(params.shape[0]):
        val = rectangle(x[0], x[1], *params[i])
        h = h.at[i].set(val)
    return h.squeeze()


# Define Obstacles, two rectangles
scenario = np.array([[3, 1.5, 1.8, 0.2, 0], [-3, -1.5, 1.8, 0.2, 0]])
h_ = partial(h_generic, scenario)
# Two separate safety functions for two barrier states
h_list = [
    partial(h_generic, scenario[[i]]) for i in range(scenario.shape[0])
]  # h for specific scenario
hmap = jax.vmap(h_, in_axes=(0, None))  # vmap'd

# Scenario Limits
xlim, ylim = [-5, 5], [-5, 5]

# Optional visualization of scenario prior to experiment
visualize = False
if visualize:
    fig, ax = visualize_safety_function(
        h_,
        xlim,
        ylim,
        start=model.x0,
        goal=model.xf,
        n1=3,
        resolution=100,
        contour=False,
    )


# Define which parameters to optimize, ordered by [p,m,c1,c2]. Replace param with None to flag to PDP to optimize it. Second dimension is for multiple barrier states.
# parameter_list = [[1.6264218, None, None, 6.072945],[None, None, None, 6.37802]] # Optimizing 5/8
parameter_list = [[None, None, None, None]] * 2  # Optimizing all 8
# Define BaS dynamics given system's dynamics and safety function
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, h_list[i], N, tol_params=parameter_list[i])
    for i in range(len(h_list))
]

# Mask to take abs of parameter list
# take_abs = [True, True, True, True, True] # Optimizing 5/8
take_abs = [True] * 8  # Optimizing all 8

# Embed parmaeterized barrier state into dynamics to make parameterized dynamics
embedded_dynamics_parameterized = embed_dynamics.EmbedDynamics(
    model, bas_parameterized, N=N
)
n_bas = embedded_dynamics_parameterized.n_bas

""" Define Inner Cost """
# State Running Cost matrix
Q = [1e-3] * model.n  # state running cost
Q += [1e-1] * n_bas  # BaS running cost
Q = np.diag(Q)
# Control Cost matrix
R = np.diag([0.5 * 1e-3] * model.m)  # input running cost
# Terminal Cost matrix
S = [50] * model.n  # state terminal cost
S += [0.05] * n_bas  # BaS terminal cost
S = np.diag(S)

# Cost for pdp, to be used in hamiltonian calculation
cost_pdp = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_parameterized.embed
).cost_pdp
# Cost for bas-embedded ddp
cost_ddp = QuadraticCost(Q, R, S, N).parameterized_cost_jax

"""Define Outer Cost"""
# Barrier functions for outer cost penalty, tolerant barrier function is designed so that there is no penalty in the safe set
bas_outer = [
    bas_dynamics.BaSDynamics(model, h_list[i], N, tol_params=[10, 100, 500, 500])
    for i in range(len(h_list))
]
embedded_dynamics_outer = embed_dynamics.EmbedDynamics(model, bas_outer, N)
outer_cost = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_outer.embed
).cost

""" Initialize PDP Solver """
# DDP and PDP iterations, convergence threshold, learning rate schedule for linesearch
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
    conv_threshold_pdp=conv_threshold,
    time_flag=False,
    alphas=alphas,
)

# Initial guess of control trajectory and parameters
ubar = np.zeros((N, model.m))
# init_params = np.random.randn(PDP_solver.n_auxvar) * 0.1 + 5
init_params = np.ones(PDP_solver.n_auxvar) * 5
# Embed initial and final conditions
x0 = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
xd = embedded_dynamics_parameterized.embed(model.xf, 0, init_params)
# Compute Trajectory
start = time.time()
sol = PDP_solver.PDP_solve(x0, ubar, xd, init_params)
X_sol = sol["Xs"][-1]
end = time.time()
print("elapsed time=", end - start)

""" Plot and print Trajectory and Safety Function """
# Get obstacle map
fig, ax = visualize_safety_function(
    h_, xlim, ylim, start=model.x0, goal=model.xf, n1=3, resolution=500, contour=False
)
# Plot trajectory and safety function
ax.plot(X_sol[:, 0], X_sol[:, 1])
h_traj = hmap(X_sol[:, :2], 0)
print("Collided: ", jnp.any(h_traj <= 0))
ax.set_aspect("equal")

fig, ax = plt.subplots(1, 1)
ax.plot(np.linspace(0, N * dt, N + 1), h_traj, "black")
ax.axhline(y=0, color="r", linestyle="--")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Safety function $h_i$")
plt.show()
