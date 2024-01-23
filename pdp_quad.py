from systems_dynamics.quadrotor import Quadrotor
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
from plottings.plot_3dvehicle import animate3dVehicle_Multi_track

from mpl_toolkits import mplot3d

"""Specify horizon, sampling time, and generate time vector"""
dt = 0.03
N = 666
times = jnp.linspace(0, N * dt - dt, N)

"""choose system's dynamics"""
model = Quadrotor(dt)

"""Start and Goal states"""
model.x0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0.1])
model.xf = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

"""Generate tracking trajectory"""
t_pi = 2 * jnp.pi * 0.96 * times / times[-1]
x_des = 6 * jnp.sin(t_pi)
y_des = 5 * jnp.sin(2 * t_pi)
z_des = 0 * t_pi
traj_desired = jnp.zeros((N, model.n))
traj_desired = traj_desired.at[:, 9].set(x_des)
traj_desired = traj_desired.at[:, 10].set(y_des)
xd = traj_desired
xd = jnp.vstack([model.x0, xd])

""" Define safety constraints (function h), generate BaS and embed into dynamics"""


# Takes spherical obstacle params and state, gives safety function h
def h_generic(params, x, k):
    h = jnp.zeros(params.shape[0])
    px, py, pz = x[9:12]
    for i, (ox, oy, oz, r) in enumerate(params):
        h = h.at[i].set((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2 - r**2)
    return h


obs_array = jnp.array(
    [
        [-0.2, -0.2, 0.0, 1.0],
        [4.9, 4.95, 0, 1],
        [4.9, -4.95, 0, 1],
        [-4.9, 4.95, 0, 1],
        [-4.9, -4.95, 0, 1],
    ]
)

# Define safety functions, 5 spheres
h1_ = partial(
    h_generic, obs_array[[0], :]
)  # First barrier state just for sphere near origin
h2_ = partial(h_generic, obs_array[1:])  # Second barrier state for all others
h_list = [h1_, h2_]
hmap = jax.vmap(partial(h_generic, obs_array), in_axes=(0, None))

# Scenario Limits
xlim, ylim = [-5, -5, -3], [5, 5, 3]


# Define which parameters to optimize, ordered by [p,m,c1,c2]. Replace param with None to flag to PDP to optimize it. Second dimension is for multiple barrier states.
parameter_list = [[None, None, None, None]] * 2  # All 8

# Define BaS dynamics given system's dynamics and safety function
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, h_list[i], N, tol_params=parameter_list[i])
    for i in range(len(h_list))
]

# Mask to take abs of parameter list
take_abs = [True] * 8  # All 8

# Embed parmaeterized barrier state into dynamics to make parameterized dynamics
embedded_dynamics_parameterized = embed_dynamics.EmbedDynamics(
    model, bas_parameterized, N=N
)
n_bas = embedded_dynamics_parameterized.n_bas

""" Define Inner Cost """
# State Running Cost matrix
Q_angle, Q_dangle, Q_vel, Q_pos, Q_dbas = 1e-5, 5e-4, 3e-3, 5e-2, 1e-3
Q = [Q_angle] * 3 + [Q_dangle] * 3 + [Q_vel] * 3 + [Q_pos] * 3 + [Q_dbas] * n_bas
Q = jnp.diag(jnp.array(Q))
# Control Cost matrix
R = 5e-4 * np.eye(embedded_dynamics_parameterized.m)
# State terminal cost matrix
S = Q * 10

# Cost for pdp, to be used in hamiltonian calculation
cost_pdp = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_parameterized.embed
).cost_pdp
cost_ddp = QuadraticCost(Q, R, S, N).parameterized_cost_jax

"""Define Outer Cost"""
outer_tol_params = [100, 100, 500, 500]
bas_outer = [
    bas_dynamics.BaSDynamics(model, h_list[i], N, tol_params=outer_tol_params)
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
# Embed initial condition and tracking trajectory
x0_embedded = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
times = jnp.hstack([times, N * dt])
xd_embedded = embedded_dynamics_parameterized.embed_traj(xd, times, init_params)

#  Compute Trajectory
start = time.time()
sol = PDP_solver.PDP_solve(x0_embedded, ubar, xd_embedded, init_params)
X_sol = sol["Xs"][-1]
end = time.time()
print("elapsed time=", end - start)

h_traj = hmap(X_sol, 0)
print("Collided after escaping: ", jnp.any(h_traj[50:] <= 0))

""" Plot and print data """
fig = plt.figure()
ax = plt.axes(projection="3d")
for ox, oy, oz, r in obs_array:
    rx, ry, rz = r, r, r
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x + ox, y + oy, z + oz, rstride=4, cstride=4, color="r", alpha=0.4)
ax.plot3D(x_des, y_des, z_des, "b--", alpha=0.5, linewidth=3)
ax.plot3D(X_sol[:, 9], X_sol[:, 10], X_sol[:, 11], "black", linewidth=2)
ax.scatter(model.x0[9], model.x0[10], model.x0[11], c="g", s=50)
ax.scatter(xd[-1, 9], xd[-1, 10], xd[-1, 11], c="r", s=40, marker="x")

ax.set_aspect("auto")
ax.set_zlim([-6, 6])
plt.show()
# fig1 = animate3dVehicle_Multi_track(1, 12, 1, model.x0, model.xf, traj_desired, times, X, U, obstacle_info)
