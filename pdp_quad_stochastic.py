# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.9.7 ('ACDS')
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
from systems_dynamics.quadrotor import Quadrotor
from costs.quadratic_cost_penalty_method import QuadraticCostPenaltyMethod
from costs.stochastic_barrier_cost import StochasticBarrierCost
from ddp_algorithms.tdbas_pdp import TDBAS_PDP
from systems_constraints import obstacles_2d
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxkern as jk
import time
from functools import partial
import jax
from bas_functions.gp_safety_function import GP_Safety
import math
from plottings.plot_3dvehicle import animate3dVehicle_Multi_track
from systems_constraints import obstacles_3d

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

obs_array = jnp.array(
    [
        [-0.2, -0.2, 0.0, 1.0],
        [4.9, 4.95, 0, 1],
        [4.9, -4.95, 0, 1],
        [-4.9, 4.95, 0, 1],
        [-4.9, -4.95, 0, 1],
    ]
)


def h_jax(params, x, k):
    h = jnp.zeros(params.shape[0])
    px, py, pz = x[9:12]
    for i, (ox, oy, oz, r) in enumerate(params):
        h = h.at[i].set((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2 - r**2)
    return h


hs = [partial(h_jax, obs_array[[0], :]), partial(h_jax, obs_array[1:])]
h_ = partial(h_jax, obs_array)

xlim, ylim = [-5, -5, -3], [5, 5, 3]

h_true = h_
hmap_true = jax.vmap(h_true, in_axes=(0, None))
gp_hs = [
    GP_Safety(
        h,
        model.n,
        x_lower=[-6, -6, -2],
        x_upper=[6, 6, 2],
        n_samples=200,
        lr=0.001,
        active_dims=[9, 10, 11],
        sampling="random",
    )
    for h in hs
]  # jk.Polynomial(degree=2)+jk.RBF())
h_ests = [gp_h.h_est for gp_h in gp_hs]
_, _, mean_error1 = gp_hs[0].test_estimation(
    res=100, border=1, option="std", twod=False
)
_, _, mean_error2 = gp_hs[1].test_estimation(
    res=100, border=1, option="std", twod=False
)

print(mean_error1, mean_error2)

# generate tracking trajectory:
t_pi = 2 * jnp.pi * 0.96 * times / times[-1]
x_des = 6 * jnp.sin(t_pi)
y_des = 5 * jnp.sin(2 * t_pi)
z_des = 0 * t_pi
traj_desired = jnp.zeros((N, model.n))
traj_desired = traj_desired.at[:, 9].set(x_des)
traj_desired = traj_desired.at[:, 10].set(y_des)
xd = traj_desired
xd = jnp.vstack([model.x0, xd])

# 1/0
# fig, ax = visualize_safety_function(h_, xlim, ylim, start = model.x0, goal = model.xf, n1=3, resolution = 100, contour = False)

# for ox, oy, r in obs_array:
#     ax.add_patch(plt.Circle((ox, oy), r, color='r'))
# plt.xlim([-5,5])
# plt.ylim([-5,5])
# plt.show()

# +
# h_true = h_
# hmap_true = jax.vmap(h_true, in_axes=(0,None))
# n_dim = 2
# gp_h = GP_Safety(h_true, 2,  x_lower = jnp.array([-10,-10]), x_upper = jnp.array([10,10]), n_samples=10, lr=0.001)

# +
# Define BaS dynamics given system's dynamics and safety function
# bas_parameterized = [bas_dynamics.BaSDynamics(model, gp_h.h_mean, N, parameterized=True)]
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, gp_h.h_mean, N, parameterized=True)
    for gp_h in gp_hs
]

parameter_list = [[None, None, None, None]] * 2
# parameter_list = [[2, None, 3, None]]
take_abs = [True, True, True, True] * 2
# -

embedded_dynamics_parameterized = embed_dynamics.EmbedDynamics(
    model, bas_parameterized, [1], parameter_list=parameter_list, jax=True
)
# embedded_dynamics_parameterized.embed(model.x0,0,np.array([1,2,3,4]))
n_bas = embedded_dynamics_parameterized.n_bas
# overwrite dynamics
# dynamics = embedded_dynamics_parameterized.system_propagate
""" Define Cost """
Q_angle, Q_dangle, Q_vel, Q_pos, Q_dbas = 1e-6, 3e-4, 6e-3, 1e-2, 1e-3
Q = [Q_angle] * 3 + [Q_dangle] * 3 + [Q_vel] * 3 + [Q_pos] * 3 + [Q_dbas] * n_bas
Q = jnp.diag(jnp.array(Q))

R = 5e-4 * np.eye(embedded_dynamics_parameterized.m)

# state terminal cost matrix
S = Q
# Define Parameterized Cost function
cost_obj = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_parameterized.embed, parameterized=True
)
# cost_ddp = cost_obj.cost_ddp
# cost_pdp = cost_obj.cost_pdp
# cost_pdp(np.array([0,0,0,0]),np.array([0,0 ]),0,np.array([1,2,3,4]))
# cost_ddp(np.array([0,0,0,0,2]),np.array([0,0]),0,np.array([1,2,3,4]))
# Define Outer Cost
# outer_cost = StochasticBarrierCost(gp_h.h_est, gp_h.out_dim, Q, R, S, N, model.n, barrier_params=[1,1,1000,1000], risk_threshold=0.01)
# Define Outer Cost
Q_outer = jnp.diag(jnp.array([0] * 9 + [1e-2] * 3))
S_outer = Q_outer
outer_tol_params = [500, 500, 500, 500]
outer_cost_obj = StochasticBarrierCost(
    h_ests,
    Q_outer,
    R,
    S_outer,
    N,
    model.n,
    barrier_params=outer_tol_params,
    risk_threshold=0.0001,
)
outer_cost = outer_cost_obj.cost
outer_cost_derivs = outer_cost_obj.getCostDerivs
""" Initialize PDP Solver """
# DDP Iterations and convergence threshold
max_iters = 200
conv_threshold = 1e-3
options = ["pen_pdp"]  # ,'bas_ddp']
ls_option = ["iterative", "parallel"][0]
PDP_solver = TDBAS_PDP(
    embedded_dynamics_parameterized,
    cost_obj,
    outer_cost,
    embedded_dynamics_parameterized.n,
    model.m,
    embedded_dynamics_parameterized.n_parameters,
    n_bas,
    N,
    take_abs=take_abs,
    vectorize_outercost=False,
    OuterCost_derivatives=outer_cost_derivs,
    pdp_options=options,
    ls_option=ls_option,
    time_flag=True,
)


ubar = np.zeros((N, model.m))
init_params = np.random.randn(PDP_solver.n_auxvar) * 0.1 + 5
init_params = np.ones(PDP_solver.n_auxvar) * 5
x0 = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
xd = jax.vmap(embedded_dynamics_parameterized.embed, in_axes=(0, None, None))(
    xd, 0, init_params
)
# Compute Trajectory
start = time.time()
sol = PDP_solver.PDP_solve(x0, ubar, xd, init_params)
X_pen = sol["Xs"][-1]
U_pen = sol["Us"][-1]
end = time.time()
print("elapsed time=", end - start)


# +
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
ax.plot3D(X_pen[:, 9], X_pen[:, 10], X_pen[:, 11], "black", linewidth=2)
ax.scatter(model.x0[9], model.x0[10], model.x0[11], c="g", s=50)
ax.scatter(xd[-1, 9], xd[-1, 10], xd[-1, 11], c="r", s=40, marker="x")

ax.set_aspect("auto")
ax.set_zlim([-6, 6])
fig, ax = gp_hs[0].plot_traj_uncertainty(X_pen, np.arange(N + 1) * dt)
fig, ax = gp_hs[1].plot_traj_uncertainty(X_pen, np.arange(N + 1) * dt, figax=(fig, ax))
print("risk pen: ", outer_cost_obj.risk_of_traj(X_pen))
print("Collided at: ", np.argwhere(hmap_true(X_pen, 0) < 0))
plt.show()
# fig1 = animate3dVehicle_Multi_track(1, 12, 1, model.x0, model.xf, traj_desired, times, X, U, obstacle_info)

# -

# gp_h.plot_traj_uncertainty(X, np.arange(N+1)*dt)
# plt.show()

# auxvar = sol['Parameters'][11]
# U = sol['Us'][-1]
# Lambdas = PDP_solver.getCostate(X, U, xd, auxvar) # Get Costate from Trajectory
# auxsys_OCs = PDP_solver.getAuxSys(X, U, Lambdas, xd, auxvar) # Get PDP Matrices
