# # %load_ext autoreload
# # %autoreload 2
import jax
from systems_dynamics.differential_drive import DifferentialDrive
from costs.quadratic_cost_penalty_method import QuadraticCostPenaltyMethod
from costs.stochastic_barrier_cost import StochasticBarrierCost
from ddp_algorithms.tdbas_pdp import TDBAS_PDP
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from bas_functions.gp_safety_function import GP_Safety

import jaxkern as jk
from functools import partial
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

# +
""" Define safety constraints (function h), generate BaS and embed into dynamics"""


def square_no_grad(x, y, cx, cy, s, r, theta):
    stheta, ctheta = (math.sin(theta), math.cos(theta))
    px, py = (
        (x - cx) * ctheta + (y - cy) * stheta,
        (x - cx) * stheta - (y - cy) * ctheta,
    )
    terms = (r * px + py, r * px - py)
    return abs(terms[0]) + abs(terms[1]) - s


def h_jax(params, n_agents, n1, x, k):
    h = jnp.zeros((params.shape[0], n_agents))
    for i in range(params.shape[0]):
        for n in range(n_agents):
            if params[i, 0] == 0:
                val = square_no_grad(x[n * n1], x[n * n1 + 1], *params[i, 1:6])
                h = h.at[i, n].set(val)

    return h.squeeze() - 1e-3


scenario = np.array(
    [[0, 3, 1.5, 1.8, 0.2, 0], [0, -3, -1.5, 1.8, 0.2, 0]]
)  # fig, ax = plt.subplots(1)
hs = [partial(h_jax, scenario[[i], :], 1, 3) for i in range(scenario.shape[0])]
h_ = partial(h_jax, scenario, 1, 3)
# hmap = jax.vmap(h_, in_axes=(0,None))

xlim, ylim = [-5, 5], [-5, 5]
fig, ax = visualize_safety_function(
    h_, xlim, ylim, start=model.x0, goal=model.xf, n1=3, resolution=100, contour=False
)

# for ox, oy, r in obs_array:
#     ax.add_patch(plt.Circle((ox, oy), r, color='r'))
# plt.xlim([-5,5])
# plt.ylim([-5,5])
# plt.show()
# -

h_true = h_
hmap_true = jax.vmap(h_true, in_axes=(0, None))
n_dim = 2
kernels = [jk.Matern52(), jk.Polynomial(degree=2)]
gp_hs = [
    GP_Safety(
        hs[i],
        2,
        x_lower=jnp.array([-5, -5]),
        x_upper=jnp.array([5, 5]),
        n_samples=200,
        lr=0.1,
        kernel=jk.ProductKernel(kernel_set=kernels),
        sampling="random",
    )
    for i in range(scenario.shape[0])
]  # jk.Polynomial(degree=2)+jk.RBF())
h_ests = [gp_h.h_est for gp_h in gp_hs]
fig, ax, mean_error1 = gp_hs[0].test_estimation(
    res=500, border=0, option="std", figax=(fig, ax)
)
fig, ax, mean_error2 = gp_hs[1].test_estimation(
    res=500, border=0, option="std", figax=(fig, ax)
)
print(mean_error1, mean_error2)
# plt.show()
# 1/0
# +
# Define BaS dynamics given system's dynamics and safety function
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, gp_h.h_mean, N, parameterized=True)
    for gp_h in gp_hs
]
# bas_parameterized = [bas_dynamics.BaSDynamics(model, h_, N, parameterized=True)]

parameter_list = [[None, None, None, None]] * 2
# parameter_list = [[2, None, 3, None]]
take_abs = [True, True, True, True] * 2
# -

embedded_dynamics_parameterized = embed_dynamics.EmbedDynamics(
    model, bas_parameterized, [1], parameter_list=parameter_list, jax=True
)
# embedded_dynamics_parameterized.embed(np.array([0,0,0,0]),0,np.array([1,2,3,4]))
n_bas = embedded_dynamics_parameterized.n_bas
# overwrite dynamics
# dynamics = embedded_dynamics_parameterized.system_propagate
""" Define Cost """
# State and Control Cost matrices
Q = [1e-3] * model.n  # state running cost
Q += [1e-1] * n_bas  # BaS running cost
# Q[dynamics.n-1, dynamics.n-1] = Q_dbas
# Q[dynamics.n-1, dynamics.n-1] = Q_dbas
Q = np.diag(Q)
R = np.diag([0.5 * 1e-3] * model.m)  # input running cost
S = [50] * model.n  # state terminal cost
S += [0.05] * n_bas  # BaS terminal cost
S = np.diag(S)
# Define Parameterized Cost function
cost_obj = QuadraticCostPenaltyMethod(
    Q, R, S, model.n, N, embedded_dynamics_parameterized.embed, parameterized=True
)
# cost_ddp = cost_obj.cost_ddp
# cost_pdp = cost_obj.cost_pdp
# cost_pdp(np.array([0,0,0,0]),np.array([0,0 ]),0,np.array([1,2,3,4]))
# cost_ddp(np.array([0,0,0,0,2]),np.array([0,0]),0,np.array([1,2,3,4]))
# Define Outer Cost
outer_tol_params = [100, 100, 1000, 1000]
outer_cost_obj = StochasticBarrierCost(
    h_ests, Q, R, S, N, model.n, barrier_params=outer_tol_params, risk_threshold=0.005
)
outer_cost = outer_cost_obj.cost
outer_cost_derivs = outer_cost_obj.getCostDerivs
# Define Outer Cost
# bas_outer = [bas_dynamics.BaSDynamics(model, h_, N, barrier_type = "tolerant_barrier", tol_params=[1,100,500,500], jax=True)]
# embedded_dynamics_outer = embed_dynamics.EmbedDynamics(model, bas_outer, [1], jax=True)
# outer_cost = QuadraticCostPenaltyMethod(Q, R, S, model.n, N, embedded_dynamics_outer.embed).cost
""" Initialize PDP Solver """
# DDP Iterations and convergence threshold
max_iters = 200
conv_threshold = 1e-3
options = ["pen_pdp"]  # ,'bas_ddp']
ls_option = ["iterative", "parallel"][1]
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
)


ubar = np.zeros((N, model.m))
# init_params = np.random.randn(PDP_solver.n_auxvar)*0.1 + 5
init_params = np.ones(PDP_solver.n_auxvar) * 5
x0 = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
xd = embedded_dynamics_parameterized.embed(model.xf, 0, init_params)
xd = jnp.tile(xd, (N + 1, 1))
# Compute Trajectory
PDP_solver.time_flag = True
start = time.time()
sol = PDP_solver.PDP_solve(x0, ubar, xd, init_params)
X = sol["Xs"][-1]
end = time.time()
print("elapsed time=", end - start)


# +
""" Plot and print data """
# fig, ax = plt.subplots(1)
# fig, ax = visualize_safety_function(h_, xlim, ylim, start = model.x0, goal = model.xf, n1=3, resolution = 500,  contour = False)
ax.plot(X[:, 0], X[:, 1])


# for ox, oy, r in obs_array:
#     ax.add_patch(plt.Circle((ox, oy), r, color='r'))
# with jax.disable_jit(True):
print("Collided true: ", jnp.any(hmap_true(X[:, :2], 0) <= 0))
print(
    "Collided est: ",
    jnp.any(jnp.array([gp_h.h_mean(X[:, :2], 0) for gp_h in gp_hs]) <= 0),
)
print("risk: ", outer_cost_obj.risk_of_traj(X))
ax.set_aspect("equal")
# -

fig, ax = gp_hs[0].plot_traj_uncertainty(X, np.arange(N + 1) * dt)
fig, ax = gp_hs[1].plot_traj_uncertaitny(X, np.arange(N + 1) * dt, figax=(fig, ax))
plt.show()

# auxvar = sol['Parameters'][11]
# U = sol['Us'][-1]
# Lambdas = PDP_solver.getCostate(X, U, xd, auxvar) # Get Costate from Trajectory
# auxsys_OCs = PDP_solver.getAuxSys(X, U, Lambdas, xd, auxvar) # Get PDP Matrices
