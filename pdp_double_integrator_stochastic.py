# from safety_embedded_ddp_python import *
from systems_dynamics.double_integrator import DoubleIntegrator
from costs.quadratic_cost_penalty_method import QuadraticCostPenaltyMethod
from costs.stochastic_barrier_cost import StochasticBarrierCost
from ddp_algorithms.tdbas_pdp import TDBAS_PDP
# from systems_constraints import obstacles_2d
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
from plottings import plot_2dvehicle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from functools import partial
import jax
from bas_functions.gp_safety_function import GP_Safety


"""Specify horizon, sampling time, and generate time vector"""
N = 250
dt = 0.02
times = jnp.linspace(0, dt * N - dt, N)

"""choose system's dynamics"""
model = DoubleIntegrator(dt)

"""Start and Goal states"""
model.x0 = jnp.array([4, 0, 0, 0])
model.xf = jnp.array([-4, 0.1, 0, 0])

""" Define safety constraints (function h), generate BaS and embed into dynamics"""
obs_array = np.array([[1.5, 1, 1], [-1.5, -1.0, 1]])
# fig, ax = plt.subplots(1)
# for ox, oy, r in obs_array:
#     ax.add_patch(plt.Circle((ox, oy), r, color='r'))
# plt.xlim([-5,5])
# plt.ylim([-5,5])
# plt.show()


def h_(x, k, obs_array):
    px, py = x[0:2]
    return (
        (px - obs_array[:, 0]) ** 2 + (py - obs_array[:, 1]) ** 2 - obs_array[:, 2] ** 2
    ).reshape(-1)


h_true = partial(h_, **{"obs_array": obs_array})
hmap_true = jax.vmap(h_true, in_axes=(0, None))
n_dim = 2
gp_h = GP_Safety(
    h_true,
    2,
    x_lower=jnp.array([-10, -10]),
    x_upper=jnp.array([10, 10]),
    n_samples=10,
    lr=0.001,
)

# Define BaS dynamics given system's dynamics and safety function
bas_parameterized = [
    bas_dynamics.BaSDynamics(model, gp_h.h_mean, N, parameterized=True)
]
parameter_list = [[None, None, None, None]]
# parameter_list = [[2, None, 3, None]]
take_abs = [True, True, True, True]

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
Q += [1e-1]  # BaS running cost
# Q[dynamics.n-1, dynamics.n-1] = Q_dbas
# Q[dynamics.n-1, dynamics.n-1] = Q_dbas
Q = np.diag(Q)
R = np.diag([0.5 * 1e-3] * model.m)  # input running cost
S = [50] * model.n  # state terminal cost
S += [0.05]  # BaS terminal cost
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
outer_cost = StochasticBarrierCost(
    gp_h.h_est,
    gp_h.out_dim,
    Q,
    R,
    S,
    N,
    model.n,
    barrier_params=[1, 10, 500, 500],
    risk_threshold=0.01,
)
""" Initialize PDP Solver """
# DDP Iterations and convergence threshold
max_iters = 200
conv_threshold = 1e-3
options = ["bas_pdp"]
PDP_solver = TDBAS_PDP(
    embedded_dynamics_parameterized,
    cost_obj,
    outer_cost.cost,
    embedded_dynamics_parameterized.n,
    model.m,
    embedded_dynamics_parameterized.n_parameters,
    n_bas,
    N,
    take_abs=take_abs,
    options=options,
    vectorize_outercost=False,
    OuterCost_derivatives=outer_cost.getCostDerivs,
)
ubar = np.zeros((N, model.m))
# init_params = np.random.randn(PDP_solver.n_auxvar)*0.1 + 5
init_params = np.ones(PDP_solver.n_auxvar) * 5
x0 = embedded_dynamics_parameterized.embed(model.x0, 0, init_params)
xd = embedded_dynamics_parameterized.embed(model.xf, 0, init_params)
xd = jnp.tile(xd, (N + 1, 1))
# Compute Trajectory
start = time.time()
sol = PDP_solver.PDP_solve(x0, ubar, xd, init_params)
X = sol["Xs"][-1]
end = time.time()
print("elapsed time=", end - start)
""" Plot and print data """
fig, ax = plt.subplots(1)
plt.plot(X[:, 0], X[:, 1])
for ox, oy, r in obs_array:
    ax.add_patch(plt.Circle((ox, oy), r, color="r"))
print("Collided true: ", jnp.any(hmap_true(X[:, :2], 0) <= 0))
print("Collided est: ", jnp.any(gp_h.h_mean(X[:, :2], 0) <= 0))
print("risk: ", outer_cost.risk_of_traj(X))
plt.gca().set_aspect("equal")
plt.show()
