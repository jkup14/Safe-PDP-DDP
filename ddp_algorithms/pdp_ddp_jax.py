import jax
from jax import custom_derivatives
from jax import device_get
from jax import hessian
from jax import jacobian
from jax import jit
from jax import lax
from jax import random
from jax import vmap
import jax.numpy as np
from functools import partial
from ddp_algorithms.jax_helpers import *
from ddp_algorithms.jax_forward_backward_passes import *


class PDPDDPJax:
    """
    Dynamic Differential Programming for given dynamics and cost function
    """

    def __init__(
        self,
        cost,
        dynamics,
        n_control,
        nbas=0,
        reject_unsafe_traj_flags=(True),
        maxiter=100,
        conv_threshold=1e-4,
        mu0=0.0,
        delta0=2.0,
        mumin=1e-6,
        mumax=1e10,
        verbose=True,
        ls=True,
        reg=True,
        alpha0=0,
        alphaf=-3,
        alphalen=11,
        u_min=None,
        u_max=None,
    ):
        """Iterative Linear Quadratic Regulator.
        Args:
        cost:      cost(x, u, t) returns scalar.
        dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
        x0: initial_state - 1D np array of shape (n, ).
        U: initial_controls - 2D np array of shape (T, m).
        maxiter: maximum iterations.
        """
        self.cost = cost
        self.dynamics = dynamics
        self.maxiter = maxiter
        self.conv_threshold = conv_threshold

        self.nbas = nbas
        self.reject_unsafe_traj_flags = reject_unsafe_traj_flags
        self.lsfunc = (
            parameterized_line_search_ddp_no_bas
            if nbas == 0
            else parameterized_line_search_ddp_bas
        )

        self.verbose = verbose

        assert not (
            ls is False and reg is False
        ), "Linsearch and Regularization can't both be off"

        self.ls = ls
        self.reg = reg

        self.alpha0, self.alphaf, self.alphalen = (
            (alpha0, alphaf, alphalen) if ls else (0, 0, 1)
        )
        self.mu0, self.delta0 = (float(mu0), float(delta0)) if reg else (0.0, 0.0)
        self.mumin = mumin
        self.mumax = mumax

        if u_min is None:
            u_min = tuple([-np.inf] * n_control)
        else:
            assert type(u_min) is tuple, "u_min must be tuple"
            assert (
                len(u_min) == n_control
            ), "Control limits are not the same shape as the control"
        if u_max is None:
            u_max = tuple([np.inf] * n_control)
        else:
            assert type(u_max) is tuple, "u_max must be tuple"
            assert (
                len(u_max) == n_control
            ), "Control limits are not the same shape as the control"
        self.u_min = u_min
        self.u_max = u_max

        if verbose:
            print(
                "JaxiLQR object created with bas set to",
                nbas != 0,
                ", linesearch",
                ("on," if ls else "off,"),
                "and regularization",
                ("on" if reg else "off"),
            )

    def compute_optimal_solution(self, x0, ubar, xd, auxvar):
        ubar = np.array(ubar) if ubar.shape[0] > ubar.shape[1] else np.array(ubar.T)
        x0 = np.squeeze(np.array(x0))
        if len(xd.shape) == 1:
            xd = np.tile(xd, (ubar.shape[0] + 1, 1))
        else:
            assert xd.shape == (ubar.shape[0] + 1, x0.shape[0]), (
                "shape of xd should either be (n_state,) for target reaching or (N+1,n_state) for tracking, it is "
                + str(xd.shape)
            )
        Xs, Us, _, _, Objs, lqr, it, ls_success, mu, delta, K, k = pdp_ilqr(
            self.cost,
            self.dynamics,
            x0,
            ubar,
            xd,
            auxvar,
            self.lsfunc,
            self.nbas,
            self.reject_unsafe_traj_flags,
            self.maxiter,
            self.conv_threshold,
            self.mu0,
            self.delta0,
            self.mumin,
            self.mumax,
            self.alpha0,
            self.alphaf,
            self.alphalen,
            self.u_min,
            self.u_max,
        )

        it = it - 1 if it == self.maxiter else it

        if self.verbose:
            print(
                "After ",
                it,
                " iterations, Linesearch ",
                ("succeeded" if ls_success else "failed"),
                " with mu=",
                str(mu),
            )
            if not ls_success:
                print("iLQR Linesearch Failed!")
        # return Xs[:,:,:it+1], Us[:,:,:it+1], Objs[:it+1], lqr, it, ls_success, mu, delta
        solution = {
            "Xs": Xs,
            "Us": Us,
            "Objs": Objs,
            "lqr": lqr,
            "iterations": it,
            "ls_success": ls_success,
            "mu": mu,
            "delta": delta,
            "K": K,
            "k": k,
        }
        return solution

    def compute_optimal_solution_for_vmap(self, x0, ubar, xd, auxvar):
        ubar = np.array(ubar) if ubar.shape[0] > ubar.shape[1] else np.array(ubar.T)
        x0 = np.squeeze(np.array(x0))
        # if len(xd.shape)==1:
        #   xd = np.tile(xd, (ubar.shape[0]+1, 1))
        # else:
        #   assert xd.shape == (ubar.shape[0]+1, x0.shape[0]), "shape of xd should either be (n_state,) for target reaching or (N+1,n_state) for tracking"
        Xs, Us, _, _, Objs, lqr, it, ls_success, mu, delta, K, k = pdp_ilqr(
            self.cost,
            self.dynamics,
            x0,
            ubar,
            xd,
            auxvar,
            self.lsfunc,
            self.nbas,
            self.reject_unsafe_traj_flags,
            self.maxiter,
            self.conv_threshold,
            self.mu0,
            self.delta0,
            self.mumin,
            self.mumax,
            self.alpha0,
            self.alphaf,
            self.alphalen,
            self.u_min,
            self.u_max,
        )

        it = np.where(it == self.maxiter, it - 1, it)

        solution = {
            "Xs": Xs,
            "Us": Us,
            "Objs": Objs,
            # 'lqr': lqr,
            "iterations": it,
            "ls_success": ls_success,
            "mu": mu,
            "delta": delta,
            "K": K,
            "k": k,
        }
        return solution


def pdp_ilqr(
    cost,
    dynamics,
    x0,
    U,
    xd,
    auxvar,
    lsfunc,
    nbas,
    reject_unsafe_traj_flags,
    maxiter,
    conv_threshold,
    mu0,
    delta0,
    mumin,
    mumax,
    alpha0,
    alphaf,
    alphalen,
    u_min,
    u_max,
):
    """Iterative Linear Quadratic Regulator.
    Args:
      cost:      cost(x, u, t) returns scalar.
      dynamics:  dynamics(x, u, t) returns next state (n, ) nd array.
      x0: initial_state - 1D np array of shape (n, ).
      U: initial_controls - 2D np array of shape (T, m).
      maxiter: maximum iterations.
    Returns:
      X: optimal state trajectory - nd array of shape (T+1, n).
      U: optimal control trajectory - nd array of shape (T, m).
      obj: final objective achieved.
      lqr: inputs to the final LQR solve.
      iteration: number of iterations upon convergence.
    """

    cost_args = auxvar
    dynamics_args = auxvar
    return ilqr_base(
        cost,
        dynamics,
        x0,
        U,
        xd,
        cost_args,
        dynamics_args,
        lsfunc,
        nbas,
        reject_unsafe_traj_flags,
        maxiter,
        conv_threshold,
        mu0,
        delta0,
        mumin,
        mumax,
        alpha0,
        alphaf,
        alphalen,
        u_min,
        u_max,
    )


@partial(jit, static_argnums=(0, 1, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20))
def ilqr_base(
    cost,
    dynamics,
    x0,
    U,
    xd,
    cost_args,
    dynamics_args,
    lsfunc,
    nbas,
    reject_unsafe_traj_flags,
    maxiter,
    conv_threshold,
    mu0,
    delta0,
    mumin,
    mumax,
    alpha0,
    alphaf,
    alphalen,
    u_min,
    u_max,
):
    """ilqr implementation."""

    T, m = U.shape
    n = x0.shape[0]

    roll = partial(parameterized_rollout, dynamics)
    quadratizer = quadratize(cost, argnums=4)
    dynamics_jacobians = linearize(dynamics, argnums=3)
    cost_gradients = linearize(cost, argnums=4)
    evaluator = partial(evaluate, cost)

    X = roll(U, x0, dynamics_args)
    timesteps = np.arange(X.shape[0])
    obj = np.sum(evaluator(X, pad(U), xd, cost_args))

    K = np.zeros((T, m, n))
    k = np.zeros((T, m))

    Xs = np.zeros((T + 1, n, maxiter))
    Us = np.zeros((T, m, maxiter))
    Objs = np.zeros((maxiter))

    def get_lqr_params(X, U):
        Q, R, M = quadratizer(X, pad(U), timesteps, xd, cost_args)
        q, r = cost_gradients(X, pad(U), timesteps, xd, cost_args)
        A, B = dynamics_jacobians(X, pad(U), np.arange(T + 1), dynamics_args)

        return (Q, q, R, r, M, A, B)

    def body(inputs):
        """Solves LQR subproblem and returns updated trajectory."""
        X, U, obj, lqr, iteration, _, _, Xs, Us, Objs, mu, delta, _, _ = inputs
        Q, q, R, r, M, A, B = lqr

        Xs = Xs.at[:, :, iteration].set(X)
        Us = Us.at[:, :, iteration].set(U)
        Objs = Objs.at[iteration].set(obj)

        K, k, _, _, deltaV0, deltaV1 = tvlqr(Q, q, R, r, M, A, B, mu)
        X, U, obj, _, forwardpass, dV = lsfunc(
            cost,
            dynamics,
            X,
            U,
            xd,
            K,
            k,
            obj,
            n - nbas,
            reject_unsafe_traj_flags,
            u_min,
            u_max,
            cost_args,
            dynamics_args,
            deltaV0,
            deltaV1,
            alpha0,
            alphaf,
            alphalen,
        )
        # print("Iteration=%d, Objective=%f, Alpha=%f, Grad-norm=%f\n" %
        #      (device_get(iteration), device_get(obj), device_get(alpha),
        #       device_get(np.linalg.norm(gradient))))

        delta, mu = lax.cond(
            forwardpass, decreaseMu, increaseMu, mu, delta, mumin, delta0
        )

        lqr = get_lqr_params(X, U)
        iteration = iteration + 1
        return X, U, obj, lqr, iteration, dV, forwardpass, Xs, Us, Objs, mu, delta, K, k

    def continuation_criterion(inputs):
        _, _, _, _, iteration, dV, forwardpass, _, _, _, mu, _, _, _ = inputs
        forwardpass = np.where(delta0 == 0, forwardpass, True)
        dVcond = np.where(dV > 0, dV >= conv_threshold, True)

        return np.logical_and(
            np.logical_and(np.logical_and(iteration < maxiter, dVcond), forwardpass),
            mu < mumax,
        )

    lqr = get_lqr_params(X, U)
    X, U, obj, lqr, it, dV, ls_success, Xs, Us, Objs, mu, delta, K, k = lax.while_loop(
        continuation_criterion,
        body,
        (X, U, obj, lqr, 0, 1e10, True, Xs, Us, Objs, mu0, delta0, K, k),
    )

    Xs = Xs.at[:, :, it].set(X)
    Us = Us.at[:, :, it].set(U)
    Objs = Objs.at[it].set(obj)

    return Xs, Us, X, U, Objs, lqr, it, ls_success, mu, delta, K, k
