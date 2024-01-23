import jax
from jax import jit
from jax import lax
from functools import partial
import jax.numpy as np
import jax.scipy as sp
from ddp_algorithms.jax_helpers import pad, evaluate


@partial(jit, static_argnums=(0, 1, 8, 9, 10, 11, 16, 17, 18))
def line_search_ddp_bas(
    cost,
    dynamics,
    X,
    U,
    xd,
    K,
    k,
    obj,
    n,
    reject_unsafe_traj_flags,
    u_min,
    u_max,
    cost_args=(),
    dynamics_args=(),
    deltaV0=0,
    deltaV1=0,
    alpha0=0,
    alphaf=-3,
    alpha_len=11,
):
    """Performs line search with respect to DDP rollouts."""

    alphas = np.power(10, np.linspace(alpha0, alphaf, alpha_len))
    obj = np.where(np.isnan(obj), np.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, *margs: np.sum(costs(X, pad(U), xd, *margs))

    def line_search(inputs):
        """Line search to find improved control sequence."""
        _, _, obj_old, alpha_ind, _, _ = inputs
        alpha = alphas[alpha_ind]
        Xnew, Unew = ddp_rollout(
            dynamics, X, U, K, k, alpha, u_min, u_max, *dynamics_args
        )
        obj_new = total_cost(Xnew, Unew, xd, *cost_args)

        obj_new = np.where(np.isnan(obj_new), obj, obj_new)
        true_reduction = obj_old - obj_new
        expected_reduction = -alpha * (deltaV0 + alpha * deltaV1)
        z = true_reduction / expected_reduction
        traj_not_rejected = is_safe_traj(Xnew[:, n:], reject_unsafe_traj_flags)
        forwardpass = np.logical_and(z >= 0, traj_not_rejected)

        # Only return new trajs if leads to a strict cost decrease
        X_return = np.where(forwardpass, Xnew, X)
        U_return = np.where(forwardpass, Unew, U)
        obj_return = np.where(forwardpass, obj_new, obj_old)
        dV = true_reduction

        return X_return, U_return, obj_return, alpha_ind + 1, forwardpass, dV

    def continuation_criteria(inputs):
        # return inputs[4] is False
        return np.logical_and(np.logical_not(inputs[4]), inputs[3] < alpha_len)

    return lax.while_loop(continuation_criteria, line_search, (X, U, obj, 0, False, 0))


@partial(jit, static_argnums=(0, 1, 8, 9, 10, 11, 16, 17, 18))
def parameterized_line_search_ddp_bas(
    cost,
    dynamics,
    X,
    U,
    xd,
    K,
    k,
    obj,
    n,
    reject_unsafe_traj_flags,
    u_min,
    u_max,
    cost_args=(),
    dynamics_args=(),
    deltaV0=0,
    deltaV1=0,
    alpha0=0,
    alphaf=-3,
    alpha_len=11,
):
    """Performs line search with respect to DDP rollouts."""

    alphas = np.power(10, np.linspace(alpha0, alphaf, alpha_len))
    obj = np.where(np.isnan(obj), np.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, xd, margs: np.sum(costs(X, pad(U), xd, margs))

    def line_search(inputs):
        """Line search to find improved control sequence."""
        _, _, obj_old, alpha_ind, _, _ = inputs
        alpha = alphas[alpha_ind]
        Xnew, Unew = parameterized_ddp_rollout(
            dynamics, X, U, K, k, alpha, u_min, u_max, dynamics_args
        )
        obj_new = total_cost(Xnew, Unew, xd, cost_args)

        obj_new = np.where(np.isnan(obj_new), obj, obj_new)
        true_reduction = obj_old - obj_new
        expected_reduction = -alpha * (deltaV0 + alpha * deltaV1)
        z = true_reduction / expected_reduction
        traj_not_rejected = is_safe_traj(Xnew[:, n:], reject_unsafe_traj_flags)
        forwardpass = np.logical_and(z >= 0, traj_not_rejected)

        # Only return new trajs if leads to a strict cost decrease
        X_return = np.where(forwardpass, Xnew, X)
        U_return = np.where(forwardpass, Unew, U)
        obj_return = np.where(forwardpass, obj_new, obj_old)
        dV = true_reduction

        return X_return, U_return, obj_return, alpha_ind + 1, forwardpass, dV

    def continuation_criteria(inputs):
        # return inputs[4] is False
        return np.logical_and(np.logical_not(inputs[4]), inputs[3] < alpha_len)

    return lax.while_loop(continuation_criteria, line_search, (X, U, obj, 0, False, 0))


def is_safe_traj(bas_traj, reject_unsafe_traj_flag):
    return np.logical_not(np.any(np.logical_and(bas_traj < 0, reject_unsafe_traj_flag)))


@partial(jit, static_argnums=(0, 1, 7, 8, 9, 10, 15, 16, 17))
def line_search_ddp_no_bas(
    cost,
    dynamics,
    X,
    U,
    K,
    k,
    obj,
    n,
    reject_unsafe_traj_flags,
    u_min,
    u_max,
    cost_args=(),
    dynamics_args=(),
    deltaV0=0,
    deltaV1=0,
    alpha0=0,
    alphaf=-3,
    alpha_len=11,
):
    """Performs line search with respect to DDP rollouts."""

    alphas = np.power(10, np.linspace(alpha0, alphaf, alpha_len))
    obj = np.where(np.isnan(obj), np.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, *margs: np.sum(costs(X, pad(U), *margs))

    def line_search(inputs):
        """Line search to find improved control sequence."""
        _, _, obj_old, alpha_ind, _, _ = inputs
        alpha = alphas[alpha_ind]
        Xnew, Unew = ddp_rollout(
            dynamics, X, U, K, k, alpha, u_min, u_max, *dynamics_args
        )
        obj_new = total_cost(Xnew, Unew, *cost_args)

        obj_new = np.where(np.isnan(obj_new), obj, obj_new)
        true_reduction = obj_old - obj_new
        expected_reduction = -alpha * (deltaV0 + alpha * deltaV1)
        z = true_reduction / expected_reduction
        forwardpass = np.logical_and(z >= 0, expected_reduction > 0)

        # Only return new trajs if leads to a strict cost decrease
        X_return = np.where(forwardpass, Xnew, X)
        U_return = np.where(forwardpass, Unew, U)
        obj_return = np.where(forwardpass, obj_new, obj_old)
        dV = true_reduction

        return X_return, U_return, obj_return, alpha_ind + 1, forwardpass, dV

    def continuation_criteria(inputs):
        # return inputs[4] is False
        return np.logical_and(np.logical_not(inputs[4]), inputs[3] < alpha_len)

    return lax.while_loop(continuation_criteria, line_search, (X, U, obj, 0, False, 0))


@partial(jit, static_argnums=(0, 1, 7, 8, 9, 10, 15, 16, 17))
def parameterized_line_search_ddp_no_bas(
    cost,
    dynamics,
    X,
    U,
    K,
    k,
    obj,
    n,
    reject_unsafe_traj_flags,
    u_min,
    u_max,
    cost_args=(),
    dynamics_args=(),
    deltaV0=0,
    deltaV1=0,
    alpha0=0,
    alphaf=-3,
    alpha_len=11,
):
    """Performs line search with respect to DDP rollouts."""

    alphas = np.power(10, np.linspace(alpha0, alphaf, alpha_len))
    obj = np.where(np.isnan(obj), np.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, margs: np.sum(costs(X, pad(U), margs))

    def line_search(inputs):
        """Line search to find improved control sequence."""
        _, _, obj_old, alpha_ind, _, _ = inputs
        alpha = alphas[alpha_ind]
        Xnew, Unew = parameterized_ddp_rollout(
            dynamics, X, U, K, k, alpha, u_min, u_max, dynamics_args
        )
        obj_new = total_cost(Xnew, Unew, cost_args)

        obj_new = np.where(np.isnan(obj_new), obj, obj_new)
        true_reduction = obj_old - obj_new
        expected_reduction = -alpha * (deltaV0 + alpha * deltaV1)
        z = true_reduction / expected_reduction
        forwardpass = np.logical_and(z >= 0, expected_reduction > 0)

        # Only return new trajs if leads to a strict cost decrease
        X_return = np.where(forwardpass, Xnew, X)
        U_return = np.where(forwardpass, Unew, U)
        obj_return = np.where(forwardpass, obj_new, obj_old)
        dV = true_reduction

        return X_return, U_return, obj_return, alpha_ind + 1, forwardpass, dV

    def continuation_criteria(inputs):
        # return inputs[4] is False
        return np.logical_and(np.logical_not(inputs[4]), inputs[3] < alpha_len)

    return lax.while_loop(continuation_criteria, line_search, (X, U, obj, 0, False, 0))


@jit
def lqr_step(Vxx, Vx, Lxx, Lx, Luu, Lu, Lxu, fx, fu, mu=0.0):
    """Single LQR Step.

    Args:
      P: [n, n] numpy array.
      p: [n] numpy array.
      Q: [n, n] numpy array.
      q: [n] numpy array.
      R: [m, m] numpy array.
      r: [m] numpy array.
      M: [n, m] numpy array.
      A: [n, n] numpy array.
      B: [n, m] numpy array.
      c: [n] numpy array.
      delta: Enforces positive definiteness by ensuring smallest eigenval > delta.

    Returns:
      P, p: updated matrices encoding quadratic value function.
      K, k: state feedback gain and affine term.
    """
    fxTVxx = fx.T @ Vxx

    Qx = Lx + fx.T @ Vx
    Qu = Lu + fu.T @ Vx
    Qxx = Lxx + fxTVxx @ fx
    Qxu = Lxu + fxTVxx @ fu
    Qux = Qxu.T
    Quu = Luu + fu.T @ Vxx @ fu + mu * np.eye(Luu.shape[0])

    invQuu = sp.linalg.inv(Quu)
    K = -invQuu @ Qux
    k = -invQuu @ Qu

    Vx = Qx + Qxu @ k
    Vxx = Qxx + Qxu @ K
    Vxx = 0.5 * (Vxx + Vxx.T)

    deltaV0 = k.T @ Qu
    deltaV1 = 0.5 * k.T @ Quu @ k

    return Vxx, Vx, K, k, deltaV0, deltaV1


def twobytwoinv(M):
    [[a, b], [c, d]] = M
    oneoverdet = 1 / (a * d - b * c)
    return oneoverdet * np.array([[d, -b], [-c, a]])


@jit
def tvlqr(Q, q, R, r, M, A, B, mu=0.0):
    """Discrete-time Finite Horizon Time-varying LQR.

    Note - for vectorization convenience, the leading dimension of R, r, M, A, B,
    C can be (T + 1) but the last row will be ignored.

    Args:
      Q: [T+1, n, n] numpy array.
      q: [T+1, n] numpy array.
      R: [T, m, m] numpy array.
      r: [T, m] numpy array.
      M: [T, n, m] numpy array.
      A: [T, n, n] numpy array.
      B: [T, n, m] numpy array.
      c: [T, n] numpy array.

    Returns:
      K: [T, m, n] Gains
      k: [T, m] Affine terms (u_t = np.matmul(K[t],  x_t) + k[t])
      P: [T+1, n, n] numpy array encoding initial value function.
      p: [T+1, n] numpy array encoding initial value function.
    """

    T = Q.shape[0] - 1
    m = R.shape[1]
    n = Q.shape[1]

    Vxx = np.zeros((T + 1, n, n))
    Vx = np.zeros((T + 1, n))
    K = np.zeros((T, m, n))
    k = np.zeros((T, m))

    Vxx = Vxx.at[-1].set(Q[T])
    Vx = Vx.at[-1].set(q[T])

    def body(tt, inputs):
        K, k, Vxx, Vx, deltaV0, deltaV1 = inputs
        t = T - 1 - tt
        Vxx_t, Vx_t, K_t, k_t, deltaV0_t, deltaV1_t = lqr_step(
            Vxx[t + 1], Vx[t + 1], Q[t], q[t], R[t], r[t], M[t], A[t], B[t], mu
        )
        K = K.at[t].set(K_t)
        k = k.at[t].set(k_t)
        Vxx = Vxx.at[t].set(Vxx_t)
        Vx = Vx.at[t].set(Vx_t)
        deltaV0 += deltaV0_t
        deltaV1 += deltaV1_t

        return K, k, Vxx, Vx, deltaV0, deltaV1

    return lax.fori_loop(0, T, body, (K, k, Vxx, Vx, 0, 0))


@partial(jit, static_argnums=(0, 6, 7))
def ddp_rollout(dynamics, X, U, K, k, alpha, u_min, u_max, *args):
    """Rollouts used in Differential Dynamic Programming.

    Args:
      dynamics: function with signature dynamics(x, u, t, *args).
      X: [T+1, n] current state trajectory.
      U: [T, m] current control sequence.
      K: [T, m, n] state feedback gains.
      k: [T, m] affine terms in state feedback.
      alpha: line search parameter.
      *args: passed to dynamics.

    Returns:
      Xnew, Unew: updated state trajectory and control sequence, via:

        del_u = alpha * k[t] + np.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        x = dynamics(Xnew[t], u, t)
    """
    n = X.shape[1]
    T, m = U.shape
    Xnew = np.zeros((T + 1, n))
    Unew = np.zeros((T, m))
    Xnew = Xnew.at[0].set(X[0])

    def body(t, inputs):
        Xnew, Unew = inputs
        del_u = alpha * k[t] + np.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        u = np.clip(u, np.array(u_min), np.array(u_max))
        x = dynamics(Xnew[t], u, t, *args)
        Unew = Unew.at[t].set(u)
        Xnew = Xnew.at[t + 1].set(x)
        return Xnew, Unew

    return lax.fori_loop(0, T, body, (Xnew, Unew))


@partial(jit, static_argnums=(0, 6, 7))
def parameterized_ddp_rollout(dynamics, X, U, K, k, alpha, u_min, u_max, args):
    """Rollouts used in Differential Dynamic Programming.

    Args:
      dynamics: function with signature dynamics(x, u, t, *args).
      X: [T+1, n] current state trajectory.
      U: [T, m] current control sequence.
      K: [T, m, n] state feedback gains.
      k: [T, m] affine terms in state feedback.
      alpha: line search parameter.
      *args: passed to dynamics.

    Returns:
      Xnew, Unew: updated state trajectory and control sequence, via:

        del_u = alpha * k[t] + np.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        x = dynamics(Xnew[t], u, t)
    """
    n = X.shape[1]
    T, m = U.shape
    Xnew = np.zeros((T + 1, n))
    Unew = np.zeros((T, m))
    Xnew = Xnew.at[0].set(X[0])

    def body(t, inputs):
        Xnew, Unew = inputs
        del_u = alpha * k[t] + np.matmul(K[t], Xnew[t] - X[t])
        u = U[t] + del_u
        u = np.clip(u, np.array(u_min), np.array(u_max))
        x = dynamics(Xnew[t], u, t, args)
        Unew = Unew.at[t].set(u)
        Xnew = Xnew.at[t + 1].set(x)
        return Xnew, Unew

    return lax.fori_loop(0, T, body, (Xnew, Unew))


def rollout(dynamics, U, x0):
    """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0.

    Args:
      dynamics: a function f(x, u, t) to rollout.
      U: (T, m) np array for control sequence.
      x0: (n, ) np array for initial state.

    Returns:
       X: (T+1, n) state trajectory.
    """
    return _rollout(dynamics, U, x0)


def _rollout(dynamics, U, x0, *args):
    def dynamics_for_scan(x, ut):
        u, t = ut
        x_next = dynamics(x, u, t, *args)

        return x_next, x_next

    return np.vstack(
        (x0, lax.scan(dynamics_for_scan, x0, (U, np.arange(U.shape[0])))[1])
    )


def parameterized_rollout(dynamics, U, x0, args):
    """Rolls-out x[t+1] = dynamics(x[t], U[t], t), x[0] = x0.

    Args:
      dynamics: a function f(x, u, t) to rollout.
      U: (T, m) np array for control sequence.
      x0: (n, ) np array for initial state.

    Returns:
       X: (T+1, n) state trajectory.
    """
    return _parameterized_rollout(dynamics, U, x0, args)


def _parameterized_rollout(dynamics, U, x0, args):
    def dynamics_for_scan(x, ut):
        u, t = ut
        x_next = dynamics(x, u, t, args)

        return x_next, x_next

    return np.vstack(
        (x0, lax.scan(dynamics_for_scan, x0, (U, np.arange(U.shape[0])))[1])
    )


def increaseMu(mu, delta, mumin, delta0):
    deltanew = np.maximum(delta0, delta * delta0)
    munew = np.maximum(mumin, mu * delta)
    return deltanew, munew


def decreaseMu(mu, delta, mumin, delta0):
    deltanew = np.minimum(1 / delta0, delta / delta0)
    munew = np.where(mu * delta > mumin, mu * delta, 0)
    return deltanew, munew
