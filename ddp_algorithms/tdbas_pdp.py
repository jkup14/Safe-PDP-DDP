import jax
from jax import jit, vmap
import jax.numpy as np
import jax.scipy as sp
from ddp_algorithms.jax_helpers import *
from functools import partial
from ddp_algorithms.pdp_ddp_jax import PDPDDPJax
from bas_functions.embed_dynamics import EmbedDynamics
import time

pad = lambda A: np.vstack((A, np.zeros((1,) + A.shape[1:])))
jax.numpy.set_printoptions(precision=4)


class TDBAS_PDP:
    """
    T-DBaS-PDP jax implementation as described in masters thesis
    args:
    bas_dyn_obj: EmbedDynamics object of BaS-embedded dynamics
    cost_pdp(xt, ut, k, xd, auxvar): cost to be used in pdp hamiltonian
    cost_ddp(x, u, k, xd, auxvar): cost to be used in ddp
    outerCost(x, u, k, xd): (vectorize_outercost=True) outer cost that pdp is minimizing w.r.t. parameters, auxvar
        can also be outerCost(X,U,xd), (vectorize_otuercost=False) where X,U,xd are the entire trajectory and tracking trajectory
    ls_option: either "iterative" or "parallel", decides what linesearch function to use
    OuterCost_derivatives_custom(X,U,xd): optional, custom dLdX, dLdU
    vectorize_outercost: flag for if outercost takes in one timestep or entire trajectory
    max_iters_pdp: maximum iterations pdp should take
    conv_threshold_pdp: cost convergence threshold for pdp
    max_iters_ilqr: maximum iterations ilqr should take each time it's called
    conv_threshold_ilqr: cost convergence threshold for ilqr
    precompile: flag for if the user wants to precompile the jax code, as the first iteration is always slower if you don't
    take_abs: mask of len(n_auxvar) for which parameters shouldn't be negative
    time_flag: flag for printing the time each part of the algorithm takes
    """

    def __init__(
        self,
        bas_dyn_obj: EmbedDynamics,
        cost_pdp,
        cost_ddp,
        outerCost,
        ls_option="iterative",
        OuterCost_derivatives_custom=None,
        vectorize_outercost=True,
        max_iters_pdp=100,
        conv_threshold_pdp=1e-5,
        max_iters_ilqr=100,
        conv_threshold_ilqr=1e-6,
        alphas=np.power(1, np.linspace(1, -5, 10)),
        precompile=True,
        take_abs=None,
        time_flag=False,
    ):
        self.n_state, self.n_model_state, self.n_control, self.n_auxvar = (
            bas_dyn_obj.n,
            bas_dyn_obj.dynamics.n,
            bas_dyn_obj.m,
            bas_dyn_obj.n_parameters,
        )
        self.max_iters_pdp = max_iters_pdp
        self.conv_threshold_pdp = conv_threshold_pdp
        self.T = bas_dyn_obj.N
        self.timesteps = np.arange(self.T + 1)
        self.bas_dyn_obj = bas_dyn_obj
        self.bas_dynamics = bas_dyn_obj.system_propagate
        self.model_dynamics = bas_dyn_obj.dynamics.system_propagate_jax
        self.cost_pdp = cost_pdp
        self.cost_ddp = cost_ddp
        self.OuterCost = outerCost

        assert ls_option in ["iterative", "parallel"]
        self.ls_option = ls_option
        self.ls_func = (
            self.iterative_linesearch
            if ls_option == "iterative"
            else self.parallel_linesearch
        )

        # For Custom dLdX and dLdU function
        self.isOuterCostDerivCustom = OuterCost_derivatives_custom is not None
        self.getOuterCostDerivs = (
            self.getOuterCostDerivsAutoDiff
            if not self.isOuterCostDerivCustom
            else OuterCost_derivatives_custom
        )

        if vectorize_outercost:
            self.diffOuterCost(
                self.OuterCost
            )  # For when the outercost takes in one timestep
        else:
            self.OuterCostEvaluate = self.OuterCost  # or the whole trajectory

        # Get PDP Matrix Autodiff functions
        self.diffPDP(self.model_dynamics, self.cost_pdp)

        # Intialize DDP Solver
        self.pdp_ddp_solver = PDPDDPJax(
            self.cost_ddp,
            self.bas_dynamics,
            self.n_control,
            nbas=bas_dyn_obj.n_bas,
            reject_unsafe_traj_flags=(False),
            maxiter=max_iters_ilqr,
            conv_threshold=conv_threshold_ilqr,
            verbose=False,
        )

        # Creates necessary parallel functions for parallel linesearch
        if self.ls_option == "parallel":
            self.createFunctionsForParallelLS()

        # Timing of differents steps of algorithm
        self.time_flag = time_flag

        # Learning rate schedule for linesearch, a bit high
        self.alphas = alphas

        # Precompile Jax Code
        if precompile:
            print("Compiling...", end="")
            self.precompile()

            print("Done Compiling")

        # For parameters that shouldn't be negative
        if take_abs is None:
            print("No take_abs given, assuming all parameters must be positive")
            self.take_abs_inds = np.arange(self.n_auxvar)
        else:
            assert (
                len(take_abs) == self.n_auxvar
            ), "Your take_abs list must be a list of bools of length=n_auxvar"
            self.take_abs_inds = np.argwhere(np.array(take_abs)).squeeze()

    # Hamiltonian definition and autodiff, H = cost + dyn' * costate, dynamics are w/o barrier state
    def diffPDP(self, model_dynamics, cost):
        parameterized_dyn = lambda x, u, t, auxvar: model_dynamics(x, u, t)

        def hamil(state, control, costate, t, xd, auxvar):
            return cost(state, control, t, xd, auxvar) + np.where(
                t == self.T,
                0,
                parameterized_dyn(state[: self.n_model_state], control, t, auxvar).T
                @ costate[: self.n_model_state],
            )

        self.Dyn_linearized = linearize_pdp(
            parameterized_dyn, argnums=4, e_ind=3
        )  # Fx, Fu, Fe

        self.H_quadratized = quadratize_pdp(
            hamil, argnums=6, e_ind=5
        )  # Hxx, Hxu, Hxe, Hux, Huu, Hue

        self.Dyn_x = linearize_x(parameterized_dyn)
        self.Cost_x = linearize_x(cost, argnums=4)

        self.AuxMatrices = (self.Dyn_linearized, self.H_quadratized)
        self.Jacobians_for_Costate = (self.Dyn_x, self.Cost_x)
        self.n_states = self.n_model_state

    # Get auxiliary control system to solve for dXdP and dUdP
    @partial(jit, static_argnums=(0,))
    def getAuxSys(self, X, U, Lambdas, xd, auxvar_value=1):
        auxvec = np.tile(auxvar_value, (self.T + 1, 1))
        Dyn_lin, H_quad = self.AuxMatrices

        dDyns = Dyn_lin(
            X[:-1, : self.n_states], U, self.timesteps[:-1], auxvec[:-1]
        )  # Fx, Fu, Fe
        ddHs = H_quad(
            X[:, : self.n_states], pad(U), Lambdas, self.timesteps, xd, auxvec
        )  # Hxx, Hxu, Hxe, Hux, Huu, Hue

        auxSys = {
            "dynF": np.array(dDyns[0]),
            "dynG": np.array(dDyns[1]),
            "dynE": np.array(dDyns[2]),
            "Hxx": np.array(ddHs[0][:-1]),
            "Hxu": np.array(ddHs[1][:-1]),
            "Hxe": np.array(ddHs[2][:-1]),
            "Hux": np.array(ddHs[3][:-1]),
            "Huu": np.array(ddHs[4][:-1]),
            "Hue": np.array(ddHs[5][:-1]),
            "hxx": np.array(ddHs[0][-1]),
            "hxe": np.array(ddHs[2][-1]),
        }
        return auxSys

    # Solve auxiliary control system OC problem
    # @partial(jit, static_argnums = (0,4))
    def auxSysOC(self, X, U, xd, auxvar):
        if self.time_flag:
            start = time.time()

        Lambda = self.getCostate(X, U, xd, auxvar)  # Get Costate from Trajectory
        auxsys_OC = self.getAuxSys(X, U, Lambda, xd, auxvar)  # Get PDP Matrices

        if self.time_flag:
            print("Autodiff time:", time.time() - start)
            start = time.time()

        # If initial condition is parameterized
        X0 = np.zeros((self.n_states, auxvar.shape[0]))

        aux_sol = solve_aux_sys(X0, self.T, auxsys_OC)
        dX_and_dU = (aux_sol["state_traj_opt"], aux_sol["control_traj_opt"])

        if self.time_flag:
            print("Aux LQR time:", time.time() - start)

        return dX_and_dU

    # Returns costate trajectory = dHdX
    @partial(jit, static_argnums=(0,))
    def getCostate(self, X, U, xd, auxvar):
        Dyn_x, Cost_x = self.Jacobians_for_Costate
        dDynx = Dyn_x(X[:-1, : self.n_states], U, self.timesteps[:-1], auxvar)
        dCostx = Cost_x(X[:, : self.n_states], pad(U), self.timesteps, xd, auxvar)
        Lambda = np.zeros((self.T + 1, self.n_states))
        Lambda = Lambda.at[-1, :].set(dCostx[-1])

        def costate_func(Lambda, k):
            Lambda = Lambda.at[k - 1, :].set(
                dCostx[k - 1] + np.dot(dDynx[k - 1], Lambda[k, :])
            )
            return Lambda, ()

        ksteps = np.arange(self.T, 0, -1)
        Lambda, _ = jax.lax.scan(costate_func, Lambda, ksteps)
        return Lambda

    # Calculates jacobians of outer cost
    def getOuterCostDerivsAutoDiff(self, X, U, xd):
        return self.OuterCost_lin(X, pad(U), self.timesteps, xd)

    # Chain rule to get derivative of outercost w.r.t. parameters
    def getDLDP(self, X, U, xd, dXdP_and_dUdP):
        dXdP, dUdP = dXdP_and_dUdP
        dLdX, dLdU = self.getOuterCostDerivs(X, U, xd)
        dLdP = 0

        def chain_rule(dLdX_t, dLdU_t, dXdP_t, dUdP_t):
            dLdP_t = np.matmul(dLdX_t[: self.n_states], dXdP_t) + np.matmul(
                dLdU_t[: self.n_states], dUdP_t
            )
            return dLdP_t

        dLdP_map = jax.vmap(chain_rule)(dLdX[:-1], dLdU[:-1], dXdP[:-1], dUdP)
        dLdP = np.sum(dLdP_map, axis=0) + np.dot(dLdX[-1, : self.n_states], dXdP[-1])

        return dLdP

    # Main algorithm, needs initial state, control vec, parameter guess, and goal state
    def PDP_solve(self, x0, U_init, xd, init_parameter):
        """
        args:
        x0: BaS embedded initial state
        U_init: initial guess of control trajectory
        xd: BaS embedded goal state or trajectory to track
        init_parameter: initial guess of parameters
        """
        # Convert to jax array
        x0, U, init_parameter = (
            np.array(x0),
            np.array(U_init),
            np.array(init_parameter),
        )

        xd = xd.squeeze()
        # If not tracking, just track a single point
        if len(xd.shape) == 1:
            xd = np.tile(xd, (U.shape[0] + 1, 1))
        else:
            assert xd.shape == (
                U.shape[0] + 1,
                x0.shape[0],
            ), "shape of xd should either be (n_state,) for target reaching or (N+1,n_state) for tracking"

        # We don't want to track a non-zero barrier state
        xd = xd.at[:, self.n_model_state :].set(0)

        # Storage
        loss_trace_outer = []
        loss_trace_inner = []
        parameter_trace = np.empty((self.max_iters_pdp, init_parameter.shape[0]))
        state_traj_array = []
        control_traj_array = []

        assert len(init_parameter) == self.n_auxvar
        # Take abs of params if needed
        current_parameter = init_parameter.at[self.take_abs_inds].set(
            np.abs(init_parameter[self.take_abs_inds])
        )

        # Amount of DDP iterations
        total_iterations_inner = 0

        # DDP Solution for first guess of parameters
        sol = self.pdp_ddp_solver.compute_optimal_solution(x0, U, xd, current_parameter)
        if not sol["ls_success"]:
            print(
                "This is generally a bad sign, consider adjusting initial parameter guess"
            )
        it = sol["iterations"]
        X = sol["Xs"][:, :, it]
        U = sol["Us"][:, :, it]

        # Outer cost evaluation of initial solve
        if self.time_flag:
            start = time.time()
        loss_trace_outer += [self.OuterCostEvaluate(X, U, xd)]
        if self.time_flag:
            print("Outer cost time:", time.time() - start)

        loss_trace_inner += [sol["Objs"][it]]
        total_iterations_inner += it

        print(
            "Outer Iter #:",
            0,
            "Outer:",
            loss_trace_outer[-1],
            "Total Inner Iterations: ",
            total_iterations_inner,
            "Inner Loss:",
            loss_trace_inner[-1],
            "Auxvar: ",
            current_parameter,
        )

        ii = 0
        dL = 1
        # PDP loop
        while ii < self.max_iters_pdp and dL > self.conv_threshold_pdp:
            # Storage
            parameter_trace = parameter_trace.at[ii, :].set(current_parameter)
            state_traj_array += [X]
            control_traj_array += [U]

            # Solve Auxiliary Control Problem using PDP to get derivative of traj w.r.t. parameters
            dXdP_and_dUdP = self.auxSysOC(X, U, xd, current_parameter)

            # Chain Rule
            if self.time_flag:
                start = time.time()

            dldp = self.getDLDP(X, U, xd, dXdP_and_dUdP)

            if self.time_flag:
                print("Chain rule time:", time.time() - start)

            # Linesearch on parameters, give previous U solution as warm start
            if self.time_flag:
                start = time.time()

            (
                Xnew,
                Unew,
                current_parameter_new,
                loss_outer_new,
                sol_new,
                pdp_ls_success,
            ) = self.ls_func(x0, U, xd, current_parameter, dldp, loss_trace_outer[-1])

            if self.time_flag:
                print("Linesearch time:", time.time() - start)

            # More Storage
            if pdp_ls_success:
                X = Xnew
                U = Unew
                sol = sol_new
                current_parameter = current_parameter_new
                loss_trace_outer += [loss_outer_new]
                dL = loss_trace_outer[-2] - loss_trace_outer[-1]
                total_iterations_inner += sol_new["iterations"]
                loss_trace_inner += [sol["Objs"][sol["iterations"]]]

            ii += 1

            if ii % 1 == 0:
                print(
                    "Outer Iter #:",
                    ii,
                    "Outer:",
                    loss_trace_outer[-1],
                    "dL:",
                    dL,
                    "Total Inner Iterations: ",
                    total_iterations_inner,
                    "Inner Loss:",
                    loss_trace_inner[-1],
                    "Auxvar: ",
                    current_parameter,
                    "dldp: ",
                    dldp,
                )

            if not pdp_ls_success:
                # PDP failed
                print("PDP linesearch couldn't reduce cost further, exiting...")
                break

        # Store final result
        parameter_trace = parameter_trace.at[ii, :].set(current_parameter)
        loss_trace_inner += [sol["Objs"][it]]
        total_iterations_inner += it
        state_traj_array += [X]
        control_traj_array += [U]

        return {
            "Xs": np.array(state_traj_array),
            "Us": np.array(control_traj_array),
            "Parameters": parameter_trace,
            "Outer Cost": np.array(loss_trace_outer),
            "Inner Cost": np.array(loss_trace_inner),
            "Total ILQR Iterations": total_iterations_inner,
            "PDP Iterations": ii,
        }

    # Traditional sequential linesearch on parameter search direction given by PDP
    def iterative_linesearch(self, x0, Uwarm, xd, current_parameter, dldp, loss_old):
        a = 0
        X = None
        U = None
        current_parameter_new = None
        loss_outer_new = None
        sol = None
        pdp_ls_success = True
        # Learning rate a is applied starting at a larger value and decreasing until a cost decrease is observed
        print("Trying alpha: ", end="")
        while a < len(self.alphas):
            alpha = self.alphas[a]
            print(np.round(alpha, 3), "", end="")
            # Gradient Descent
            gradient_step = alpha * dldp
            current_parameter_temp = current_parameter - gradient_step
            current_parameter_temp = current_parameter_temp.at[self.take_abs_inds].set(
                np.abs(current_parameter_temp[self.take_abs_inds])
            )

            # Solve ddp using new parameters
            sol_temp = self.pdp_ddp_solver.compute_optimal_solution(
                x0, Uwarm, xd, current_parameter_temp
            )

            it = sol_temp["iterations"]
            ddp_ls_success = sol_temp["ls_success"]

            X_sol_temp, U_sol_temp = (
                sol_temp["Xs"][:, :, it],
                sol_temp["Us"][:, :, it],
            )
            loss_new = self.OuterCostEvaluate(X_sol_temp, U_sol_temp, xd)

            # Check if there a loss reduction and ddp linesearch was successful
            success = np.logical_and(loss_new < loss_old, ddp_ls_success)
            if success:
                current_parameter_new = current_parameter_temp
                loss_outer_new = loss_new
                sol = sol_temp
                X = X_sol_temp
                U = U_sol_temp
                break
            # If cost did not decrease or ddp linesearch failed, increment a and decrease learning rate
            a += 1

        print("")
        # If parameter update search direction failed
        if a == len(self.alphas):
            print("PDP Linesearch Failed!")
            pdp_ls_success = False
        return (X, U, current_parameter_new, loss_outer_new, sol, pdp_ls_success)

    # Parallel linesearch to get better results, maybe slower depending on your jax/GPU
    def parallel_linesearch(self, x0, Uwarm, xd, current_parameter, dldp, loss_old):
        X, U, current_parameter_new, loss_outer_new, sol = [None] * 5

        n_alphas = self.alphas.shape[0]
        # change in parameters [nalphas, naux]
        gradient_steps_parallel = self.alphas[:, np.newaxis] * dldp[np.newaxis, :]
        # Go down gradient on parameters
        parameter_candidates_parallel = (
            current_parameter[np.newaxis, :] - gradient_steps_parallel
        )
        # Abs any that need it
        parameter_candidates_parallel = parameter_candidates_parallel.at[
            :, self.take_abs_inds
        ].set(np.abs(parameter_candidates_parallel[:, self.take_abs_inds]))

        # Parallel iLQR solve for all alphas
        if self.time_flag:
            start = time.time()
        sol_candidates_parallel = self.parallel_solver(
            x0, Uwarm, xd, parameter_candidates_parallel
        )
        if self.time_flag:
            print("iLQR time:", time.time() - start)

        # Extract solution at final ilqr iteration
        it_parallel = sol_candidates_parallel["iterations"]
        X_sol_alpha, U_sol_alpha = (
            sol_candidates_parallel["Xs"][np.arange(n_alphas), :, :, it_parallel],
            sol_candidates_parallel["Us"][np.arange(n_alphas), :, :, it_parallel],
        )
        # Which alphas resulted in ilqr line search success
        ls_successes_parallel = sol_candidates_parallel["ls_success"]

        # Evaluate outercost
        if self.time_flag:
            start = time.time()
        loss_news_parallel = self.OuterCostEvalMapped(X_sol_alpha, U_sol_alpha, xd)
        if self.time_flag:
            print("Outer cost time:", time.time() - start)

        # Throw out any that failed ddp linesearch by giving inf cost
        loss_news_parallel = loss_news_parallel.at[
            np.logical_not(ls_successes_parallel)
        ].set(np.inf)

        # Extract min loss and check for pdp linesearch success
        min_loss_ind = np.nanargmin(loss_news_parallel)
        loss_outer_candidate = loss_news_parallel[min_loss_ind]
        pdp_ls_success = loss_outer_candidate < loss_old
        if pdp_ls_success:
            # Extract solution dictionary, as the one from vmap gives a dict of lists
            sol = {
                key: val[min_loss_ind] for (key, val) in sol_candidates_parallel.items()
            }
            X = sol["Xs"][:, :, sol["iterations"]]
            U = sol["Us"][:, :, sol["iterations"]]
            current_parameter_new = parameter_candidates_parallel[min_loss_ind]
            loss_outer_new = loss_outer_candidate
            print("Chose alpha=", self.alphas[min_loss_ind])

        return (X, U, current_parameter_new, loss_outer_new, sol, pdp_ls_success)

    # Runs algorithm with dummy variables to make jax compile the code, subsequent runs are much faster
    def precompile(self):
        if self.time_flag:
            start = time.time()

        dummyXparallel = np.ones((self.alphas.shape[0], self.T + 1, self.n_state))
        dummyX = dummyXparallel[0]
        dummyx0 = dummyX[0]
        dummyU = np.ones((self.T, self.n_control))
        dummyxd = dummyX
        dummyauxvar = np.ones((self.n_auxvar))
        dummyauxvarparallel = np.ones((self.alphas.shape[0], self.n_auxvar))
        dummyUparallel = np.ones((self.alphas.shape[0], self.T, self.n_control))

        self.pdp_ddp_solver.compute_optimal_solution(
            dummyx0,
            dummyU,
            dummyxd,
            dummyauxvar,
        )

        if self.ls_option == "parallel":
            self.parallel_solver(
                dummyx0,
                dummyU,
                dummyxd,
                dummyauxvarparallel,
            )
            self.OuterCostEvalMapped(
                dummyXparallel,
                dummyUparallel,
                np.ones((self.T + 1, self.n_state)),
            )

        # Outer cost calculation
        self.OuterCostEvaluate(
            dummyX,
            dummyU,
            dummyxd,
        )

        if self.isOuterCostDerivCustom:
            self.getOuterCostDerivs(
                dummyX,
                dummyU,
                dummyxd,
            )

        dummydXdP_and_dUdP = self.auxSysOC(dummyX, dummyU, dummyxd, dummyauxvar)
        self.getDLDP(dummyX, dummyU, dummyxd, dummydXdP_and_dUdP)

        if self.time_flag:
            print("Compile time:", time.time() - start)

    # Creates parallel ilqr solver and outer cost evaluator
    def createFunctionsForParallelLS(self):
        self.parallel_solver = vmap(
            self.pdp_ddp_solver.compute_optimal_solution_for_vmap,
            in_axes=(None, None, None, 0),
        )
        self.OuterCostEvalMapped = vmap(self.OuterCostEvaluate, in_axes=(0, 0, None))

    # Autodifferentiates outer cost
    def diffOuterCost(self, cost):
        self.OuterCost_lin = linearize(cost, argnums=4)
        eval_func = partial(evaluate, cost)
        self.OuterCostEvaluate = lambda X, U, xd: np.sum(eval_func(X, pad(U), xd))


# PDP Auxiliary Control System Solved, as described in the paper
def solve_aux_sys(ini_state, horizon, auxsys_OC):
    dynF = auxsys_OC["dynF"]
    dynG = auxsys_OC["dynG"]
    dynE = auxsys_OC["dynE"]
    Hxx = auxsys_OC["Hxx"]
    Huu = auxsys_OC["Huu"]
    Hxu = auxsys_OC["Hxu"]
    Hux = auxsys_OC["Hux"]
    Hxe = auxsys_OC["Hxe"]
    Hue = auxsys_OC["Hue"]
    hxx = auxsys_OC["hxx"]
    hxe = auxsys_OC["hxe"]
    n_state = dynF.shape[-1]
    n_control = dynG.shape[-1]
    n_batch = dynE.shape[-1] if dynE is not None else None
    n_batch = np.size(ini_state, 1) if ini_state.ndim == 2 else n_batch

    I = np.eye(n_state)
    IplusP_nextRinv = np.zeros((horizon, n_state, n_state))
    PP = np.zeros((horizon, n_state, n_state))
    WW = np.zeros((horizon, n_state, n_batch))
    PP = PP.at[-1].set(hxx)
    WW = WW.at[-1].set(hxe)

    # Non-recursive part of backward pass to be vmap'd
    def backward_pass_map(dynF_t, dynG_t, dynE_t, Huu_t, Hxu_t, Hue_t, Hxx_t, Hxe_t):
        invHuu_t = sp.linalg.inv(Huu_t)
        GinvHuu = np.matmul(dynG_t, invHuu_t)
        HxuinvHuu = np.matmul(Hxu_t, invHuu_t)
        A_t = dynF_t - np.matmul(GinvHuu, np.transpose(Hxu_t))
        R_t = np.matmul(GinvHuu, np.transpose(dynG_t))
        M_t = dynE_t - np.matmul(GinvHuu, Hue_t)
        Q_t = Hxx_t - np.matmul(HxuinvHuu, np.transpose(Hxu_t))
        N_t = Hxe_t - np.matmul(HxuinvHuu, Hue_t)
        return invHuu_t, A_t, R_t, M_t, Q_t, N_t

    invHuu, A, R, M, Q, N = vmap(backward_pass_map)(
        dynF, dynG, dynE, Huu, Hxu, Hue, Hxx, Hxe
    )

    # Recursive part of backward pass for jax.lax.scan
    def pdp_backward_pass(carry, t):
        PP, WW, IplusP_nextRinv = carry
        P_next = PP[t]
        W_next = WW[t]
        IplusP_nextRinv_t = sp.linalg.inv(I + np.matmul(P_next, R[t]))
        temp_mat = np.matmul(np.transpose(A[t]), IplusP_nextRinv_t)
        P_curr = Q[t] + np.matmul(temp_mat, np.matmul(P_next, A[t]))
        W_curr = N[t] + np.matmul(temp_mat, W_next + np.matmul(P_next, M[t]))

        PP = PP.at[t - 1].set(P_curr)
        WW = WW.at[t - 1].set(W_curr)
        IplusP_nextRinv = IplusP_nextRinv.at[t].set(IplusP_nextRinv_t)

        carry = (PP, WW, IplusP_nextRinv)
        stack = ()
        return carry, stack

    tsteps = np.arange(horizon - 1, 0, -1)
    (PP, WW, IplusP_nextRinv), () = jax.lax.scan(
        pdp_backward_pass, (PP, WW, IplusP_nextRinv), tsteps
    )

    # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
    # consistent with the PDP paper in Lemma 4.2
    def pdp_forward_pass(x_t, t):
        P_next = PP[t]
        W_next = WW[t]

        u_t = -np.matmul(
            invHuu[t], np.matmul(np.transpose(Hxu[t]), x_t) + Hue[t]
        ) - np.linalg.multi_dot(
            [
                invHuu[t],
                np.transpose(dynG[t]),
                IplusP_nextRinv[t],
                (
                    np.matmul(np.matmul(P_next, A[t]), x_t)
                    + np.matmul(P_next, M[t])
                    + W_next
                ),
            ]
        )

        x_next = np.matmul(dynF[t], x_t) + np.matmul(dynG[t], u_t) + dynE[t]
        lambda_next = np.matmul(P_next, x_next) + W_next
        return x_next, (x_next, u_t, lambda_next)

    _, (state_traj_opt, control_traj_opt, costate_traj_opt) = jax.lax.scan(
        pdp_forward_pass, ini_state, np.arange(horizon)
    )
    state_traj_opt = np.concatenate([ini_state[np.newaxis, ...], state_traj_opt])

    opt_sol = {
        "state_traj_opt": state_traj_opt,
        "control_traj_opt": control_traj_opt,
        "costate_traj_opt": costate_traj_opt,
    }
    return opt_sol
