import os
import numpy as np
import scipy.linalg

from acados_template import AcadosOcp, AcadosOcpSolver

from chain_mass_model import export_chain_mass_model

from plot_utils import *
from utils import create_casados_integrator, generate_butcher_tableu
from chain_utils import (
    compute_steady_state,
    save_results_as_json,
    export_chain_mass_integrator,
)
import matplotlib.pyplot as plt


from casadi import *
import time

def export_chain_mass_ocp_solver(chain_params, integrator_opts):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # chain parameters
    n_mass = chain_params["n_mass"]
    M = chain_params["n_mass"] - 2  # number of intermediate masses
    Ts = chain_params["Ts"]
    N = chain_params["N"]
    with_wall = chain_params["with_wall"]
    yPosWall = chain_params["yPosWall"]
    m = chain_params["m"]
    D = chain_params["D"]
    L = chain_params["L"]

    nlp_iter = chain_params["nlp_iter"]
    nlp_tol = chain_params["nlp_tol"]
    qp_solver = chain_params["qp_solver"]

    # export model
    model = export_chain_mass_model(chain_params)

    # set model
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    Tf = N * Ts

    # initial state
    xPosFirstMass = np.zeros((3, 1))
    xEndRef = np.zeros((3, 1))
    xEndRef[0] = L * (M + 1) * 6

    xrest = compute_steady_state(chain_params, xPosFirstMass, xEndRef)

    x0 = xrest

    # set dimensions
    ocp.dims.N = N

    # set cost module
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Q = 2 * np.diagflat(np.ones((nx, 1)))
    q_diag = np.ones((nx, 1))
    strong_penalty = M + 1
    q_diag[3 * M] = strong_penalty
    q_diag[3 * M + 1] = strong_penalty
    q_diag[3 * M + 2] = strong_penalty
    Q = 2 * np.diagflat(q_diag)

    R = 2 * np.diagflat(1e-2 * np.ones((nu, 1)))

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    yref = np.vstack((xrest, np.zeros((nu, 1)))).flatten()
    ocp.cost.yref = yref
    ocp.cost.yref_e = xrest.flatten()

    # set constraints
    umax = 1 * np.ones((nu,))

    ocp.constraints.constr_type = "BGH"
    ocp.constraints.lbu = -umax
    ocp.constraints.ubu = umax
    ocp.constraints.x0 = x0.reshape((nx,))
    ocp.constraints.idxbu = np.array(range(nu))

    # wall constraint
    if with_wall:
        nbx = M + 1
        Jbx = np.zeros((nbx, nx))
        for i in range(nbx):
            Jbx[i, 3 * i + 1] = 1.0

        ocp.constraints.Jbx = Jbx
        ocp.constraints.lbx = yPosWall * np.ones((nbx,))
        ocp.constraints.ubx = 1e9 * np.ones((nbx,))

        if chain_params["slacked_wall"]:
            # slacks
            ocp.constraints.Jsbx = np.eye(nbx)
            L2_pen = 1e3
            L1_pen = 1
            ocp.cost.Zl = L2_pen * np.ones((nbx,))
            ocp.cost.Zu = L2_pen * np.ones((nbx,))
            ocp.cost.zl = L1_pen * np.ones((nbx,))
            ocp.cost.zu = L1_pen * np.ones((nbx,))

    # solver options
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.qp_solver_iter_max = 1000
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.nlp_solver_max_iter = nlp_iter

    ocp.solver_options.sim_method_num_stages = integrator_opts["num_stages"]
    ocp.solver_options.sim_method_num_steps = integrator_opts["num_steps"]
    if integrator_opts["collocation_scheme"] == "radau":
        ocp.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    elif integrator_opts["collocation_scheme"] == "legendre":
        ocp.solver_options.collocation_type = "GAUSS_LEGENDRE"
    else:
        raise Exception(
            "integrator_opts['collocation_scheme'] must be radau or legendre."
        )

    ocp.solver_options.qp_solver_cond_N = N // 4
    ocp.solver_options.qp_tol = nlp_tol/10
    ocp.solver_options.tol = nlp_tol
    # ocp.solver_options.nlp_solver_tol_eq = 1e-9
    # ocp.solver_options.levenberg_marquardt = 1e-1

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file="acados_ocp_" + model.name + ".json"
    )

    return acados_ocp_solver


def run_nominal_control_open_loop(chain_params, integrator_opts=None):

    # chain parameters
    u_init = chain_params["u_init"]
    yPosWall = chain_params["yPosWall"]

    N = chain_params["N"]
    N_run = chain_params["N_run"]
    save_results = chain_params["save_results"]
    show_plots = chain_params["show_plots"]

    if integrator_opts is None:
        print("integrator_opts not given, using default!")
        integrator_opts = {
            "type": "implicit",
            "collocation_scheme": "legendre",
            "num_stages": 2,
            "num_steps": 2,
            "newton_iter": 10,
            "tol": 1e-10,
        }

    if chain_params["nlp_solver"] == "acados":
        acados_ocp_solver = export_chain_mass_ocp_solver(chain_params, integrator_opts)
        nx = acados_ocp_solver.acados_ocp.dims.nx
        nu = acados_ocp_solver.acados_ocp.dims.nu
        xrest = acados_ocp_solver.acados_ocp.constraints.lbx_0
    elif chain_params["nlp_solver"] in ["IPOPT", "IPOPT_nohess"]:
        # chain parameters
        M = chain_params["n_mass"] - 2  # number of intermediate masses
        Ts = chain_params["Ts"]
        N = chain_params["N"]
        with_wall = chain_params["with_wall"]
        yPosWall = chain_params["yPosWall"]
        L = chain_params["L"]

        # export model
        model = export_chain_mass_model(chain_params)

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        dT = Ts

        # initial state
        xPosFirstMass = np.zeros((3, 1))
        xEndRef = np.zeros((3, 1))
        xEndRef[0] = L * (M + 1) * 6

        xrest = compute_steady_state(chain_params, xPosFirstMass, xEndRef)

        x0 = xrest

        opti = Opti()

        # Decision variables for states
        X_sym = opti.variable(nx, N + 1)
        U_sym = opti.variable(nu, N)

        t0 = time.time()
        # Gap-closing shooting constraints
        if integrator_opts["implementation"] in ["casados", "acados_internal"]:
            casadi_integrator = create_casados_integrator(
                model, integrator_opts, dt=dT, use_cython=True
            )

        if integrator_opts["implementation"] in ["casados_ctypes"]:
            casadi_integrator = create_casados_integrator(
                model, integrator_opts, dt=dT, use_cython=False
            )

        elif integrator_opts["implementation"] == "casados_gnsf":
            casadi_integrator = create_casados_integrator(
                model, integrator_opts, dt=dT, use_cython=True, integrator_type="GNSF"
            )

        elif integrator_opts["implementation"] == "casados_RK4":
            casadi_integrator = create_casados_integrator(
                model, integrator_opts, dt=dT, use_cython=True, integrator_type="RK4"
            )

        elif integrator_opts["implementation"] == "collocation":
            casadi_integrator = integrator(
                "casadi_integrator",
                "collocation",
                {"x": model.x, "p": model.u, "ode": model.f_expl_expr},
                {
                    "tf": dT,
                    "collocation_scheme": integrator_opts["collocation_scheme"],
                    "number_of_finite_elements": integrator_opts["num_steps"],
                    "interpolation_order": integrator_opts["num_stages"],
                    "rootfinder_options": {"abstol": integrator_opts["tol"]},
                },
            )
        elif integrator_opts["implementation"] in ["cvodes", "idas"]:
            casadi_integrator = integrator(
                "casadi_integrator",
                integrator_opts["implementation"],
                {"x": model.x, "p": model.u, "ode": model.f_expl_expr},
                {
                    "tf": dT,
                    "abstol": integrator_opts["tol"] / integrator_opts["num_steps"]
                    # "collocation_scheme": integrator_opts["collocation_scheme"],
                    # "number_of_finite_elements": integrator_opts["num_steps"],
                    # "interpolation_order": integrator_opts["num_stages"],
                    # "rootfinder_options": {"abstol": integrator_opts["tol"]},
                },
            )
        elif integrator_opts["implementation"] == "RK4":
            casadi_integrator = integrator(
                "casadi_integrator",
                "rk",
                {"x": model.x, "p": model.u, "ode": model.f_expl_expr},
                {
                    "tf": dT,
                    "jit": False,  # error Code generation not supported for RungeKutta
                },
            )
        print(f"time to create integrator with {integrator_opts=} {time.time()-t0} s.")

        if integrator_opts["implementation"] == "direct_collocation":
            num_stages = integrator_opts["num_stages"]
            num_steps = integrator_opts["num_steps"]
            A, b, c, _ = generate_butcher_tableu(
                num_stages, integrator_opts["collocation_scheme"]
            )

            f_expl_fun = Function(
                f"f_expl_{model.name}", [model.x, model.u], [model.f_expl_expr]
            )
            K_sym = opti.variable(nx * num_stages * num_steps, N)

            T_sim_step = dT / num_steps
            for k in range(N):
                x = X_sym[:, k]
                u = U_sym[:, k]
                x_next = x
                for i_step in range(num_steps):
                    offset_k = i_step * num_stages * nx
                    # build stage values
                    xki = num_stages * [x_next]
                    for i in range(num_stages):
                        for j in range(num_stages):
                            kj = K_sym[offset_k + j * nx : offset_k + (j + 1) * nx, k]
                            xki[i] += T_sim_step * A[i, j] * kj
                    # IRK equations
                    for i in range(num_stages):
                        ki = K_sym[offset_k + i * nx : offset_k + (i + 1) * nx, k]
                        opti.subject_to(ki == f_expl_fun(xki[i], u))
                    # output equation
                    for i in range(num_stages):
                        ki = K_sym[offset_k + i * nx : offset_k + (i + 1) * nx, k]
                        x_next += T_sim_step * b[i] * ki
                opti.subject_to(X_sym[:, k + 1] == x_next)
        elif integrator_opts["implementation"] == "casados_multi":
            integrator_list = []
            for k in range(N):
                integrator_list.append(
                    create_casados_integrator(
                        model,
                        integrator_opts,
                        dt=dT,
                        use_cython=True,
                        code_reuse=(k != 0),
                    )
                )
                x = X_sym[:, k]
                u = U_sym[:, k]
                x_next = integrator_list[k](x0=x, p=u)["xf"]
                opti.subject_to(X_sym[:, k + 1] == x_next)
        else:
            for k in range(N):
                x = X_sym[:, k]
                u = U_sym[:, k]
                x_next = casadi_integrator(x0=x, p=u)["xf"]
                opti.subject_to(X_sym[:, k + 1] == x_next)

        # set cost
        Q_mat = 2 * np.diagflat(np.ones((nx, 1)))
        q_diag = np.ones((nx, 1))
        strong_penalty = M + 1
        q_diag[3 * M] = strong_penalty
        q_diag[3 * M + 1] = strong_penalty
        q_diag[3 * M + 2] = strong_penalty
        Q_mat = 2 * np.diagflat(q_diag)

        R_mat = 2 * np.diagflat(1e-2 * np.ones((nu, 1)))

        objective = 0
        for k in range(N):
            x_err = X_sym[:, k] - xrest
            u = U_sym[:, k]
            objective += dT / 2 * (x_err.T @ Q_mat @ x_err + u.T @ R_mat @ u)

        x_err = X_sym[:, k] - xrest
        objective += 1 / 2 * (x_err.T @ Q_mat @ x_err)

        opti.minimize(objective)

        # set constraints
        # Path constraints
        umax = 1 * np.ones((nu,))
        opti.subject_to(-umax <= (U_sym <= umax))

        # initial constraint
        x0_opti = opti.parameter(nx, 1)
        opti.subject_to(X_sym[:, 0] == x0_opti)
        opti.set_value(x0_opti, x0)

        if chain_params["nlp_solver"] == "IPOPT":
            opti.solver("ipopt", {"ipopt.tol": chain_params["nlp_tol"]})
        elif chain_params["nlp_solver"] == "IPOPT_nohess":
            opti.solver(
                "ipopt",
                {},
                {
                    "hessian_approximation": "limited-memory",
                    "max_iter": 6000,
                    "limited_memory_update_type": "bfgs",
                },
            )
        # opti.solver('sqpmethod',{"qpsol":"qrqp","convexify_strategy": "regularize", "print_iteration":True,"print_time":True,"print_status":True,"print_header":False,"max_iter":10000,"qpsol_options": {"print_iter":False,"print_header":False}})

        # wall constraint
        # with_wall = False
        if with_wall:
            nbx = M + 1
            y_states = [3 * i + 1 for i in range(nbx)]

            # ocp.constraints.lbx = yPosWall * np.ones((nbx,))
            # ocp.constraints.ubx = 1e9 * np.ones((nbx,))
            for k in range(N):
                opti.subject_to(X_sym[y_states, k] > yPosWall)

            # if chain_params['slacked_wall']:
            #     # slacks
            #     ocp.constraints.Jsbx = np.eye(nbx)
            #     L2_pen = 1e3
            #     L1_pen = 1
            #     ocp.cost.Zl = L2_pen * np.ones((nbx,))
            #     ocp.cost.Zu = L2_pen * np.ones((nbx,))
            #     ocp.cost.zl = L1_pen * np.ones((nbx,))
            #     ocp.cost.zu = L1_pen * np.ones((nbx,))
    else:
        raise NotImplementedError

    acados_integrator = export_chain_mass_integrator(chain_params)

    #%% get initial state by disturbing the rest position a bit
    xcurrent = xrest.reshape((nx,))
    for i in range(5):
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", u_init)

        status = acados_integrator.solve()
        if status != 0:
            raise Exception(f"acados integrator returned status {status}")

        # update state
        xcurrent = acados_integrator.get("x")

    timings = np.zeros((N_run,))

    if chain_params["nlp_solver"] == "acados":
        simX = np.ndarray((N + 1, nx))
        simU = np.ndarray((N, nu))

        # initialize
        acados_ocp_solver.set(0, "x", xcurrent)
        acados_ocp_solver.store_iterate("default_init.json", overwrite=True)

        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)
    elif chain_params["nlp_solver"] in ["IPOPT", "IPOPT_nohess"]:

        opti.set_initial(X_sym[:, 0], xcurrent)
        for k in range(N):
            opti.set_initial(X_sym[:, k + 1], xrest)
            opti.set_initial(U_sym[:, k], np.zeros((nu, 1)))

        # x0
        opti.set_value(x0_opti, xcurrent)

    # experiment loop
    for i in range(N_run):

        if chain_params["nlp_solver"] == "acados":
            acados_ocp_solver.load_iterate("default_init.json")

            # solve ocp
            status = acados_ocp_solver.solve()
            timings[i] = acados_ocp_solver.get_stats("time_tot")[0]

            # if i == 0:
            acados_ocp_solver.print_statistics()

            if status != 0:
                acados_ocp_solver.print_statistics()
                raise Exception(
                    f"acados acados_ocp_solver returned status {status} in time step {i}"
                )
            timings[i] = acados_ocp_solver.get_stats("time_tot")
            stats = dict()
            stats["iter_count"] = acados_ocp_solver.get_stats("sqp_iter")

            if i == 0:
                for i in range(N + 1):
                    simX[i, :] = acados_ocp_solver.get(i, "x")
                for i in range(N):
                    simU[i, :] = acados_ocp_solver.get(i, "u")
        else:
            # initialize
            opti.set_initial(X_sym[:, 0], xcurrent)
            for k in range(N):
                opti.set_initial(X_sym[:, k + 1], xrest)
                opti.set_initial(U_sym[:, k], np.zeros((nu, 1)))

            sol = opti.solve()
            simX = sol.value(X_sym).T
            simU = sol.value(U_sym).T
            stats = sol.stats()
            timings[i] = stats["t_wall_total"]

        if integrator_opts["implementation"] == "acados_internal":
            timings[i] = (
                casadi_integrator.time_sim
                + casadi_integrator.time_forw
                + casadi_integrator.time_adj
                + casadi_integrator.time_hess
            )
            casadi_integrator.reset_timings()

    results = {
        "simX": simX,
        "simU": simU,
        "timings": timings,
        "chain_params": chain_params,
        "integrator_opts": integrator_opts,
        "stats": stats,
    }

    #%% plot results
    if os.environ.get("ACADOS_ON_TRAVIS") is None and show_plots:
        plot_chain_control_traj(simU)
        plot_chain_position_traj(simX, yPosWall=yPosWall)
        plot_chain_velocity_traj(simX)

        # animate_chain_position(simX, xPosFirstMass, yPosWall=yPosWall)
        # animate_chain_position_3D(simX, xPosFirstMass)

        plt.show()

    if save_results:
        save_results_as_json(results, chain_params, integrator_opts)

    return


def main():
    from chain_utils import get_chain_params

    chain_params = get_chain_params()

    integrator_opts = {
        "implementation": "casados",  # direct_collocation, collocation, casados, "RK4"
        "type": "implicit",
        "collocation_scheme": "legendre",
        "num_stages": 1,
        "num_steps": 2,
        "newton_iter": 10,
        "tol": 1e-10
    }

    for n_mass in range(3, 4):
        # adjust parameters wrt experiment
        chain_params["n_mass"] = n_mass
        chain_params["show_plots"] = True
        chain_params["slacked_wall"] = False
        chain_params["save_results"] = False
        # chain_params["qp_solver"] = "PARTIAL_CONDENSING_OSQP"
        # chain_params["nlp_solver"] = "IPOPT"
        chain_params["nlp_solver"] = "acados"

        run_nominal_control_open_loop(
            chain_params, integrator_opts=integrator_opts
        )
        # run_nominal_control_open_loop(chain_params, framework='acados', integrator_opts=integrator_opts)


if __name__ == "__main__":
    main()
