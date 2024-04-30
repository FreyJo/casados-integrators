import numpy as np

from casadi import *

from pendulum_model import export_pendulum_ode_model, plot_pendulum
from casados_integrator import CasadosIntegrator


from utils import *


def run_forward_sim(integrator_opts, plot_traj=False, use_acados=True, use_cython=False):
    Nsim = 200
    dt = 0.1
    u0 = np.array([0.0])
    x0 = np.array([0.0, np.pi + 1, 0.0, 0.0])
    nx = len(x0)
    nu = len(u0)

    # create integrator
    model = export_pendulum_ode_model()
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython
        )
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
    print(f"\n created test_integrator:\n{test_integrator}\n")

    # test call
    result = test_integrator(x0=x0, p=u0)["xf"]
    print(f"test_integrator test eval, result: {result}")

    # print(f"test_integrator.has_jacobian(): {test_integrator.has_jacobian()}")
    # print(f"test_integrator.jacobian(): {test_integrator.jacobian()}")

    # test_integrator_sens = test_integrator.jacobian()
    # print(f"created test_integrator_sens {test_integrator_sens}\n")

    # open loop simulation
    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim, nu))
    x_current = x0
    simX[0, :] = x_current

    for i in range(Nsim):
        simX[i + 1, :] = (
            test_integrator(x0=simX[i, :], p=u0)["xf"].full().reshape((nx,))
        )
        simU[i, :] = u0

    # # test call jacobian
    # sens = test_integrator_sens(x0, u0, x0)
    # print(f"sens {sens}")

    if plot_traj:
        plot_pendulum(dt * np.array(range(Nsim + 1)), 10, simU, simX, latexify=False)

    results = {"X": simX, "U": simU}

    print(f"test_forward_sim: SUCCESS!\n")

    return results


def evaluate_jacobian(integrator_opts, use_acados=True, use_cython=False):
    dt = 0.1
    u0 = np.array([0.0])
    x0 = np.array([0.0, np.pi + 1, 0.0, 0.0])

    xsym = MX.sym("x", 4, 1)
    usym = MX.sym("u")

    # create integrator
    model = export_pendulum_ode_model()
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython
        )
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
    print(f"\n created test_integrator:\n{test_integrator}\n")

    jac_x_expr = jacobian(test_integrator(x0=xsym, p=usym)["xf"], xsym)
    jac_x_fun = Function("casados_jacobian_x", [xsym, usym], [jac_x_expr])
    jac_x_result = jac_x_fun(x0, u0).full()
    # print(f'jac_x_result \n{np.array(jac_x_result)}')

    jac_u_expr = jacobian(test_integrator(x0=xsym, p=usym)["xf"], usym)
    jac_u_fun = Function("casados_jacobian_x", [xsym, usym], [jac_u_expr])
    jac_u_result = jac_u_fun(x0, u0).full()
    # print(f'jac_u_result \n{np.array(jac_u_result)}')

    result = {"jac_x": jac_x_result, "jac_u": jac_u_result}

    del (
        test_integrator,
        jac_u_fun,
        jac_x_fun,
        jac_u_result,
        jac_x_result,
        jac_x_expr,
        jac_u_expr,
    )
    return result


def evaluate_adjoint(integrator_opts, use_acados=True, use_cython=False):
    # NOTE: doesnt work (yet).
    print(f"evaluate_adjoint with use_acados = {use_acados} \n")
    dt = 0.1
    u0 = np.array([0.0])
    x0 = np.array([0.0, np.pi + 1, 0.0, 0.0])

    xsym = MX.sym("x", 4, 1)
    usym = MX.sym("u")
    seed_sym = MX.sym("seed_sym", 4, 1)
    adj_seed = np.ones(4)

    # create integrator
    model = export_pendulum_ode_model()
    if use_acados:
        test_integrator = create_casados_integrator(
            model, integrator_opts, dt=dt, use_cython=use_cython
        )
        integrator_out = test_integrator(xsym, usym)
    else:
        test_integrator = create_casadi_integrator(model, integrator_opts, dt=dt)
        integrator_out = test_integrator(xsym, usym, [], [], [], [])[0]


    adj_expr_xu = jtimes(
        integrator_out, vertcat(xsym, usym), seed_sym, True
    )
    adj_fun_xu = Function("casados_adj", [xsym, usym, seed_sym], [adj_expr_xu])

    print(f"\n created test_integrator adjoiont:\n{adj_fun_xu}\n")

    adj_result_xu = adj_fun_xu(x0, u0, adj_seed)
    print(f"adj_result_xu {adj_result_xu}")

    result = {"adj_result_xu": adj_result_xu}

    return result


def test_adjoint(integrator_opts):
    # test adjoint
    dt = 0.1
    u0 = np.array([0.0])
    x0 = np.array([0.0, np.pi + 1, 0.0, 0.0])
    adj_seed = np.ones(4)

    #
    # adj_fun = casados_integrator.get_reverse(1, None, None, None, None)
    # print(f"\nadj_fun {adj_fun}\n")
    # adj_out = adj_fun(x0, u0, nominal_out, adj_seed)
    # print(f"\nevaluated adj_fun got {adj_out}\n")

    xsym = MX.sym("x", 4, 1)
    usym = MX.sym("u")

    model = export_pendulum_ode_model()
    casados_integrator = create_casados_integrator(model, integrator_opts, dt=dt)

    seed_sym = MX.sym("seed_sym", 4, 1)
    # NOTE: True forces reverse AD
    adj_expr_x = jtimes(casados_integrator(xsym, usym), xsym, seed_sym, True)
    adj_fun_x = Function("casados_adj", [xsym, usym, seed_sym], [adj_expr_x])
    adj_result_x = adj_fun_x(x0, u0, adj_seed)
    print(f"adj_result_x {adj_result_x}")

    adj_expr_xu = jtimes(
        casados_integrator(xsym, usym), vertcat(xsym, usym), seed_sym, True
    )
    adj_fun_xu = Function("casados_adj", [xsym, usym, seed_sym], [adj_expr_xu])
    adj_result_xu = adj_fun_xu(x0, u0, adj_seed)
    print(f"adj_result_xu {adj_result_xu}")

    print(f"test_adjoint: SUCCESS!\n")


def test_hessian(integrator_opts):
    # test hessian
    dt = 0.1
    u0 = np.array([0.0])
    x0 = np.array([0.0, np.pi + 1, 0.0, 0.0])
    adj_seed = np.ones(4)
    nominal_out = x0

    model = export_pendulum_ode_model()
    casados_integrator = create_casados_integrator(model, integrator_opts, dt=dt)

    xsym = MX.sym("x", 4, 1)
    usym = MX.sym("u")
    seed_sym = MX.sym("seed_sym", 4, 1)
    # adj_expr = jtimes(casados_integrator(xsym, usym), xsym, seed_sym)
    # adj_fun = Function('casados_adj', [xsym, usym, seed_sym], [adj_expr])

    # hess_expr = jacobian(adj_expr, xsym)
    # hess_x = jacobian(adj_expr[0], xsym)
    # print(f"\nhess_x {hess_x}\n")

    # xu version
    adj_expr_xu = jtimes(
        casados_integrator(xsym, usym), vertcat(xsym, usym), seed_sym, True
    )
    adj_fun_xu = Function("casados_adj", [xsym, usym, seed_sym], [adj_expr_xu])
    hess_expr = jacobian(adj_expr_xu, vertcat(xsym, usym))
    print(f"\nhess_expr {hess_expr}\n")

    hess_fun = Function("hess_fun", [xsym, usym, seed_sym], [hess_expr])
    print(f"\nhess_fun {hess_fun}\n")
    hess_result = hess_fun(x0, u0, adj_seed)
    print(f"\nhess_result {np.array(hess_result)}\n")

    # hess_out = hess_fun(x0, u0, adj_seed, adj_out)
    # print(f"\nevaluated hess_fun got {hess_out}\n")
    print(f"test_hessian: SUCCESS!\n")


def solve_ocp_nlp(
    integrator_opts,
    use_acados_integrator=True,
    SINGLE_INTEGRATOR=True,
    plot_traj=False,
    use_cython=False,
):
    model = export_pendulum_ode_model()
    x = model.x
    u = model.u

    nx = max(x.shape)
    nu = max(u.shape)

    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-2])

    Tf = 1.0
    N = 20
    dT = Tf / N

    opti = Opti()

    # Decision variables for states
    X_sym = opti.variable(nx, N + 1)
    # Aliases for states
    pos = X_sym[0, :]

    # Decision variables for control vector
    U_sym = opti.variable(nu, N)  # force [N]

    # Gap-closing shooting constraints
    if use_acados_integrator:
        if SINGLE_INTEGRATOR:
            casados_integrator = create_casados_integrator(
                model, integrator_opts, dt=dT, use_cython=use_cython
            )
        else:
            integrator_list = []

        for k in range(N):
            x = X_sym[:, k]
            u = U_sym[:, k]
            if SINGLE_INTEGRATOR:
                x_next = casados_integrator(x, u)
            else:
                integrator_list.append(
                    create_casados_integrator(
                        model, integrator_opts, dt=dT, use_cython=use_cython
                    )
                )
                casados_integrator = integrator_list[-1]
                x_next = casados_integrator(x, u)

            opti.subject_to(X_sym[:, k + 1] == x_next)
    else:
        # hard coded RK4
        #
        # Continuous system dynamics as a CasADi Function
        # f = Function('f', [x, u],[model.f_expl_expr])
        # for k in range(N):
        #     x = X_sym[:,k]
        #     u = U_sym[:,k]
        #     k1 = f(x, u)
        #     k2 = f(x + dT/2 * k1, u)
        #     k3 = f(x + dT/2 * k2, u)
        #     k4 = f(x + dT * k3, u)
        #     x_next = x+dT/6*(k1 +2*k2 +2*k3 +k4)
        #     opti.subject_to(X_sym[:,k+1]==x_next)

        # https://web.casadi.org/python-api/#integrator
        casadi_integrator = integrator(
            "casadi_integrator",
            "collocation",
            {"x": x, "p": u, "ode": model.f_expl_expr},
            {
                "tf": dT,
                "collocation_scheme": integrator_opts["collocation_scheme"],
                "number_of_finite_elements": integrator_opts["num_steps"],
                "interpolation_order": integrator_opts["num_stages"],
                "rootfinder_options": {"abstol": integrator_opts["tol"]},
            },
        )
        # print(casadi_integrator)
        for k in range(N):
            x = X_sym[:, k]
            u = U_sym[:, k]
            x_next = casadi_integrator(x0=x, p=u)["xf"]
            opti.subject_to(X_sym[:, k + 1] == x_next)

    # Path constraints
    opti.subject_to(-3 <= (pos <= 3))  # Syntax -3 <= pos <= 3 not supported in Python
    umax = 80.0
    opti.subject_to(-umax <= (U_sym <= umax))

    # Initial and terminal constraints
    x0 = 1 * np.array([0.0, np.pi, 0.0, 0.0])
    opti.subject_to(X_sym[:, 0] == x0)
    # opti.subject_to(X_sym[:,-1]==vertcat(0,0,0,0))

    # Objective: regularization of controls
    objective = 0
    for k in range(N):
        x = X_sym[:, k + 1]
        u = U_sym[:, k]
        objective += x.T @ Q_mat @ x + u.T @ R_mat @ u

    opti.minimize(objective)

    # SOLVER SETTINGS
    # Setting without hessian
    # opti.solver('ipopt', {}, {'hessian_approximation': 'limited-memory', "max_iter":6000, "limited_memory_update_type": "sr1"})
    # opti.solver('sqpmethod',{"qpsol":"qrqp","convexify_strategy": "regularize", "hessian_approximation": "GN", "print_iteration":True,"print_time":True,"print_status":True,"print_header":False,"max_iter":10000,"qpsol_options": {"print_iter":False,"print_header":False}})

    # Setting with hessian
    # opti.solver('sqpmethod',{"qpsol":"qrqp","convexify_strategy": "regularize", "print_iteration":True,"print_time":True,"print_status":True,"print_header":False,"max_iter":10000,"qpsol_options": {"print_iter":False,"print_header":False}})
    opti.solver("ipopt", {"ipopt.tol": integrator_opts["tol"] * N})

    # initialize
    x_init = np.zeros(X_sym.shape)
    x_init[:, 0] = x0
    u_init = np.zeros(N)
    for k in range(N):
        x_init[:, k + 1] = x0 * (k - N) / N
        u_init[k] = 0.0
    opti.set_initial(X_sym, x_init)
    opti.set_initial(U_sym, u_init)

    # solve
    sol = opti.solve()

    X_sol = sol.value(X_sym)
    U_sol = sol.value(U_sym)

    if plot_traj:
        plot_pendulum(np.linspace(0, Tf, N + 1), umax, U_sol.T, X_sol.T, latexify=False)

    print(
        f"\nsolve_ocp_nlp: SUCESSS, with use_acados_integrator = {use_acados_integrator}!\n"
    )

    stats = sol.stats()

    # timings = dict()
    if use_acados_integrator:
        time_sim = casados_integrator.time_sim
        time_forw = casados_integrator.time_forw
        time_adj = casados_integrator.time_adj
        time_hess = casados_integrator.time_hess
        print(
            f"time spent in sim: {time_sim:.3f} s, forw: {time_forw:.3f} s, adj: {time_adj:.3f} s, hess: {time_hess:.3f} s"
        )
        print(
            f"time spent in acados overall: {(time_sim+time_forw+time_adj+time_hess):.3f} s"
        )

    results = {"X": X_sol, "U": U_sol, "stats": stats}

    return results


def main():
    integrator_opts = {
        "type": "implicit",
        "collocation_scheme": "radau",
        "num_stages": 6,
        "num_steps": 3,
        "newton_iter": 10,
        "tol": 1e-6,
    }

    test_adjoint(integrator_opts)
    test_hessian(integrator_opts)
    # results_casadi = solve_ocp_nlp(integrator_opts, use_acados_integrator=False)
    # results_acados = solve_ocp_nlp(integrator_opts, use_acados_integrator=True)


if __name__ == "__main__":
    main()
