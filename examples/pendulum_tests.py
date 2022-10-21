import numpy as np

from pendulum_example import run_forward_sim, evaluate_jacobian, evaluate_adjoint, solve_ocp_nlp


def test_forward_sim(integrator_opts):

    results_acados = run_forward_sim(integrator_opts, use_acados=True)
    results_casadi = run_forward_sim(integrator_opts, use_acados=False, use_cython=True)

    Nsim = len(results_casadi["U"])

    for key in results_acados:
        a, b = results_acados[key], results_casadi[key]
        err = np.max(np.abs(a - b))
        tol = integrator_opts["tol"] * 10
        if err > tol:
            raise Exception(
                f"results {key} have error {err}. should be less than {tol:.1e}."
            )
        else:
            print(f"test_forward_sim: error in {key} is {err}.")
    print(f"acados and native casadi integrators match wrt tol {tol:.1e}.")


def test_jacobian(integrator_opts):

    results_casadi = evaluate_jacobian(integrator_opts, use_acados=False)
    results_acados = evaluate_jacobian(integrator_opts, use_acados=True, use_cython=True)

    for key in results_acados:
        a, b = results_acados[key], results_casadi[key]
        err = np.max(np.abs(a - b))
        tol = integrator_opts["tol"]
        if err > tol:
            print(f"results_acados[{key}], results_casadi[{key}]: {results_acados[key], results_casadi[key]}")
            raise Exception(
                f"results {key} have error {err}. should be less than {tol:.1e}."
            )

    print(f"acados and native casadi integrators Jacobians match wrt tol {tol:.1e}.")
    print(f"test_jacobian: SUCCESS!\n")


def test_adjoint(integrator_opts):

    results_casadi = evaluate_adjoint(integrator_opts, use_acados=False)
    results_acados = evaluate_adjoint(integrator_opts, use_acados=True)

    for key in results_acados:
        a, b = results_acados[key], results_casadi[key]
        err = np.max(np.abs(a - b))
        tol = integrator_opts["tol"]
        if err > tol:
            print(f"results_acados[{key}], results_casadi[{key}]: {results_acados[key], results_casadi[key]}")
            raise Exception(
                f"results {key} have error {err}. should be less than {tol:.1e}."
            )

    print(f"acados and native casadi integrators Jacobians match wrt tol {tol:.1e}.")
    print(f"test_adjoint: SUCCESS!\n")


def test_ocp_nlp(integrator_opts):

    results_acados = solve_ocp_nlp(integrator_opts, use_acados_integrator=True, use_cython=True)
    # results_acados = solve_ocp_nlp(integrator_opts, use_acados_integrator=True, use_cython=True)
    results_casadi = solve_ocp_nlp(integrator_opts, use_acados_integrator=False)
    # results_acados = solve_ocp_nlp(integrator_opts, use_acados_integrator=True)

    N_horizon = len(results_casadi["U"])
    for key in ['X', 'U']:
        a, b = results_acados[key], results_casadi[key]
        err = np.max(np.abs(a - b))
        tol = integrator_opts["tol"] * N_horizon
        if err > tol:
            raise Exception(
                f"results {key} have error {err}. should be less than {tol:.1e}."
            )
        else:
            print(f"test_ocp_nlp: error in {key} is {err}.")

    print(f"NLP solutions match wrt tol {tol:.1e}.")
    print(f"test_ocp_nlp: SUCCESS!\n")


def main():
    integrator_opts = {
        "type": "implicit",
        "collocation_scheme": "radau",
        "num_stages": 4,
        "num_steps": 1,
        "newton_iter": 10,
        "tol": 1e-8,
    }
    test_forward_sim(integrator_opts)
    test_jacobian(integrator_opts)
    test_adjoint(integrator_opts)
    # NOTE: either run ocp_nlp test or the ones above!
    # Some library is not unloaded properly.
    # test_ocp_nlp(integrator_opts)


if __name__ == "__main__":
    main()


########
# S_forw, sensitivities of simulation result wrt x,u:
#  [[ 1.00000000e+00 -1.90521058e-03  1.00000000e-01 -2.80933385e-04
#    4.68083346e-03]
#  [ 0.00000000e+00  9.70136578e-01  0.00000000e+00  9.91426012e-02
#   -3.18777388e-03]
#  [ 0.00000000e+00 -3.45453498e-02  1.00000000e+00 -8.21255293e-03
#    9.38428112e-02]
#  [ 0.00000000e+00 -6.03901611e-01  0.00000000e+00  9.73811281e-01
#   -6.44232042e-02]]
# Sx, sensitivities of simulation result wrt x:
#  [[ 1.00000000e+00 -1.90521058e-03  1.00000000e-01 -2.80933385e-04]
#  [ 0.00000000e+00  9.70136578e-01  0.00000000e+00  9.91426012e-02]
#  [ 0.00000000e+00 -3.45453498e-02  1.00000000e+00 -8.21255293e-03]
#  [ 0.00000000e+00 -6.03901611e-01  0.00000000e+00  9.73811281e-01]]
# Su, sensitivities of simulation result wrt u:
#  [[ 0.00468083]
#  [-0.00318777]
#  [ 0.09384281]
#  [-0.0644232 ]]
# S_adj, adjoint sensitivities:
#  [1.         0.32978441 1.1        1.0644604  0.03091267]
# S_hess, second order sensitivities:
#  [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00]
#  [ 0.00000000e+00  9.44186650e-01  0.00000000e+00  3.76713414e-02
#    9.66817002e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00]
#  [ 0.00000000e+00  3.76713414e-02  0.00000000e+00  6.11885544e-03
#    4.66172112e-03]
#  [ 0.00000000e+00  9.66817002e-02  0.00000000e+00  4.66172112e-03
#   -1.97432764e-04]]
