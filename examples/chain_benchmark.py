import numpy as np
from chain_utils import get_chain_params, load_results_from_json
from chain_example import run_nominal_control_open_loop

import matplotlib.pyplot as plt
from plot_utils import *

FIGURES_PATH = '../latex/figures/'

IMPLEMENTATIONS = [
    "casados",
    "casados_ctypes",
    "collocation",
    "cvodes",
    "idas",
    "direct_collocation",
    "acados_GN",
    "RK4",
    "casados_RK4",
]  # , "casados_multi"]




#settings for 2, 1!
# IMPLEMENTATIONS = ["acados_GN"]
NLP_SOLVER = "IPOPT"
INTEGRATOR_DEFAULT_OPTS = {
    "implementation": "cvodes",
    "type": "implicit",
    "collocation_scheme": "legendre",
    "num_stages": 2,
    "num_steps": 1,
    "newton_iter": 10,
    "tol": 1e-8,
}

# 4, 4
# IMPLEMENTATIONS = ["casados", "casados_gnsf", "cvodes", "collocation", "RK4", "casados_RK4"]
# IMPLEMENTATIONS = ["casados", "casados_gnsf"]#, "collocation"]#S "cvodes", "RK4", "casados_RK4"]
# NLP_SOLVER = "IPOPT_nohess"
IMPLEMENTATIONS = ["casados", "casados_ctypes", "collocation", "cvodes", "idas", "direct_collocation","acados_GN",] #, "casados_multi"]
# # IMPLEMENTATIONS = ["casados_multi"]
INTEGRATOR_DEFAULT_OPTS = {
    "implementation": "cvodes",
    "type": "implicit",
    "collocation_scheme": "legendre",
    "num_stages": 4,
    "num_steps": 4,
    "newton_iter": 10,
    "tol": 1e-8,
}

N_MASSES = [3, 4, 5, 6, 7]
# N_MASSES = [7]


# reference solution
def get_reference_settings():
    chain_params = get_chain_params()
    integrator_opts = INTEGRATOR_DEFAULT_OPTS.copy()
    chain_params["show_plots"] = False
    chain_params["slacked_wall"] = False
    chain_params["save_results"] = True
    integrator_opts["tol"] = 1e-10
    integrator_opts["implementation"] = "cvodes"
    integrator_opts["num_stages"] = 3
    integrator_opts["num_steps"] = 1
    chain_params["nlp_tol"] = 1e-11
    chain_params["nlp_solver"] = "IPOPT"

    return chain_params, integrator_opts


def generate_reference_solutions():
    chain_params, integrator_opts = get_reference_settings()
    for n_mass in N_MASSES:
        chain_params["n_mass"] = n_mass
        run_nominal_control_open_loop(
            chain_params, integrator_opts=integrator_opts
        )


def get_reference_solutions(n_mass):
    chain_params, integrator_opts = get_reference_settings()
    chain_params["n_mass"] = n_mass
    results = load_results_from_json(chain_params, integrator_opts)
    simX_ref = np.array(results["simX"])
    simU_ref = np.array(results["simU"])
    return simX_ref, simU_ref


def plot_reference_solutions():
    chain_params = get_chain_params()
    yPosWall = chain_params["yPosWall"]
    for n_mass in N_MASSES:
        simX, simU = get_reference_solutions(n_mass)
        plot_chain_control_traj(simU)
        plot_chain_position_traj(simX, yPosWall=yPosWall)
        plot_chain_velocity_traj(simX)

        # animate_chain_position(simX, xPosFirstMass, yPosWall=yPosWall)
        # animate_chain_position_3D(simX, xPosFirstMass)

        plt.show()


# benchmark
def run_benchmark():
    chain_params = get_chain_params()
    integrator_opts = INTEGRATOR_DEFAULT_OPTS.copy()
    chain_params["show_plots"] = False
    chain_params["slacked_wall"] = False
    chain_params["save_results"] = True

    for n_mass in N_MASSES:
        for implementation in IMPLEMENTATIONS:
            # adjust parameters wrt experiment
            chain_params["n_mass"] = n_mass
            if implementation == "acados_GN":
                chain_params["nlp_solver"] = "acados"
            else:
                chain_params["nlp_solver"] = NLP_SOLVER
            # chain_params["qp_solver"] = "PARTIAL_CONDENSING_OSQP"
            integrator_opts["implementation"] = implementation
            run_nominal_control_open_loop(
                chain_params, integrator_opts=integrator_opts
            )


def eval_benchmark():
    chain_params = get_chain_params()
    integrator_opts = INTEGRATOR_DEFAULT_OPTS.copy()
    chain_params["show_plots"] = False
    chain_params["slacked_wall"] = False

    plt.figure()
    latexify()

    MARKERS = ["<", ">", "d", "v", "^", "o", "1", "2", "3", "4"]
    for iplot, implementation in enumerate(IMPLEMENTATIONS):
        timings_implementation = np.zeros(len(N_MASSES))

        for i, n_mass in enumerate(N_MASSES):
            simX_ref, simU_ref = get_reference_solutions(n_mass)
            # adjust parameters wrt experiment
            chain_params["n_mass"] = n_mass
            integrator_opts["implementation"] = implementation
            if implementation == "acados_GN":
                chain_params["nlp_solver"] = "acados"
            else:
                chain_params["nlp_solver"] = NLP_SOLVER

            results = load_results_from_json(chain_params, integrator_opts)
            stats = results["stats"]
            # timings_implementation[i] = np.min(results["timings"])
            timings_implementation[i] = np.min(results["timings"]) / stats['iter_count']
            # timing_functions = stats['t_wall_nlp_hess_l'] + stats['t_wall_nlp_f'] + stats['t_wall_nlp_g'] + stats['t_wall_nlp_jac_g']
            timing_functions = None
            print(
                f"{implementation}, timing: {results['timings']}s timing functions {timing_functions} iter: {stats['iter_count']} "
            )
            # import pdb; pdb.set_trace()
            print(f"diffX: {np.max(np.abs(np.array(results['simX'])-simX_ref))}")
            print(f"diffU: {np.max(np.abs(np.array(results['simU'])-simU_ref))}")

        label = implementation
        if label == "casados":
            label = r"\texttt{casados} IRK cython"
        elif label == "collocation":
            label = r"\texttt{CasADi} IRK"
        elif label == "RK4":
            label = r"\texttt{CasADi} RK4"
        elif label == "casados_RK4":
            label = r"\texttt{casados} RK4"
        elif label == "casados_ctypes":
            label = r"\texttt{casados} IRK ctypes"
        elif label == "direct_collocation":
            label = "direct collocation IRK"
        elif label == "idas":
            label = r"\texttt{IDAS}"
        elif label == "cvodes":
            label = r"\texttt{CVODES}"
        elif label == "acados_GN":
            label = r"\texttt{acados} Gau\ss-Newton SQP"
        plt.plot(N_MASSES, timings_implementation, label=label, marker=MARKERS[iplot])
    plt.legend()
    plt.xlabel("number of masses")
    plt.ylabel('CPU time / NLP iteration [s]')
    # plt.ylabel("CPU time [s]")
    plt.yscale("log")
    plt.xticks(N_MASSES)
    plt.grid()
    fig_filename = f"{FIGURES_PATH}chain_benchmark_nm_{N_MASSES[0]}_{N_MASSES[-1]}_stages{INTEGRATOR_DEFAULT_OPTS['num_stages']}_steps{INTEGRATOR_DEFAULT_OPTS['num_steps']}_{NLP_SOLVER}.pdf"
    plt.savefig(fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05)
    print(f"file saved as {fig_filename=}")
    plt.show()


if __name__ == "__main__":
    # generate_reference_solutions()
    # plot_reference_solutions()
    # run_benchmark()
    eval_benchmark()
