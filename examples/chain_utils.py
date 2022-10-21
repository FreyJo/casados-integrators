#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;

import json, os
import numpy as np
import casadi as ca
from chain_mass_model import export_chain_mass_model, export_disturbed_chain_mass_model
from acados_template import AcadosSim, AcadosSimSolver, np_array_to_list


def get_chain_params():
    params = dict()

    params["n_mass"] = 5
    params["Ts"] = 0.2
    params["Tsim"] = 5
    params["N"] = 40
    params["u_init"] = np.array([-1, 1, 1])
    params["with_wall"] = True
    params["slacked_wall"] = False
    params["yPosWall"] = -0.05  # Dimitris: - 0.1;
    params["m"] = 0.033  # mass of the balls
    params["D"] = 1.0  # spring constant
    params["L"] = 0.033  # rest length of spring
    params["perturb_scale"] = 1e-2
    params["perturb_scale"] = 5e-3

    params["save_results"] = True
    params["show_plots"] = False
    params["nlp_iter"] = 50
    params["seed"] = 50
    params["nlp_tol"] = 1e-6
    params["qp_solver"] = "PARTIAL_CONDENSING_HPIPM"
    params["N_run"] = 1
    params["nlp_solver"] = "IPOPT"

    return params


def compute_steady_state(chain_params, xPosFirstMass, xEndRef):

    n_mass = chain_params["n_mass"]
    model = export_chain_mass_model(chain_params)
    nx = model.x.shape[0]
    M = int((nx / 3 - 1) / 2)

    # initial guess for state
    pos0_x = np.linspace(xPosFirstMass[0], xEndRef[0], n_mass)
    x0 = np.zeros((nx, 1))
    x0[: 3 * (M + 1) : 3] = pos0_x[1:].reshape((M + 1, 1))

    # decision variables
    w = [model.x, model.xdot, model.u]
    # initial guess
    w0 = ca.vertcat(*[x0, np.zeros(model.xdot.shape), np.zeros(model.u.shape)])

    # constraints
    g = []
    g += [model.f_impl_expr]  # steady state
    g += [model.x[3 * M : 3 * (M + 1)] - xEndRef]  # fix position of last mass
    g += [model.u]  # don't actuate controlled mass

    # misuse IPOPT as nonlinear equation solver
    nlp = {"x": ca.vertcat(*w), "f": 0, "g": ca.vertcat(*g)}

    solver = ca.nlpsol(
        "solver", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0}
    )
    sol = solver(x0=w0, lbg=0, ubg=0)

    wrest = sol["x"].full()
    xrest = wrest[:nx]

    return xrest


def sampleFromEllipsoid(w, Z):
    """
    draws uniform sample from ellipsoid with center w and variability matrix Z
    """

    n = w.shape[0]  # dimension
    lam, v = np.linalg.eig(Z)

    # sample in hypersphere
    r = np.random.rand() ** (1 / n)  # radial position of sample
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    x *= r
    # project to ellipsoid
    y = v @ (np.sqrt(lam) * x) + w

    return y


def get_results_filename_from_params(chain_params, integrator_opts, id=""):
    # filename = f'{id}nm_{chain_params["n_mass"]}_N_{chain_params["N"]}'
    filename = f'{id}{chain_params["nlp_solver"]}_nm_{chain_params["n_mass"]}_N_{chain_params["N"]}'
    filename += f'_{integrator_opts["implementation"]}_{integrator_opts["type"]}_{integrator_opts["collocation_scheme"]}'
    filename += f'_stages_{integrator_opts["num_stages"]}_steps_{integrator_opts["num_steps"]}_iter_{integrator_opts["newton_iter"]}'
    filename += f'_tol_{integrator_opts["tol"]}'
    filename += (
        f'_wall_{chain_params["with_wall"]}_slacked_{chain_params["slacked_wall"]}.json'
    )
    json_file = os.path.join("results", filename)
    return json_file


def save_results_as_json(
    results: dict, chain_params: dict, integrator_opts: dict, id=""
):
    results["chain_params"] = chain_params

    if not os.path.exists("results"):
        os.mkdir("results")

    json_file = get_results_filename_from_params(chain_params, integrator_opts, id=id)

    with open(json_file, "w") as f:
        json.dump(results, f, default=np_array_to_list, indent=4, sort_keys=True)

    return


def load_results_from_json(chain_params: dict, integrator_opts: dict, id=""):
    json_file = get_results_filename_from_params(chain_params, integrator_opts, id=id)

    with open(json_file, "r") as f:
        results = json.load(f)

    return results


def export_chain_mass_integrator(chain_params):

    # create ocp object to formulate the simulation problem
    sim = AcadosSim()

    # export model
    M = chain_params["n_mass"] - 2  # number of intermediate masses
    model = export_disturbed_chain_mass_model(chain_params)
    sim.model = model

    # disturbances
    nparam = 3 * M
    sim.parameter_values = np.zeros((nparam,))

    # solver options
    sim.solver_options.integrator_type = "IRK"

    sim.solver_options.sim_method_num_stages = 2
    sim.solver_options.sim_method_num_steps = 2

    # set prediction horizon
    sim.solver_options.Tsim = chain_params["Ts"]

    # acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
    acados_integrator = AcadosSimSolver(
        sim, json_file="acados_sim_" + model.name + ".json"
    )

    return acados_integrator
