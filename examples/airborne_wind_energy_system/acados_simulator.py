import pickle
from casados_integrator import CasadosIntegrator
from acados_template import AcadosModel, AcadosSimSolver, AcadosSim
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def create_awe_casados_integrator(dae, ts, collocation_opts=None, record_time=False, with_sensitivities=True, use_cython=True):

    # dimensions
    nx = dae.size1_in(1)
    nu = dae.size1_in(2)
    nz = dae.size1_in(3)

    # create acados model
    model = AcadosModel()
    model.x = ca.MX.sym('x',nx)
    model.u = ca.MX.sym('u',nu)
    model.z = ca.MX.sym('z', nz)
    model.xdot = ca.MX.sym('xdot', nx)
    model.p = []
    model.name = 'awe_simulation_test'

    # f(xdot, x, u, z) = 0
    model.f_impl_expr = dae(model.xdot, model.x, model.u[:nu], model.z)

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = ts
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 4 # nlp.d
    sim.solver_options.num_steps = 1
    # sim.solver_options.newton_iter = 20
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    if with_sensitivities:
        sim.solver_options.sens_forw = True
        sim.solver_options.sens_algebraic = False
        sim.solver_options.sens_hess = True
        sim.solver_options.sens_adj = True

    if collocation_opts is not None:
        # sim.solver_options.T = collocation_opts['tf']
        sim.solver_options.num_steps = collocation_opts['number_of_finite_elements']
        sim.solver_options.num_stages = collocation_opts['interpolation_order']
        if collocation_opts['collocation_scheme'] == 'radau':
            sim.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        sim.solver_options.newton_tol = collocation_opts['rootfinder_options']['abstolStep']
        sim.solver_options.newton_iter = collocation_opts['rootfinder_options']['max_iter']

    function_opts = {"record_time": record_time}

    casados_integrator = CasadosIntegrator(sim, use_cython=use_cython)
    # reformat for tunempc
    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    xf = casados_integrator(x, u)
    f = ca.Function('f', [x,u], [xf], ['x0','p'], ['xf'], function_opts)
    l = ca.Function('l', [x,u], [xf[-1]], function_opts)

    return casados_integrator, f, l

def remove_energy_state_from_initial_guess(wsol, N, nx, nu):

    wnew = []
    for k in range(N):
        wnew.append(wsol[k*(nx+nu): k*(nx+nu)+nx-1])
        wnew.append(wsol[k*(nx+nu)+nx: k*(nx+nu)+nx+nu])

    wnew = ca.vertcat(*wnew)

    return wnew

if __name__ == "__main__":

    # load user input
    with open('user_input_lift_mode_v5_N40_energy.pkl','rb') as f:
        user_input = pickle.load(f)

    # casados integrator
    casados_integrator, _, _ = create_awe_casados_integrator(user_input['dyn'], user_input['ts'])

    # test casados integrator
    N = 40
    nx = 24
    nu = 4
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    solX = np.ndarray((N+1, nx))


    simX[0,:] = user_input['w0'][:nx].full().squeeze()
    for k in range(N):
        solX[k,:] = user_input['w0'][k*(nx+nu):k*(nx+nu)+nx].full().squeeze()
        simU[k,:] = user_input['w0'][k*(nx+nu)+nx:(k+1)*(nx+nu)].full().squeeze()
        simX[k+1,:] = casados_integrator(solX[k,:], simU[k,:]).full().reshape((nx,))
    solX[-1,:] = user_input['w0'][0:nx].full().squeeze()

    plt.figure()
    plt.plot(simX[:,1], simX[:,2], label = 'casados sim')
    plt.plot(solX[:,1], solX[:,2], label = 'ocp solution')
    plt.xlabel('y [m]')
    plt.ylabel('z [m]')
    plt.legend()

    plt.figure()
    plt.plot(simX[:,-1])
    plt.show()

