import acados_simulator
import casadi as ca
import pickle
import time
import numpy as np

N_reps = 1
def get_time_casadi_fun(fun):
    return fun.stats()['t_wall_total']


def run_simulation(integrator, l_fun, x0, controls, N):

    # test simulation

    if isinstance(x0, list):
        x_sim = [x0[0]]
    x_sim = [x0]
    l_sim = [0.0]

    for rep in range(N_reps):
        timings = []
        for k in range(N-1):
            print(f"sim_test {k=}")
            # print(k)
            if isinstance(x0, list):
                x_sim.append(integrator(x0[k], controls[k]).full().squeeze())
                l_sim.append(l_sim[-1] + l_fun(x0[k], controls[k]).full().squeeze())
            else:
                x_sim.append(integrator(x_sim[k], controls[k]).full().squeeze())
                l_sim.append(l_sim[-1] + l_fun(x_sim[k], controls[k]).full().squeeze())
            timings.append(get_time_casadi_fun(integrator))
        if rep == 0:
            timings_min = timings
        else:
            timings_min = [min(timings[i], timings_min[i]) for i in range(len(timings))]
    # print(f"{timings_min=}, mean: {np.mean(timings_min)}")
    return x_sim, l_sim, timings_min

def run_jacobian_test(integrator, x0_list, controls):
    x, u = integrator.mx_in()
    integrator_jac = ca.jacobian(integrator(x, u), ca.vertcat(x, u))
    jac_fun = ca.Function('integrator_jac', [x, u], [integrator_jac], {"record_time": True})
    jac_list = []
    N = len(x0_list)
    for rep in range(N_reps):
        timings = []
        for k in range(N-1):
            print(f"jac_test {k=}")
            jac_list.append(jac_fun(x0_list[k], controls[k]).full())
            timings.append(get_time_casadi_fun(jac_fun))
        if rep == 0:
            timings_min = timings
        else:
            timings_min = [min(timings[i], timings_min[i]) for i in range(len(timings))]
    print(f"{timings_min=}, mean: {np.mean(timings_min)}")
    return jac_list, timings_min


def timing_comparison(timing_list, title=''):
    print(f"Timing comparison {title}")
    print(LABELS, "speedup")
    for label, metric in [('mean', np.mean), ('median', np.median), ('max', np.max), ('min', np.min)]:
        timing_values = [1e3*metric(t) for t in timing_list]
        timing_strings = [f'{t:.4f}' for t in timing_values]
        speedup = timing_values[1] / timing_values[0]
        print(f"{label} & {' & '.join(timing_strings)}, {speedup:.2f}")

LABELS = ['casados', 'casadi', 'idas']

def main():
    # user input settings
    system_type = 'drag_mode'
    dof = 6
    N = 40
    TOL = 1e-10

    # load user input
    file_name = '{}_{}DOF_N{}.pkl'.format(system_type, dof, N)
    with open('user_input_{}'.format(file_name),'rb') as f:
        user_input = pickle.load(f)

    x0 = user_input['wsol'][0][:-1].full().squeeze()
    controls = [user_input['wsol'][2*k+1] for k in range(N-1)]
    collocation_opts = {
            'tf': 1/N,
            'number_of_finite_elements': 1,
            'collocation_scheme':'radau',
            # 'rootfinder': 'fast_newton',
            'interpolation_order': 4,
            'rootfinder_options':
                {'line_search': False, 'abstolStep': TOL, 'max_iter': 20, 'print_iteration': False} #, 'abstol': TOL

            # 'jit': True #   #error Code generation not supported for Collocation
        }
    N_sim = 40

    # CASADI SIMULATOR
    dae = user_input['dae']
    dae['x'] = ca.SX.sym('x', dae['x'], 1)
    dae['z'] = ca.SX.sym('z', dae['z'], 1)
    dae['p'] = ca.SX.sym('p', dae['p'], 1)
    dae['alg'] = dae['alg'](dae['x'], dae['z'], dae['p'])
    dae['quad'] = dae['quad'](dae['x'], dae['z'], dae['p'])
    dae['ode'] = dae['ode'](dae['x'], dae['z'], dae['p'])


    function_opts = {"record_time": True}
    x_awe = ca.MX.sym('x', dae['x'].shape[0]-1, 1)
    p_awe = ca.MX.sym('p', dae['p'].shape[0]-6,1)

    # CASADI
    t0 = time.time()
    integrator = ca.integrator('F', 'collocation', dae, collocation_opts)
    out = integrator(x0 = ca.vertcat(x_awe,0), p = ca.vertcat(ca.DM.zeros(6,1),p_awe), z0 = user_input['z0'])
    f_casadi = ca.Function('f_casadi', [x_awe, p_awe], [out['xf'][:-1]], function_opts)
    l_casadi = ca.Function('l_casadi', [x_awe, p_awe], [out['xf'][-1]])
    print(f"time to create casadi integrator {time.time() - t0} s")
    x_sim_casadi, l_sim_casadi, timings_casadi = run_simulation(f_casadi, l_casadi, x0, controls, N_sim)
    jacs_casadi, timings_jac_casadi = run_jacobian_test(f_casadi, x_sim_casadi, controls)

    # IDAS
    t0 = time.time()
    integrator = ca.integrator(
            'F',
            'idas',
            dae,
            {"tf": 1/N, "max_num_steps": 100000, "abstol": TOL, "reltol": TOL, "sensitivity_method": "staggered",
            "newton_scheme": "direct"}
        )
    out = integrator(x0 = ca.vertcat(x_awe,0), p = ca.vertcat(ca.DM.zeros(6,1),p_awe), z0 = user_input['z0'])
    f_idas = ca.Function('f_idas', [x_awe, p_awe], [out['xf'][:-1]], function_opts)
    l_idas = ca.Function('l_idas', [x_awe, p_awe], [out['xf'][-1]])
    print(f"time to create idas integrator {time.time() - t0} s")
    x_sim_idas, l_sim_idas, timings_idas = run_simulation(f_idas, l_idas, x_sim_casadi, controls, N_sim)
    # x_sim_idas, l_sim_idas, timings_idas = run_simulation(f_idas, l_idas, x0, controls, N_sim)
    jacs_idas, timings_jac_idas = run_jacobian_test(f_idas, x_sim_casadi, controls)

    # CASADOS
    x_sim_casadi = [ca.vertcat(x_sim_casadi[kk], l_sim_casadi[kk]).full().squeeze() for kk in range(len(x_sim_casadi))]
    t0 = time.time()
    _, f_casados, l_casados = acados_simulator.create_awe_casados_integrator(user_input['dyn'], user_input['ts'], collocation_opts=collocation_opts, record_time=True, with_sensitivities=True, use_cython=True)
    print(f"time to create casados integrator {time.time() - t0} s")
    x_sim_casados, l_sim_casados, timings_casados = run_simulation(f_casados, l_casados, ca.vertcat(x0, 0.0).full().squeeze(), controls, N_sim)
    jacs_casados, timings_jac_casados = run_jacobian_test(f_casados, x_sim_casadi, controls)





    # # IDAS:
    # try:
    #     integrator = ca.integrator(
    #         'F',
    #         'idas',
    #         dae,
    #         {"max_num_steps": 10000, "reltol": 1e-3}
    #     )
    # Option list: http://casadi.sourceforge.net/api/html/dd/d1b/group__integrator.html#plugin_Integrator_idas
    #     out = integrator(x0 = ca.vertcat(x_awe,0), p = ca.vertcat(ca.DM.zeros(6,1),p_awe), z0 = user_input['z0'])
    #     f_idas = ca.Function('f_idas', [x_awe, p_awe], [out['xf'][:-1]], function_opts)
    #     l_idas = ca.Function('l_idas', [x_awe, p_awe], [out['xf'][-1]])

    #     x_sim_idas, l_sim_idas, timings_idas = run_simulation(f_idas, l_idas, x0, controls, N)

    # except Exception as err:
    #     print(f"{err=}")
    #     # IDASolve returned "IDA_TOO_MUCH_WORK". Consult IDAS documentation.')
    #     import pdb; pdb.set_trace()

    # timing evaluation:
    timing_comparison([timings_casados, timings_casadi, timings_idas], title='simulation')
    timing_comparison([timings_jac_casados, timings_jac_casadi, timings_jac_idas], title='forward sensitivities')

    jacobian_errors = [np.max(np.abs(jacs_casadi[ii] - jacs_casados[ii][:-1, :-1])) for ii in range(len(jacs_casadi))]
    # print(f"{jacobian_errors=}")
    # print(f"max jac diff {max(jacobian_errors)}")

    import matplotlib.pyplot as plt
    plt.plot([xx[0] for xx in x_sim_casadi], label = 'casadi')
    import ipdb; ipdb.set_trace()
    plt.plot([xx[0] for xx in x_sim_casados], label ='casados', linestyle='dashed')
    plt.title('Kite x-position [m]')
    plt.legend()
    plt.show()

    x_diff = [np.abs(x_sim_casadi[i] - x_sim_casados[i]) for i in range(len(x_sim_casadi))]
    max_diff_x = np.max(x_diff)
    print(f"{max_diff_x=}")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()