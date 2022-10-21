import tunempc
import acados_simulator
import awebox as awe
import ampyx_ap2_settings
import numpy as np
import casadi as ca
import casadi.tools as ct
import pickle
from awebox.logger.logger import Logger as awelogger
from tunempc.logger import Logger
import logging
Logger.logger.setLevel('DEBUG')
awelogger.logger.setLevel('DEBUG')

def setup_dc_ocp():

    # make awebox options dict
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options, dof = 6)

    # trajectory should be a single drag-mode cycle
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
    options['params.tether.kappa'] = 0.1
    options['params.kappa_r'] = 1

    # wind model
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # nlp discretization
    options['nlp.collocation.u_param'] = 'zoh'
    N = 40
    options['nlp.n_k'] = N
    options['nlp.collocation.d'] = 4
    options['nlp.cost.output_quadrature'] = False

    options['solver.linear_solver'] = 'ma57'
    options['solver.mu_target'] = 0.0

    options['model.system_bounds.theta.t_f'] = [13.4564, 13.4564] # [s]
    options['user_options.trajectory.fixed_params'] = {'diam_t': 2e-3, 'l_t': 434.287}

    # initialize and generate suitable initial guess trial
    dc_ocp = awe.Trial(options, 'single_kite_drag_mode')
    dc_ocp.build()
    dc_ocp.optimize(final_homotopy_step='power')

    return dc_ocp

def setup_ms_ocp(dc_ocp):

    N = dc_ocp.options['nlp']['n_k']

    # initial guess to compute optimal parameters
    w_init = []
    for k in range(N):
        w_init.append(dc_ocp.optimization.V_opt['x',k])
        w_init.append(dc_ocp.optimization.V_opt['u', k][6:10]) # rm u_fict

    # extract model data and prepare functions for MS-OCP
    model = dc_ocp.generate_optimal_model()
    wsol = w_init
    w0 = ca.vertcat(*w_init)

    x_shape = model['dae']['x'].shape
    x = ca.MX.sym('x',*x_shape)
    x_awe = x
    x_0 = w0[:x_shape[0]]
    xh = ca.MX.sym('x', x_shape[0]-1, x_shape[1])
    xh_awe = ca.vertcat(xh, 0)

    # remove fictitious forces.
    dof = 6
    u_shape = model['dae']['p'].shape
    u_shape = (u_shape[0]-dof, u_shape[1])
    u = ca.MX.sym('u',*u_shape)
    u_awe = ct.vertcat(ca.DM.zeros(dof,1),u)
    u_0 = ct.vertcat(ca.DM.zeros(dof,1), w0[x_shape[0]:x_shape[0]+u_shape[0]])

    # remove algebraic variable
    algf  = ca.Function('algf', [model['dae']['x'], model['dae']['p'], model['dae']['z']], [model['dae']['alg']])
    A = ca.jacobian(model['dae']['alg'], model['dae']['z'])
    b = - algf(model['dae']['x'], model['dae']['p'], model['dae']['z'](0.0))
    rootfinder = ca.Function('rootfinder', [model['dae']['z'], model['dae']['x'], model['dae']['p']], [ca.solve(A,b)])

    z_0 = rootfinder(0.1, x_0, u_0)
    z = rootfinder(z_0, xh_awe, u_awe)
    constraints = ca.vertcat(
        -model['constraints'](xh_awe, u_awe,z),
        -model['var_bounds_fun'](xh_awe, u_awe,z)
    )

    # remove redundant constraints
    constraints_new = []
    for i in range(constraints.shape[0]):
        if True in ca.which_depends(constraints[i],ca.vertcat(xh,u)):
            constraints_new.append(constraints[i])
    h =  ca.Function('h',[xh,u],[ca.vertcat(*constraints_new)])

    # save time-continuous dynamics
    xdot = ca.MX.sym('xdot', x.shape[0])
    xdot_awe = xdot
    rm_indeces = []
    z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])
    indeces = [k for k in range(x_awe.shape[0]+z.shape[0]) if k not in rm_indeces]
    alg = model['dae']['alg'][indeces]
    alg_energy = ca.vertcat(alg[:-1], model['dae']['z']['xdot'][-1] - model['dae']['quad']/model['t_f'])
    alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg_energy])
    dyn = ca.Function(
        'dae',
        [xdot,x,u,z],
        [alg_fun(x_awe, u_awe, ct.vertcat(xdot_awe, z))],
        ['xdot','x','u','z'],
        ['dyn'])

    _, f, l = acados_simulator.create_awe_casados_integrator(dyn, model['t_f']/N)

    # casados-MS warmstart options
    opts = {}
    opts['solver'] = {
        'ipopt': {
            'mu_init': 1e-2,
            'warm_start_init_point':'yes',
            'max_iter': 2000,
            'acceptable_iter': 5,
            'linear_solver': 'ma57',
            'warm_start_bound_push': 1e-9,
            'warm_start_slack_bound_push': 1e-9,
            'warm_start_mult_bound_push': 1e-9,
        },
    }

    # set-up multiple shooting OCP
    ms_ocp = tunempc.Tuner(
        f = f,
        l = l,
        h = h,
        p = N,
        opts = opts
    )

    # initial guess for energy state
    w_init = ms_ocp.pocp.w(w0)
    w_init['x', 0, -1] = 0.0
    for k in range(N-1):
        w_init['x', k+1, -1] = f(w_init['x',k], w_init['u',k])[-1]
    w_init['x', 0, -1] = f(w_init['x',-1], w_init['u',-1])[-1]
    w_init = w_init.cat

    return ms_ocp, w_init

def main():

    # setup direct collocation OCP with initial guess
    dc_ocp = setup_dc_ocp()

    # solve direct collocation OCP
    dc_ocp.optimize(warmstart_file = dc_ocp)

    # setup multiple shooting OCP
    ms_ocp, ms_initial = setup_ms_ocp(dc_ocp)

    # solve multiple shooting OCP
    _ = ms_ocp.solve_ocp(w0 = ms_initial)

    # extract statistics
    dc_stats = dc_ocp.optimization.stats
    t_wall_total = dc_stats['t_wall_total']
    t_func_eval = dc_stats['t_wall_nlp_f'] + dc_stats['t_wall_nlp_g'] + \
        dc_stats['t_wall_nlp_grad'] + dc_stats['t_wall_nlp_grad_f'] + \
        dc_stats['t_wall_nlp_hess_l'] + dc_stats['t_wall_nlp_jac_g']
    t_step_comp = t_wall_total - t_func_eval

    print('===========================')
    print('Direct Collocation Timings:')
    print('t_wall_total: {}'.format(t_wall_total))
    print('t_func_eval: {}'.format(t_func_eval))
    print('t_step_comp: {}'.format(t_step_comp))
    print('===========================')

    # solve multiple shooting OCP
    ms_stats = ms_ocp.pocp.solver.stats()
    t_wall_total = ms_stats['t_wall_total']
    t_func_eval = ms_stats['t_wall_nlp_f'] + ms_stats['t_wall_nlp_g'] + \
        ms_stats['t_wall_nlp_grad'] + ms_stats['t_wall_nlp_grad_f'] + \
        ms_stats['t_wall_nlp_hess_l'] + ms_stats['t_wall_nlp_jac_g']
    t_step_comp = t_wall_total - t_func_eval

    print('===========================')
    print('Direct Multiple-Shooting casados Timings:')
    print('t_wall_total: {}'.format(t_wall_total))
    print('t_func_eval: {}'.format(t_func_eval))
    print('t_step_comp: {}'.format(t_step_comp))
    print('===========================')


if __name__ == "__main__":
    main()
