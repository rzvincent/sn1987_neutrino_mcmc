
osc = True  # Set to True for oscillation, False for no oscillation 

if osc:
    import SignalRate_2d_osc as s2d   # with oscillation
else:
    import SignalRate_2d as s2d     # without oscillation

import os
import numpy as np

path_sample = r'./Bin/calculation/samples_HK_osc.npy' if osc else r'./Bin/calculation/samples_HK.npy'
from scipy.optimize import minimize, root_scalar
import definition as df
from Simulation_Spectrum import time_limit
import multiprocessing as mp
from functools import partial

path_ret_osc = r'./Data/m_lambda_space_2d_osc.npy' 
path_ret = r'./Data/m_lambda_space_2d.npy'

precision = 64
E_range = (5, 50)
t_range = (0.01, time_limit('max'))

E_min, E_max = E_range
t_min, t_max = t_range

range_k = [t_range, E_range,  (-1, 1)]

def SR_HK(t, E_e, c, *args):

    R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu = args

    # N_K = 1.43e32 # for K-II

    N_K = 1.26e34 # for HK

    # eff_K = 0.932/np.sqrt(1+(34/(12-7*E_e+E_e**2))**2)        # This one is not well-fitted

    eff_K = 1

    ret = N_K * s2d.cs_at_c(c, s2d.E_nu(E_e, c)) * s2d.flux_2d(t, s2d.E_nu(E_e, c), R_c, T_c, tau_c, M_a, T_a, tau_a, m_phi, lambda_nu) * s2d.E_nu_at_E_e(E_e, c) * eff_K

    if osc: # To make the sample size comparable with the non-oscillation case
        k = 1994/1784   # calculated before
        ret = k * ret

    return ret


params = [0, 0, 0, 0, 0, 0, -4, -16]    # null hypothesis parameters for the flux function

if osc:
    print("Running with oscillation effects. Make sure to import SignalRate_2d_osc as s2d.")
else:
    print("Running without oscillation effects. Make sure to import SignalRate_2d as s2d.")


t_samples = np.load(path_sample)[:, 0]
E_samples = np.load(path_sample)[:, 1]

print('sample shape', t_samples.shape, E_samples.shape)

def param_set(log_m, log_lambda):
    return [0, 0, 0, 0, 0, 0, log_m, log_lambda]

def log_likelihood(log_m, log_lambda):

    fun1_K = lambda x: SR_HK(t_samples, x, 0, *param_set(log_m, log_lambda))

    part1_K = df.gl3_integrate(SR_HK, range_k, precision, args=param_set(log_m, log_lambda))

    part2_K = df.gl_integrate(fun1_K, E_min, E_max, precision = 64)

    part3_K = 1e-4

    ret_K = -part1_K + np.sum(np.log(part3_K + part2_K))

    return ret_K

def chi_2(log_m, log_lambda):
    return -2 * log_likelihood(log_m, log_lambda)


def find_limit_for_log_m(log_m, target_chi2):
    """
    Worker function that will be sent to individual CPU cores.
    """
    best_log_lambda = -10
    upper_bracket = best_log_lambda + 7  # Adjust if you expect the limit to be higher than this.

    def root_func(l_lambda):
        return chi_2(log_m, l_lambda) - target_chi2
        
    try:
        res_root = root_scalar(root_func, bracket=[best_log_lambda, upper_bracket], method='brentq')
        print(f'best_log_lambda {best_log_lambda} log_m: {log_m:.4f} log_lambda: {res_root.root:.4f}')
        return res_root.root
        
    except ValueError:
        print(f'best_log_lambda {best_log_lambda}')
        print(f"Failed to bracket root at log_m = {log_m}. Check the bracket range.")
        return np.nan


if __name__ == '__main__':

    log_m_vals = np.concatenate((np.linspace(0, 2.3, 30), np.linspace(2.31, 3, 70)))

    chi2_min = chi_2(-4, -16) 
    target_chi2 = chi2_min + 3.84

    worker_func = partial(find_limit_for_log_m, target_chi2=target_chi2)

    ctx = mp.get_context('spawn')  # Use 'spawn' to avoid issues on Windows
    num_cores = ctx.cpu_count()
    print(f"Starting multiprocessing pool with {num_cores} cores...")
    
    with ctx.Pool(processes=num_cores) as pool:
        results = pool.map(worker_func, log_m_vals)
    
    log_lambda_95CL = np.array(results)
    print(log_lambda_95CL)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(path_ret_osc), exist_ok=True)
    if osc:
        np.save(path_ret_osc, np.array([log_m_vals, log_lambda_95CL]).T)
    else:
        np.save(path_ret, np.array([log_m_vals, log_lambda_95CL]).T)

    from matplotlib import pyplot as plt

    plt.plot(log_m_vals, log_lambda_95CL, label='95% CL Limit')
    plt.xlabel('log10(m_phi/eV)')
    plt.ylabel('log10(lambda_nu)')
    plt.xlim(0, 3)
    plt.ylim(-10, -2)
    
    plot_path = './Data/plots/HK_sm_2d_limit.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)