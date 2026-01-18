import numpy as np
import multiprocessing as mp
import pandas as pd
import textmyself as tm
import time
from joblib import Parallel, delayed
from itertools import zip_longest
import laser.mllmod as ml


def index_block(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    blocks = tuple(zip_longest(*args, fillvalue=fillvalue))
    index_list = []
    for block in blocks:
        index_list.append(list(x for x in block if x is not fillvalue))
    return index_list


class params_qdl(object):
    def __init__(self, index, r_1, r_2, loss_db,
                 tau_gr, tau_rg, tau_par_r, tau_par_g, tau_sp, mu_r, mu_g, g_0, gbar_0, alpha,
                 q_max, tau_prp, disp, z):
        self.index = index
        self.r_1 = r_1
        self.r_2 = r_2
        self.loss_db = loss_db
        self.tau_gr = tau_gr
        self.tau_rg = tau_rg
        self.tau_par_r = tau_par_r
        self.tau_par_g = tau_par_g
        self.tau_sp = tau_sp
        self.mu_r = mu_r
        self.mu_g = mu_g
        self.g_0 = g_0
        self.gbar_0 = gbar_0
        self.alpha = alpha
        self.q_max = q_max
        self.tau_prp = tau_prp
        self.disp = disp
        self.z = z


def init_params():
    dirpath = 'C:/Research/Projects/Quantum Dot Comb Lasers/figures/set_02/'
    text_string = 'Mode-locked QD laser simulations complete (AWM): '
 
    r_1 = 0.5
    r_2 = 1.0
    loss_db = 2.5
    tau_grp = 63.4
    tau_prp = 0.15 / tau_grp

    gamma = np.sqrt(r_1 * r_2 * 10.0**(-loss_db/10.0))
    tau_pho = -0.5/np.log(np.abs(gamma))

    tau_gr = 1.0 / tau_grp
    tau_rg = 1.5 / tau_grp

    tau_par_r = 1000.0 / tau_grp
    tau_par_g = 1000.0 / tau_grp
    tau_sp = 2000.0 / tau_grp

    mu_r = 7.8
    mu_g = 2.0

    g_0 = 30.0 * tau_pho
    alpha = 0.0
    z = []
    
    q_max = 30
    t_max = 100000
    max_step = np.inf
    
    pump_list = np.linspace(1.1, 1.5, 9)
    
    x = np.linspace(np.log10(1.0e-07), np.log10(1.0e-05), 31)
    disp_list = -(10**x)

    params_list = []
    index = 0
    for gbar_0 in pump_list:
        for disp in disp_list:
            index += 1
            params_list.append(params_qdl(index, r_1, r_2, loss_db,
                                          tau_gr, tau_rg, tau_par_r, tau_par_g, tau_sp, mu_r, mu_g, g_0, gbar_0, alpha,
                                          q_max, tau_prp, [disp], z))
    
    job_df_list = []
    r_1_df_list = []
    tau_grp_df_list = []
    tau_pho_df_list = []
    alpha_df_list = []
    gbar_0_df_list = []
    disp_df_list = []
    for params in params_list:
        job_df_list.append(params.index)
        r_1_df_list.append(params.r_1)
        tau_grp_df_list.append(tau_grp)
        tau_pho_df_list.append(tau_pho)
        alpha_df_list.append(params.alpha)
        gbar_0_df_list.append(params.gbar_0)
        disp_df_list.append(params.disp[0])

    data_frame = pd.DataFrame({'Job #': job_df_list, 'r_1': r_1_df_list, 'tau_grp': tau_grp_df_list, 'tau_pho':tau_pho_df_list, 'alpha': alpha_df_list, 'gbar_0': gbar_0_df_list, 'D_2': disp_df_list})
    data_frame.to_excel(dirpath + 'set_02.xlsx', sheet_name='Set 02', index=False)

    return params_list, t_max, max_step, dirpath, text_string


def par_integrate(params, t_max, max_step, dirpath, index):
    assert params.index == index, 'params index {} does not match job index {}'.format(params.index, index)

    params_res = ml.LaserResonatorParameters(params.r_1, params.r_2, params.loss_db)

    params_mat = ml.LaserMaterialParametersQDL(params.tau_gr, params.tau_rg, params.tau_par_r, params.tau_par_g, params.tau_sp,
                                               params.mu_r, params.mu_g, params.g_0, params.gbar_0, params.alpha)

    freq_shifts = ml.FrequencyShifts(params.q_max, params.tau_prp, params_res.tau_pho, params.disp, params_mat)
    config_res = ml.LaserConfigurationSHB(params_res)
#    amplifier = ml.ActiveLaserMediumFWM(params_mat, config_res, freq_shifts, z=params.z)
    amplifier = ml.ActiveLaserMediumAWM(params_mat, config_res, freq_shifts, z=params.z)

    model = ml.ModeLockedLaserModel(params_res, freq_shifts, amplifier)
#    model.integrate(t_max, max_step=max_step, method='BDF', show=False, dirpath=dirpath, job_index=index)
    model.integrate(t_max, show=False, dirpath=dirpath, job_index=index)


if __name__ == '__main__':
    params_list, t_max, max_step, dirpath, text_string = init_params()
    num_jobs = len(params_list)
    
    num_cores = min(num_jobs, 2 * mp.cpu_count() // 3)
    print('{} jobs requested, using {} cores.'.format(num_jobs, num_cores))

    start = time.perf_counter()

    for index_list in index_block(np.arange(num_jobs), num_cores):
        Parallel(n_jobs=num_cores)(delayed(par_integrate)(params_list[index], t_max, max_step, dirpath, index + 1) for index in index_list)

    finish = time.perf_counter()
    
    elapsed = int(round(finish - start))
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        time_str = "Elapsed time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds)
    elif minutes and not hours:
        time_str = "Elapsed time: {} minutes, {} seconds".format(minutes, seconds)
    else:
        time_str = "Elapsed time: {} seconds ({})".format(seconds)

    text_template = text_string + '{} jobs\n' + time_str + ' ({:.{prec}} sec/job)'
    text_message = text_template.format(num_jobs, float(elapsed) / num_jobs, prec=4)

    print(text_message)
    tm.textmyself(text_message)
