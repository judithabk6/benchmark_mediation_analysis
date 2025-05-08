#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pathlib import Path
from scipy.special import expit
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns

from med_bench.get_simulated_data import simulate_data




exp_path = 'results/simulations/20250215_simulations'


dim_m_list = [1, 1, 5] * 3
m_type_list = ['binary', 'continuous', 'continuous'] * 3
wt_list = [0.5, 0.5, 0.2] + [2, 0.8, 0.3] + [4, 2, 1]
wm_list = [0.5, 0.5, 0.2] + [10, 5, 5] + [2, 1, 1]
y_m_setting = [[False, False],
               [True, False],
               [False, True],
               [True, True]]
n_sizes = [500, 1000, 10000]
nb_rep = 200
dim_x = 5

cols = ['rep_id', 'n', 'dim_x', 'dim_m', 'type_m', 'wt_list', 'wm_list',
        'm_misspec', 'y_misspec', 'mediated_prop', 'total', 'direct_1',
        'direct_0', 'indirect_1', 'indirect_0']
res_list = list()

sim_idx = 1
for n in n_sizes:
    for nrep in range(nb_rep):
        for dim_setting in range(9):
            for y_m_set in y_m_setting:
                sim_idx += 1
                rg = default_rng(int(sim_idx * n/100 * (dim_setting + 1)))
                m_set, y_set = y_m_set

                foldername = '{}/rep{}_n{}_setting{}_misspecM{}_misspecY{}'.format(
                    exp_path, nrep, n, dim_setting, m_set, y_set)
                pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
                (x, t, m, y, total, theta_1, theta_0, delta_1, delta_0, p_t, th_p_t_mx) = \
                    simulate_data(n, rg, mis_spec_m=m_set, mis_spec_y=y_set, dim_x=dim_x,
                      dim_m=dim_m_list[dim_setting],
                      seed=5, type_m=m_type_list[dim_setting], sigma_y=0.5,
                      sigma_m=0.5, beta_t_factor=wt_list[dim_setting],
                      beta_m_factor=wm_list[dim_setting])
                param_list = [nrep, n, 5, dim_m_list[dim_setting], m_type_list[dim_setting], 
                              wt_list[dim_setting], wm_list[dim_setting],
                              m_set, y_set, delta_1/total, total, theta_1,
                              theta_0, delta_1, delta_0]
                if (n == 500) and (nrep==0):
                    # making some summary thing to have an overview of all settings
                    print(len(res_list))
                    res_list.append(param_list)
                    sns.distplot(th_p_t_mx[t.ravel()==0])
                    sns.distplot(th_p_t_mx[t.ravel()==1])
                    plt.savefig('{}/{}_overlap_t_m_x.pdf'.format(exp_path, len(res_list)))
                    plt.close()


                data_cols = ['x_{}'.format(i) for i in range(dim_x)] +\
                            ['t', 'y'] +\
                            ['m_{}'.format(i) for i in range(dim_m_list[dim_setting])]
                data_df = pd.DataFrame(np.hstack((x, t, y, m)),
                                       columns=data_cols)
                data_df.to_csv('{}/data.csv'.format(foldername), index=False)
                param_df = pd.DataFrame([param_list], columns=cols)
                param_df.to_csv('{}/param.csv'.format(foldername), index=False)

res_df = pd.DataFrame(res_list, columns=cols)
res_df.to_csv('{}/simulation_description.csv'.format(exp_path), index=False)
