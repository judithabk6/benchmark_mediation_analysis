#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
this script is the new version of a previous script to integrate
the better imports, the renaming of the estimators, and test the
refactor (PR#64)
"""
import time
import sys
import pandas as pd
import numpy as np
from numpy.random import default_rng

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

from med_bench.utils.utils import _get_regularization_parameters
from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_dml import DoubleMachineLearning
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_ipw import InversePropensityWeighting
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.estimation.mediation_tmle import TMLE

CV_FOLDS = 5

def _transform_outputs(causal_effects):
    """Transforms outputs in the old format

    Args:
        causal_effects (dict): dictionary of causal effects

    Returns:
        list: list of causal effects
    """
    total = causal_effects["total_effect"]
    direct_treated = causal_effects["direct_effect_treated"]
    direct_control = causal_effects["direct_effect_control"]
    indirect_treated = causal_effects["indirect_effect_treated"]
    indirect_control = causal_effects["indirect_effect_control"]
    return np.array(
        [total,
         direct_treated,
         direct_control,
         indirect_treated,
         indirect_control]
    ).astype(float)


    # Helper function for regularized regressor and classifier initialization
def _get_regressor_and_classifier(regularize=True, forest=False):
    if not forest:
        cs, alphas = _get_regularization_parameters(regularization=regularize)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
    else:
        clf = RandomForestClassifier(n_estimators=100,
                                     min_samples_leaf=10,
                                     max_depth=10,
                                     random_state=25)
        reg = RandomForestRegressor(n_estimators=100,
                                    min_samples_leaf=10,
                                    max_depth=10,
                                    random_state=25)
    return clf, reg

def _get_estimation_results(x, t, m, y, estimator_name, binary_mediator):
    """Dynamically selects and calls an estimator to estimate total,
    direct, and indirect effects."""

    effects = [np.nan*5]  # Initialize variable to store the effects

    if estimator_name == "coefficient_product":
        estimator = CoefficientProduct(regularize=False)
    else:
        # get classfier and regressor according to parameters
        regularize = True
        forest = False
        cf_n_splits = 0
        if "noreg" in estimator_name:
            regularize = False
        if "forest" in estimator_name:
            forest=True
        clf, reg = _get_regressor_and_classifier(
            regularize=regularize, forest=forest)

        # select estimator according to the name
        if 'ipw' in estimator_name:
            estimator = InversePropensityWeighting(
                clip=1e-6, trim=0, classifier=clf, prop_ratio="treatment")
        elif 'g_computation_exp' in estimator_name:
            estimator = GComputation(
                regressor=reg, classifier=clf, integration="explicit")
            if not binary_mediator:
                return effects
        elif 'g_computation_imp' in estimator_name:
            estimator = GComputation(
                regressor=reg, classifier=clf, integration="implicit")
        elif 'multiply_robust' in estimator_name:
            estimator = MultiplyRobust(
                clip=1e-6, trim=0, 
                prop_ratio="mediator",
                normalized=True,
                regressor=reg,
                classifier=clf,
                integration="explicit",
            )
            if not binary_mediator:
                return effects
        elif 'DML' in estimator_name:
            estimator = MultiplyRobust(
              clip=1e-6, trim=0, 
                normalized=True,
                regressor=reg,
                classifier=clf,
                prop_ratio="treatment",
                integration="implicit",
                )
        elif 'TMLE' in estimator_name:
            estimator = TMLE(
              clip=1e-6, trim=0, 
                regressor=reg, classifier=clf, prop_ratio="treatment")
        else:
            raise ValueError('invalid estimator name')

    if "cf" in estimator_name:
        cf_n_splits = 2
        causal_effects = estimator.cross_fit_estimate(
            t, m, x, y, n_splits=cf_n_splits)
    else:
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
    effects = _transform_outputs(causal_effects)
    return effects   

estimator_list = ['coefficient_product',
                  'mediation_ipw_noreg',
                  'mediation_ipw_reg',
                  'mediation_ipw_forest',
                  'mediation_ipw_reg_cf',
                  'mediation_ipw_forest_cf',
                  'mediation_g_computation_exp_noreg',
                  'mediation_g_computation_exp_reg',
                  'mediation_g_computation_exp_forest',
                  'mediation_g_computation_exp_reg_cf',
                  'mediation_g_computation_exp_forest_cf',
                  'mediation_g_computation_imp_noreg',
                  'mediation_g_computation_imp_reg',
                  'mediation_g_computation_imp_forest',
                  'mediation_g_computation_imp_reg_cf',
                  'mediation_g_computation_imp_forest_cf',
                  'mediation_multiply_robust_noreg',
                  'mediation_multiply_robust_reg',
                  'mediation_multiply_robust_forest',
                  'mediation_multiply_robust_reg_cf',
                  'mediation_multiply_robust_forest_cf',
                  "mediation_DML_noreg",
                  "mediation_DML_reg",
                  "mediation_DML_reg_cf",
                  'mediation_DML_forest',
                  'mediation_DML_forest_cf']


if __name__ == '__main__':
    folderpath = sys.argv[1]
    # folderpath = "/data/parietal/store2/work/jabecass/results/simulations/20250215_simulations/rep0_n500_setting0_misspecMTrue_misspecYTrue"
    # folderpath = "/data/parietal/store2/work/jabecass/results/simulations/20250215_simulations/rep0_n1000_setting8_misspecMFalse_misspecYFalse"
    # folderpath = "/data/parietal/store2/work/jabecass/results/simulations/20250215_simulations/rep0_n10000_setting0_misspecMTrue_misspecYFalse"

    data_df = pd.read_csv('{}/data.csv'.format(folderpath))
    y = data_df.y.values
    t = data_df['t'].values
    x_cols = [c for c in data_df.columns if "x" in c]
    m_cols = [c for c in data_df.columns if "m" in c]
    x = data_df[x_cols].values
    m = data_df[m_cols].values

    param_df = pd.read_csv('{}/param.csv'.format(folderpath))
    out_cols = param_df.columns
    base_list = list(param_df.values[0])

    if len(np.unique(m)) == 2:
        binary_mediator = True
    else:
        binary_mediator = False

    res_list = list()
    for estimator in estimator_list:
        print(estimator)
        val_list = list(base_list)
        val_list.append(estimator)
        start = time.time()
        try:
            effects = _get_estimation_results(
                x, t.ravel(), m, y.ravel(), estimator, binary_mediator)
        except ValueError:
            effects = [np.nan*5]
        duration = time.time() - start
        val_list += list(effects)
        val_list.append(duration)
        res_list.append(val_list)

    final_cols = list(out_cols) + \
        ['estimator', 'total_effect', 'direct_treated_effect',
         'direct_control_effect', 'indirect_treated_effect',
         'indirect_control_effect', 'duration']

    res_df = pd.DataFrame(res_list, columns=final_cols)
    res_df.to_csv('{}/estimation_results.csv'.format(folderpath),
                  index=False, sep='\t')

    boot = 100
    final_cols = list(out_cols) + \
        ['estimator', 'total_effect', 'direct_treated_effect',
         'direct_control_effect', 'indirect_treated_effect',
         'indirect_control_effect', 'duration', 'boot_idx']
    res_list = list()
    for b_idx in range(boot):
        rg = default_rng((b_idx+1)*67)
        ind = rg.choice(len(y), len(y), replace=True)
        y_b, t_b, m_b, x_b = y[ind], t[ind], m[ind, :], x[ind, :]

        
        for estimator in estimator_list:
            print(estimator)
            val_list = list(base_list)
            val_list.append(estimator)
            start = time.time()
            try:
                effects = _get_estimation_results(
                    x_b, t_b.ravel(), m_b, y_b.ravel(), estimator, binary_mediator)
            except ValueError:
                effects = [np.nan*5]
            duration = time.time() - start
            val_list += list(effects)
            val_list.append(duration)
            val_list.append(b_idx)
            res_list.append(val_list)


    res_df = pd.DataFrame(res_list, columns=final_cols)
    res_df.to_csv('{}/estimation_results_bootstrap.csv'.format(folderpath),
                  index=False, sep='\t')

    res_df = pd.read_csv('{}/estimation_results.csv'.format(folderpath),
                         sep='\t')
    boostrap_df = pd.read_csv(
        '{}/estimation_results_bootstrap.csv'.format(folderpath), sep='\t')

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    effect_cols = ['total_effect',
                   'direct_treated_effect',
                   'direct_control_effect',
                   'indirect_treated_effect',
                   'indirect_control_effect']
    true_effects = ['total',
                    'direct_1',
                    'direct_0',
                    'indirect_1',
                    'indirect_0']
    agg_dict = {"{}_{}".format(c, p): pd.NamedAgg(column=c, aggfunc=percentile(p)) for c in effect_cols for p in (2.5, 97.5)}

    bootstrap_intervals = boostrap_df.groupby('estimator').agg(**agg_dict)
    complete_df = pd.merge(res_df, bootstrap_intervals, left_on="estimator", right_index=True)
    for i in range(len(effect_cols)):
        effect = effect_cols[i]
        true_effect = true_effects[i]
        complete_df = complete_df.assign(
            **{'{}_contains_true'.format(effect): ((complete_df['{}_2.5'.format(effect)]<=complete_df[true_effect])&
                                                   (complete_df['{}_97.5'.format(effect)]>=complete_df[true_effect])), 
               '{}_interval_width'.format(effect): complete_df['{}_97.5'.format(effect)]-complete_df['{}_2.5'.format(effect)]})
    complete_df.to_csv('{}/estimation_results_complete.csv'.format(folderpath),
                  index=False, sep='\t')

