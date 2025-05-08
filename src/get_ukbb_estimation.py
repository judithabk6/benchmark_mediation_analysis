#!/usr/bin/env python

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
                normalized=True,
                regressor=reg,
                classifier=clf,
                prop_ratio="treatment",
                integration="implicit",
                )
        elif 'TMLE' in estimator_name:
            estimator = TMLE(
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

pca_mental_health_lifestyle_ncp, mri_data_type, treatment_name = sys.argv[1:]
pca_mental_health_lifestyle_ncp = int(pca_mental_health_lifestyle_ncp)


all_data_final_nona = pd.read_csv('/scratch/jabecass/tmp/20230721_fixed_ukbb_dataset_ncp{}.csv'.format(pca_mental_health_lifestyle_ncp), sep='\t')
potential_mediator_cols = [c for c in all_data_final_nona.columns if 'mri' in c]
unwanted_cols = [c for c in all_data_final_nona.columns if 'Unnamed' in c]
all_data_final_nona.drop(columns=unwanted_cols + ['eid'], inplace=True)



confounders_to_remove = {'Qualifications-2.0_1.0': [c for c in all_data_final_nona if 'Qualifications' in c],
                         'diabetic': ['diabetic'],
                         'hypercholesterol': ['hypercholesterol'],
                         'hypertensive': ['hypertensive'],
                         'cMDD_final': ['cMDD_final'] + [c for c in all_data_final_nona if 'mh_pc' in c],
                         'life_long_depression': ['life_long_depression', 'cMDD_final'] + [c for c in all_data_final_nona if 'mh_pc' in c],
                         'social_support': ['social_support'],
                         'inflammation': ['inflammation'],
                         'current_smoker': ['current_never_smoker', 'previous_never_smoker'] + [c for c in all_data_final_nona.columns if (("tobacco_smoking" in c) or ('Ever_smoked' in c) or ('Smoking_status' in c))],
                         'current_never_smoker': ['current_smoker', 'previous_never_smoker'] + [c for c in all_data_final_nona.columns if (("tobacco_smoking" in c) or ('Ever_smoked' in c) or ('Smoking_status' in c))],
                         'previous_never_smoker': ['current_smoker', 'current_never_smoker'] + [c for c in all_data_final_nona.columns if (("tobacco_smoking" in c) or ('Ever_smoked' in c) or ('Smoking_status' in c))],
                         'alcohol_above_nhs_reco': ['total_alcohol_weekly_units_2'] + [c for c in all_data_final_nona.columns if "Alcohol_intake_frequency" in c],
                         'obesity': ['Body_mass_index_(BMI)-2.0']
                        }
if treatment_name in ['current_never_smoker', 'previous_never_smoker']:
    sub_df = all_data_final_nona.dropna(subset=[treatment_name])
else:
    sub_df = all_data_final_nona.copy()

mediator_columns = [c for c in sub_df.columns if '{}_pc'.format(mri_data_type) in c]
outcome_columns = ['g_factor']
treatment_columns = [treatment_name]

confounder_columns = set(sub_df.columns).difference(potential_mediator_cols)\
                                                              .difference(confounders_to_remove[treatment_name])\
                                                              .difference(outcome_columns)\
                                                              .difference(treatment_columns)\
                                                              .difference(['current_never_smoker', 'previous_never_smoker'])

confounder_columns.remove('Adopted_as_a_child-0.0_1.0')

x = sub_df[list(confounder_columns)].values
y = -sub_df[outcome_columns].values
t = sub_df[treatment_columns].values
m = sub_df[mediator_columns].values

ok_idx = (~np.isnan(t)).ravel()
x, y, t, m = x[ok_idx, :], y[ok_idx, :], t[ok_idx, :], m[ok_idx, :]

res_list = list()
for estimator in estimator_list:
    for i in range(boot):
        rg = default_rng((i+1)*67)
        ind = rg.choice(len(y), len(y), replace=True)
        y_b, t_b, m_b, x_b = y[ind], t[ind], m[ind], x[ind, :]

        try:
            effects = get_estimation(x_b, t_b.ravel(), m_b, y_b.ravel(), estimator, 5)
            print(treatment_name, i, effects)
        except (RRuntimeError, ValueError) as e:
            effects = [np.nan*6]
        res_list.append([i, estimator] + list(effects))
df = pd.DataFrame(res_list)
df.to_csv('/scratch/jabecass/results/20250315-{}-{}-ncp{}-g_factor.csv'.format(treatment_name, mri_data_type, pca_mental_health_lifestyle_ncp), sep='\t', index=False)
