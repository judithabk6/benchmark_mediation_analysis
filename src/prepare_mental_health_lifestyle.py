#!/usr/bin/env python

import pandas as pd
import numpy as np
from src import ukbb_variables
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.random import default_rng


ukbb_path = '/data/UKBB'
# file prepared from the html webpage https://biobank.ctsu.ox.ac.uk/crystal/list.cgi
annotated_variable_df = pd.read_csv('utils/20220318_all_variable_description.csv', sep='\t')

interesting_categories_mental_health = ['Social support', 'Mental health',
                                        'Addictions', 'Alcohol use', 'Anxiety',
                                        'Depression',
                                        'Happiness and subjective well-being',
                                        'Unusual and psychotic experiences',
                                        'Self-harm behaviours',
                                        'Traumatic events', 'Mania',
                                        'Mental distress']
interesting_categories_diet = ['Diet']
interesting_categories_lifestyle_diet = ['Physical activity',
                                            'Electronic device use',
                                            'Sleep', 'Smoking', 'Alcohol',
                                            'Cannabis use', 'Diet']
interesting_categories_lifestyle_diet_no_smoking_nor_alcohol = ['Physical activity',
                                            'Electronic device use',
                                            'Sleep', 
                                            'Cannabis use', 'Diet']


#################################
#### mental health variables ####
#################################

mental_health_variables = annotated_variable_df[annotated_variable_df.Category.isin(interesting_categories_mental_health)]
mental_health_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=['eid'] + mental_health_variables.variable_name.to_list())
annotated_variable_df.index = annotated_variable_df.variable_name.to_list()
dict_name_mental = {n: str(n).replace(str(n).split('-')[0], str(annotated_variable_df.loc[n, 'nice_description'])) for n in mental_health_data.columns}
dict_name_mental.update({'eid': 'eid'})
mental_health_data.rename(columns=dict_name_mental, inplace=True)
mental_health_data_restr = mental_health_data[np.array(mental_health_data.columns)[mental_health_data.describe(include='all').loc['count']>100000]]

imaging_proxy_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=['eid', '25005-2.0'])

mental_health_data_restr.drop('Date_of_completing_mental_health_questionnaire-0.0', axis=1, inplace=True)
dummy_cols = [c for c in mental_health_data_restr.columns if (c != 'Neuroticism_score-0.0') and (c != 'eid')]
dummy_mental_data = pd.get_dummies(mental_health_data_restr, columns=dummy_cols)
for c in dummy_cols:
    dummy_mental_data.loc[mental_health_data_restr[c].isnull(), dummy_mental_data.columns.str.startswith("{}_".format(c))] = np.nan

cols_to_keep = np.array(dummy_mental_data.columns)[dummy_mental_data.sum(axis=0)>10000]
dummy_mental_data_rel = dummy_mental_data[cols_to_keep]
dummy_mental_data_rel.to_csv('tmp/ukbb_mental_health_binary.csv')

#########################################
#### lifestyle variables ####
#############################

lifestyle_variables = annotated_variable_df[annotated_variable_df.Category.isin(interesting_categories_lifestyle_diet_no_smoking_nor_alcohol)]
# to simplify just a little
lifestyle_variables = lifestyle_variables[(lifestyle_variables.repetition_id==0.0)&(lifestyle_variables.visit_id==0.0)]
lifestyle_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + lifestyle_variables.variable_name.to_list())
lifestyle_data_restr = lifestyle_data[np.array(lifestyle_data.columns)[lifestyle_data.describe(include='all').loc['count']>100000]]

missing_cols = lifestyle_variables[~lifestyle_variables.variable_name.isin(lifestyle_data.columns)]
present_cols = lifestyle_variables[lifestyle_variables.variable_name.isin(lifestyle_data.columns)]
present_cols_restr = lifestyle_variables[lifestyle_variables.variable_name.isin(lifestyle_data_restr.columns)]

dummy_cols = [dict_name_lifestyle[c] for c in present_cols_restr[present_cols_restr.field_type.str.contains('categorical')].variable_name]
dummy_lifestyle_data = pd.get_dummies(lifestyle_data_restr, columns=dummy_cols)
for c in dummy_cols:
    dummy_lifestyle_data.loc[lifestyle_data_restr[c].isnull(), dummy_lifestyle_data.columns.str.startswith("{}_".format(c))] = np.nan

cols_to_keep = np.array(dummy_lifestyle_data.columns)[dummy_lifestyle_data.sum(axis=0)>10000]
dummy_lifestyle_data_rel = dummy_lifestyle_data[cols_to_keep]
dummy_lifestyle_data_rel.to_csv('tmp/ukbb_lifestyle_binary.csv')


