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
from tableone import TableOne


ukbb_path = '/data/UKBB'
# file prepared from the html webpage https://biobank.ctsu.ox.ac.uk/crystal/list.cgi
annotated_variable_df = pd.read_csv('utils/20220318_all_variable_description.csv', sep='\t')

####################
### new g-factor ###
####################

test_variables = ['398-0.2', '398-2.2', '20023-0.0', '20023-2.0', '20018-0.0',
                  '20018-2.0', '20016-0.0', '20016-2.0', '4282-0.0', '4282-2.0',
                  '6348-2.0', '6350-2.0', '23324-2.0', '21004-2.0',
                  '6373-2.0'] # '20197-2.0' is missing
cog_test_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + test_variables)
baseline_fields = ['398-{}.2', '20016-{}.0', '20023-{}.0', '20018-{}.0']
baseline_variables_visit0 = [c.format(0) for c in baseline_fields]
baseline_variables_visit2 = [c.format(2) for c in baseline_fields]
all_variables_visit2 = [c for c in test_variables if "-2" in c]



cog_baseline_visit_0 = cog_test_data[["eid"] + baseline_variables_visit0].dropna()
cog_baseline_visit_2 = cog_test_data[["eid"] + baseline_variables_visit2].dropna()
cog_all_visit_2 = cog_test_data[["eid"] + all_variables_visit2].dropna()

# recode field 20018
cog_baseline_visit_2.loc[cog_baseline_visit_2['20018-2.0']==0, "20018-2.0"] = -1
cog_baseline_visit_2.loc[cog_baseline_visit_2['20018-2.0']==2, "20018-2.0"] = 0
# recode field 20023
cog_baseline_visit_2 = cog_baseline_visit_2.assign(**{'20023-2.0':-cog_baseline_visit_2["20023-2.0"]})

pca_10 = make_pipeline(StandardScaler(), PCA(n_components=1))
res_pca = pca_10.fit_transform(cog_baseline_visit_2[baseline_variables_visit2].values)
cog_baseline_visit_2 = cog_baseline_visit_2.assign(g_factor=res_pca)


#####################
### new image pca ###
#####################
image_categories = ['dMRI skeleton', 'Subcortical volumes (FIRST)', 'Regional grey matter volumes (FAST)', 'T1 structural brain MRI']
image_variables = annotated_variable_df[annotated_variable_df.Category.isin(image_categories)]
# to simplify just a little
image_variables = image_variables[(image_variables.visit_id==2.0)]
image_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + image_variables.variable_name.to_list())
image_data_thresh = image_data.dropna(thresh=10)
image_data_smri_dmri = image_data.dropna()
pca_smri_dmri = make_pipeline(StandardScaler(), PCA(n_components=20))
pca_smri_dmri.fit(image_data_smri_dmri[image_variables.variable_name.to_list()].values)

var_df = pd.DataFrame(pca_smri_dmri['pca'].explained_variance_ratio_, columns=['percentage of variance'])
var_df.index = 'd' + (var_df.index + 1).astype(str)
var_df.to_csv('results/20240421_smri_dmri_pca.csv')

smri_dmri_pca_df = pd.DataFrame(pca_smri_dmri.transform(image_data_smri_dmri[image_variables.variable_name.to_list()].values)[:, :6])
smri_dmri_pca_df.index = image_data_smri_dmri.eid
smri_dmri_pca_df.to_csv('/data/parietal/store/work/jabecass/tmp/20220623_ukbb_smri_dmri_npc6.csv')


smri_variables = image_variables[image_variables.Category.isin(['Subcortical volumes (FIRST)', 'Regional grey matter volumes (FAST)', 'T1 structural brain MRI'])]
dmri_variables = image_variables[image_variables.Category == 'dMRI skeleton']

# pca with smri only
pca_smri = make_pipeline(StandardScaler(), PCA(n_components=20))
pca_smri.fit(image_data_smri_dmri[smri_variables.variable_name.to_list()].values)


smri_pca_df = pd.DataFrame(pca_smri.transform(image_data_smri_dmri[smri_variables.variable_name.to_list()].values)[:, :4])
smri_pca_df.index = image_data_smri_dmri.eid

# pca with dmri only
pca_dmri = make_pipeline(StandardScaler(), PCA(n_components=20))
pca_dmri.fit(image_data_smri_dmri[dmri_variables.variable_name.to_list()].values)

dmri_pca_df = pd.DataFrame(pca_dmri.transform(image_data_smri_dmri[dmri_variables.variable_name.to_list()].values)[:, :6])
dmri_pca_df.index = image_data_smri_dmri.eid

smri_dmri_pca_df.columns = ['smri_dmri_pc_{}'.format(i) for i in smri_dmri_pca_df.columns]
smri_pca_df.columns = ['smri_pc_{}'.format(i) for i in smri_pca_df.columns]
dmri_pca_df.columns = ['dmri_pc_{}'.format(i) for i in dmri_pca_df.columns]
mri_pca_df = pd.concat([smri_dmri_pca_df, smri_pca_df, dmri_pca_df], axis=1)

#####################
### hypertension ####
#####################
tension_field_ids = ["20002", "20003", "6177", "6153", "2966", "6150", "4080", "4079"]
tension_variables = annotated_variable_df[annotated_variable_df['Field ID'].isin(tension_field_ids)]
tension_variables = tension_variables[(tension_variables.visit_id==2.0)]

tension_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + tension_variables.variable_name.to_list())

def get_declarative_disease_present(field_id, visit_id, disease_code_list, df,
                                    new_col_name):
    col_list = [c for c in df.columns if '{}-{}'.format(field_id, visit_id) in c]
    disease_df = df[col_list]
    exclusion_list = list()
    for dis in disease_code_list:
        exclusion_list.append(disease_df.values == dis)
    df = df.assign(**{new_col_name: np.sum(exclusion_list, axis=0).sum(axis=1).astype(bool).astype(int)})
    return df

tension_codes = [1065, 1072]
tension_data = get_declarative_disease_present(20002, 2, tension_codes, tension_data,
                                    'self_declared_hypertension_2.0')

diabetes_codes = [1220, 1222, 1223]
tension_data = get_declarative_disease_present(20002, 2, diabetes_codes, tension_data,
                                    'self_declared_diabetes_2.0')

cholesterol_codes = [1473]
tension_data = get_declarative_disease_present(20002, 2, cholesterol_codes, tension_data,
                                    'self_declared_cholesterol_2.0')

diabetic_drug_codes = [1140874646, 1140874664, 1140874674, 1140874686, 1140874706, 1140874718, 1140874744, 1140884600, 1141189090, 1141171646, 1140868902, 1141168660, 1141152590, 1140883066, 1140874650, 1140874652, 1140874658, 1140874660, 1140874666, 1140874678, 1140874680, 1140874690, 1140874712, 1140874716, 1140874724, 1140874726, 1140874728, 1140874732, 1140874736, 1140874740, 1140874746, 1141177600, 1141171652, 1140868908, 1141168668, 1141157284]
tension_data = get_declarative_disease_present(20003, 2, diabetic_drug_codes, tension_data,
                                    'self_declared_diabetes_drug_2.0')

exclusion_codes = {'Dementia or Alzheimer’s disease': 1263,
'Parkinson’s disease': 1262,
'Chronic degenerative neurological': 1258,
'Guillain-Barré syndrome': 1256,
'Multiple Sclerosis': 1261,
'Other demyelinating disease': 1397,
'Stroke or ischaemic stroke': 1081,
'Brain cancer': 1032,
'Brain haemorrhage': 1491,
'Brain/intracranial abscess': 1245,
'Cerebral aneurysm': 1425,
'Cerebral palsy': 1433,
'Encephalitis': 1246,
'Epilepsy': 1264,
'Head injury': 1266,
'Infections of the nervous system   ': 1244,
'Ischaemic stroke': 1583,
'Meningeal cancer': 1031,
'Meningioma (benign)': 1659,
'Meningitis': 1247,
'Motor Neuron Disease': 1259,
'Neurological injury/trauma': 1240,
'Spina bifida': 1524,
'Subdural haematoma': 1083,
'Subarachnoid haemorrhage': 1086,
'Transient ischaemic attack':  1082}
tension_data = get_declarative_disease_present(20002, 2, exclusion_codes.values(), tension_data,
                                    'self_declared_mental_exclude_2.0')

tension_data = tension_data.assign(hypertensive=tension_data.apply(lambda x: (
    x['self_declared_hypertension_2.0'] + 
    (x['6177-2.0'] == 2)  + 
    (x['6177-2.1'] == 2) + 
    (x['6177-2.2'] == 2) +
    (x['6150-2.0'] == 4)  + 
    (x['6150-2.1'] == 4) + 
    (x['6150-2.2'] == 4) +
    (x['6150-2.3'] == 4) +
    (x['6153-2.0'] == 2)  + 
    (x['6153-2.1'] == 2) + 
    (x['6153-2.2'] == 2) +
    (x['6153-2.3'] == 2) 
    ).astype(bool).astype(int), axis=1))


tension_data = tension_data.assign(hypercholesterol=tension_data.apply(lambda x: (
    x['self_declared_cholesterol_2.0'] + 
    (x['6177-2.0'] == 1)  + 
    (x['6177-2.1'] == 1) + 
    (x['6177-2.2'] == 1) +
    (x['6153-2.0'] == 1)  + 
    (x['6153-2.1'] == 1) + 
    (x['6153-2.2'] == 1) +
    (x['6153-2.3'] == 1) 
    ).astype(bool).astype(int), axis=1))


tension_data = tension_data.assign(diabetic=tension_data.apply(lambda x: (
    x['self_declared_diabetes_2.0'] + 
    (x['6177-2.0'] == 3)  + 
    (x['6177-2.1'] == 3) + 
    (x['6177-2.2'] == 3) +
    (x['6153-2.0'] == 3)  + 
    (x['6153-2.1'] == 3) + 
    (x['6153-2.2'] == 3) +
    (x['6153-2.3'] == 3) 
    ).astype(bool).astype(int), axis=1))





##########################
#### McManus           ###
##########################
child_stress_fields = ['20487', '20488', '20489', '20490', '20491']
adult_stress_fields = ['20521', '20522', '20523', '20524', '20525']
stress_variables = annotated_variable_df[annotated_variable_df['Field ID'].isin(child_stress_fields + adult_stress_fields)]
stress_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + stress_variables.variable_name.to_list())

stress_nona = stress_data.dropna()
pos_stress_nona = stress_nona[stress_nona.min(axis=1)>=0]

#########################
### stolicyn 2020 ####
######################
mental_health_data = mental_health_data.assign(cMDD_crit1=mental_health_data['Frequency_of_depressed_mood_in_last_2_weeks-2.0']>=3)
mental_health_data.loc[mental_health_data['Frequency_of_depressed_mood_in_last_2_weeks-2.0'].isna(), "cMDD_crit1"] = np.nan
mental_health_data = mental_health_data.assign(cMDD_crit2=mental_health_data['Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-2.0']>=3)
mental_health_data.loc[mental_health_data['Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-2.0'].isna(), "cMDD_crit2"] = np.nan
mental_health_data = mental_health_data.assign(cMDD_crit3=mental_health_data['Happiness-2.0']>=5)
mental_health_data.loc[mental_health_data['Happiness-2.0'].isna(), "cMDD_crit3"] = np.nan
mental_health_data = mental_health_data.assign(cMDD_crit4_mood=(mental_health_data['Miserableness-2.0']==1) |
                                                               (mental_health_data['Fed-up_feelings-2.0']==1) |
                                                               (mental_health_data['Frequency_of_depressed_mood_in_last_2_weeks-2.0']==2) |
                                                               (mental_health_data['Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-2.0']==2) |
                                                               (mental_health_data['Happiness-2.0']==4))
mental_health_data.loc[((mental_health_data['Happiness-2.0'].isna())|
                       (mental_health_data['Frequency_of_unenthusiasm_/_disinterest_in_last_2_weeks-2.0'].isna())|
                       (mental_health_data['Frequency_of_depressed_mood_in_last_2_weeks-2.0'].isna())|
                       (mental_health_data['Miserableness-2.0'].isna())|
                       (mental_health_data['Fed-up_feelings-2.0'].isna())), "cMDD_crit4_mood"] = np.nan

sleep_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid', '1170-2.0', '1200-2.0'])
mental_health_data = mental_health_data.assign(cMDD_crit4_sleep=(sleep_data['1170-2.0']==1) |
                                                                (sleep_data['1170-2.0']==2) |
                                                                (sleep_data['1200-2.0']==3))
mental_health_data.loc[((sleep_data['1170-2.0'].isna())|
                       (sleep_data['1200-2.0'].isna())), "cMDD_crit4_sleep"] = np.nan

mental_health_data = mental_health_data.assign(cMDD_crit4_motor=(mental_health_data['Frequency_of_tenseness_/_restlessness_in_last_2_weeks-2.0']>=2) |
                                                               (mental_health_data['Frequency_of_tiredness_/_lethargy_in_last_2_weeks-2.0']==4))

mental_health_data.loc[((mental_health_data['Frequency_of_tenseness_/_restlessness_in_last_2_weeks-2.0'].isna())|
                       (mental_health_data['Frequency_of_tiredness_/_lethargy_in_last_2_weeks-2.0'].isna())), "cMDD_crit4_motor"] = np.nan

mental_health_data = mental_health_data.assign(cMDD_crit4_interpersonal=(mental_health_data['Irritability-2.0']==1) |
                                                               (mental_health_data['Sensitivity_/_hurt_feelings-2.0']==1) |
                                                               (mental_health_data['Loneliness,_isolation-2.0']==1) |
                                                               (mental_health_data['Guilty_feelings-2.0']==1))
mental_health_data.loc[((mental_health_data['Irritability-2.0'].isna())|
                       (mental_health_data['Guilty_feelings-2.0'].isna())|
                       (mental_health_data['Loneliness,_isolation-2.0'].isna())|
                       (mental_health_data['Sensitivity_/_hurt_feelings-2.0'].isna())), "cMDD_crit4_interpersonal"] = np.nan

mental_health_data = mental_health_data.assign(cMDD_crit4_final=(mental_health_data['cMDD_crit4_mood'] +
                                                               mental_health_data['cMDD_crit4_sleep'] +
                                                               mental_health_data['cMDD_crit4_motor'] +
                                                               mental_health_data['cMDD_crit4_interpersonal']) >= 3)
mental_health_data.loc[((mental_health_data['cMDD_crit4_mood'].isna())|
                       (mental_health_data['cMDD_crit4_interpersonal'].isna())|
                       (mental_health_data['cMDD_crit4_motor'].isna())|
                       (mental_health_data['cMDD_crit4_sleep'].isna())), "cMDD_crit4_final"] = np.nan

mental_health_data = mental_health_data.assign(cMDD_final=(mental_health_data['cMDD_crit4_final'] +
                                                               mental_health_data['cMDD_crit1'] +
                                                               mental_health_data['cMDD_crit2'] +
                                                               mental_health_data['cMDD_crit3']) >= 1)
mental_health_data.loc[((mental_health_data['cMDD_crit4_final'].isna())|
                       (mental_health_data['cMDD_crit3'].isna())|
                       (mental_health_data['cMDD_crit2'].isna())|
                       (mental_health_data['cMDD_crit1'].isna())), "cMDD_final"] = np.nan


mental_health_data = mental_health_data.assign(depression_symptoms=((mental_health_data['Ever_had_prolonged_feelings_of_sadness_or_depression-0.0']==1).astype(int) +
                                                               (mental_health_data['Ever_had_prolonged_loss_of_interest_in_normal_activities-0.0']==1).astype(int) +
                                                               (mental_health_data['Did_your_sleep_change?-0.0']==1).astype(int) +
                                                               (mental_health_data['Difficulty_concentrating_during_worst_depression-0.0']==1).astype(int) +
                                                               (mental_health_data['Feelings_of_tiredness_during_worst_episode_of_depression-0.0']==1).astype(int) +
                                                               (mental_health_data['Feelings_of_worthlessness_during_worst_period_of_depression-0.0']==1).astype(int) +
                                                               (mental_health_data['Thoughts_of_death_during_worst_depression-0.0']==1).astype(int) +
                                                               (mental_health_data['Weight_change_during_worst_episode_of_depression-0.0']==1).astype(int) >= 5))
mental_health_data.loc[((mental_health_data['Ever_had_prolonged_feelings_of_sadness_or_depression-0.0'].isna())|
                       (mental_health_data['Ever_had_prolonged_loss_of_interest_in_normal_activities-0.0'].isna())), "depression_symptoms"] = np.nan

mental_health_data = mental_health_data.assign(low_mood=(mental_health_data['Fraction_of_day_affected_during_worst_episode_of_depression-0.0']>=3)&
                                                        (mental_health_data['Frequency_of_depressed_days_during_worst_episode_of_depression-0.0']>=2))

mental_health_data.loc[((mental_health_data['Fraction_of_day_affected_during_worst_episode_of_depression-0.0'].isna())|
                       (mental_health_data['Frequency_of_depressed_days_during_worst_episode_of_depression-0.0'].isna())), "low_mood"] = np.nan

mental_health_data = mental_health_data.assign(impairment=mental_health_data['Impact_on_normal_roles_during_worst_period_of_depression-0.0']>=2)
mental_health_data.loc[mental_health_data['Impact_on_normal_roles_during_worst_period_of_depression-0.0'].isna(), "impairment"] = np.nan

mental_health_data = mental_health_data.assign(life_long_depression=mental_health_data.depression_symptoms & mental_health_data.low_mood & mental_health_data.impairment)


###############################
### social support (danilo) ###
###############################
mental_health_data = mental_health_data.assign(social_support=mental_health_data['Able_to_confide-2.0']>=4)
mental_health_data.loc[mental_health_data['Able_to_confide-2.0'].isna(), "social_support"] = np.nan


############################
### c-reactive protein ####
###########################
# positive > 3 (https://www.testing.com/tests/c-reactive-protein-crp/)
metabolic_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid', '30710-0.0', '23099-2.0'])
metabolic_data = metabolic_data.assign(inflammation=metabolic_data['30710-0.0']>3)
metabolic_data.loc[metabolic_data['30710-0.0'].isna(), "inflammation"] = np.nan


##########################
#### tobacco smoking ####
#########################

monthly_alcohol_variables = ['4407-2.0', '4418-2.0', '4429-2.0', '4440-2.0', '4451-2.0', '4462-2.0']
weekly_alcohol_variables = ['1568-2.0', '1578-2.0', '1588-2.0', '1598-2.0', '1608-2.0', '5364-2.0']
# the order is hence red wine, white wine and champagne, beer and cider, spirits,
# fortified wine and other
order_alcohol_units_conversion = [1.7, 1.7, 2.4, 1, 1.2, 1.2]

smoking_alcohol_cols = ['eid', '20160-2.0', '1558-2.0', '1249-2.0', '1249-0.0', '1239-2.0', '1239-0.0', '20117-2.0']
alcool_tobacco_data = pd.read_csv(ukbb_path + '/ukb49582.csv',
    usecols=lambda x: x in smoking_alcohol_cols + monthly_alcohol_variables + weekly_alcohol_variables)
alcool_tobacco_data = alcool_tobacco_data.assign(ever_smoked=alcool_tobacco_data['1249-2.0']<=2)
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0'].isna(), "ever_smoked"] = np.nan
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']==-3, "ever_smoked"] = np.nan
# current vs never regularly at visit 0
alcool_tobacco_data = alcool_tobacco_data.assign(
    smoking_current_vs_never_regularly_0=0)
alcool_tobacco_data.loc[alcool_tobacco_data['1249-0.0']<=1, "smoking_current_vs_never_regularly_0"] = -1
alcool_tobacco_data.loc[alcool_tobacco_data['1239-0.0']==1, "smoking_current_vs_never_regularly_0"] = 1
alcool_tobacco_data.loc[alcool_tobacco_data['1239-0.0']==2, "smoking_current_vs_never_regularly_0"] = -1
alcool_tobacco_data.loc[alcool_tobacco_data['1249-0.0']>=3, "smoking_current_vs_never_regularly_0"] = 0

alcool_tobacco_data = alcool_tobacco_data.assign(
    smoking_current_vs_never_regularly_2=0)
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']<=1, "smoking_current_vs_never_regularly_2"] = -1
alcool_tobacco_data.loc[alcool_tobacco_data['1239-2.0']==1, "smoking_current_vs_never_regularly_2"] = 1
alcool_tobacco_data.loc[alcool_tobacco_data['1239-2.0']==2, "smoking_current_vs_never_regularly_2"] = -1
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']>=3, "smoking_current_vs_never_regularly_2"] = 0

alcool_tobacco_data = alcool_tobacco_data.assign(
    smoking_current_never_previous_2=np.nan)
alcool_tobacco_data.loc[alcool_tobacco_data['1239-2.0']==1, "smoking_current_never_previous_2"] = "current"
alcool_tobacco_data.loc[alcool_tobacco_data['1239-2.0']==2, "smoking_current_never_previous_2"] = "current"
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']==2, "smoking_current_never_previous_2"] = "previous"
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']==1, "smoking_current_never_previous_2"] = "previous"
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']==3, "smoking_current_never_previous_2"] = "never"
alcool_tobacco_data.loc[alcool_tobacco_data['1249-2.0']==4, "smoking_current_never_previous_2"] = "never"





##########################
#### alcohol ###
################
# https://doi.org/10.1017%2FS0033291719000667
alcool_tobacco_data = alcool_tobacco_data.assign(alcohol_drinker=alcool_tobacco_data['1558-2.0']<=3)
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0'].isna(), "alcohol_drinker"] = np.nan
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==-3, "alcohol_drinker"] = np.nan

alcool_tobacco_data = alcool_tobacco_data.assign(
    alcohol_current_never_previous_2=np.nan)
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']<=3, "alcohol_current_never_previous_2"] = "current_often"
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==-3, "alcohol_current_never_previous_2"] = np.nan
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==4, "alcohol_current_never_previous_2"] = "current_rarely"
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==5, "alcohol_current_never_previous_2"] = "current_rarely"
alcool_tobacco_data.loc[alcool_tobacco_data['20117-2.0']==0, "alcohol_current_never_previous_2"] = "never"
alcool_tobacco_data.loc[alcool_tobacco_data['20117-2.0']==1, "alcohol_current_never_previous_2"] = "previous"


alcool_tobacco_data = alcool_tobacco_data.assign(
    current_smoker=(alcool_tobacco_data.smoking_current_never_previous_2=="current").astype(int))
alcool_tobacco_data = alcool_tobacco_data.assign(
    current_never_smoker=(alcool_tobacco_data.smoking_current_never_previous_2=="current").astype(int))
alcool_tobacco_data.loc[alcool_tobacco_data['smoking_current_never_previous_2']=="previous", "current_never_smoker"] = np.nan

alcool_tobacco_data = alcool_tobacco_data.assign(
    previous_never_smoker=(alcool_tobacco_data.smoking_current_never_previous_2=="previous").astype(int))
alcool_tobacco_data.loc[alcool_tobacco_data['smoking_current_never_previous_2']=="current", "previous_never_smoker"] = np.nan

# Topiwala 2022
# https://www.sciencedirect.com/science/article/pii/S2213158222001310?via%3Dihub#s0155
alcool_tobacco_data = alcool_tobacco_data.assign(
    alcohol_weekly_units_2=alcool_tobacco_data[weekly_alcohol_variables].values.dot(np.array(order_alcohol_units_conversion)))
alcool_tobacco_data.loc[alcool_tobacco_data[weekly_alcohol_variables].values.min(axis=1)<0, "alcohol_weekly_units_2"] = np.nan

alcool_tobacco_data = alcool_tobacco_data.assign(
    alcohol_monthly_units_2=alcool_tobacco_data[monthly_alcohol_variables].values.dot(np.array(order_alcohol_units_conversion)))
alcool_tobacco_data.loc[alcool_tobacco_data[monthly_alcohol_variables].values.min(axis=1)<0, "alcohol_monthly_units_2"] = np.nan
alcool_tobacco_data = alcool_tobacco_data.assign(
    alcohol_monthly_weekly_conv_units_2=alcool_tobacco_data.alcohol_monthly_units_2/4.3)

alcool_tobacco_data = alcool_tobacco_data.assign(
    total_alcohol_weekly_units_2=alcool_tobacco_data.alcohol_weekly_units_2)
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==6, "total_alcohol_weekly_units_2"] = 0
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==5, "total_alcohol_weekly_units_2"] = alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==5, "alcohol_monthly_weekly_conv_units_2"]
alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==4, "total_alcohol_weekly_units_2"] = alcool_tobacco_data.loc[alcool_tobacco_data['1558-2.0']==4, "alcohol_monthly_weekly_conv_units_2"]

alcool_tobacco_data = alcool_tobacco_data.assign(alcohol_above_nhs_reco=(alcool_tobacco_data.total_alcohol_weekly_units_2>14).astype(int))


##############################
#### merge everything ###
#########################

other_confounders_newby = ['6138', '20116', '31', '21300', '54', '21001', '21000',
                     '25000', '25756', '25757', '25758', '25759']
other_confounders_ju = ['6142', '189', '20160', '1239', '1249', '1558']
other_confounders_bzdok = ['845', '728', '738', '699', '826', '709',
                           '1873', '1883']
confounders_visit0 = ['1647', '1677', '1687', '1697', '1707', '1767', '1777',
                      '1787']

# add mental health and lifestyle
confounder_variables = annotated_variable_df[annotated_variable_df['Field ID'].isin(other_confounders_newby + other_confounders_ju + other_confounders_bzdok)]
confounder_variables = confounder_variables[(confounder_variables.visit_id.isin([0.0, 2.0]))&(confounder_variables.repetition_id==0.0)]
confounder_variables = confounder_variables.sort_values(by=['field_id', 'visit_id'], ascending=False).drop_duplicates(subset=['Field ID'])
confounder_variables_visit0 = annotated_variable_df[(annotated_variable_df['Field ID'].isin(confounders_visit0))&(annotated_variable_df.visit_id==0.0)]
conf_variables_list = confounder_variables.variable_name.to_list() + \
                      confounder_variables_visit0.variable_name.to_list() + \
                    ['21003-2.0', '25756-2.0', '25757-2.0', '25758-2.0', '25759-2.0']
conf_variables_list.remove('845-2.0') #too much missing data, even at visit 0


for pca_mental_health_lifestyle_ncp in [2, 5]:
    pca_mh = pd.read_csv('tmp/ukbb_mental_health_binary_pca_coord_dim5_ncp{}.csv'.format(pca_mental_health_lifestyle_ncp))
    pca_ls = pd.read_csv('tmp/ukbb_lifestyle_binary_pca_coord_dim5_ncp{}.csv'.format(pca_mental_health_lifestyle_ncp))
    pca_mh.columns = ['mh_pc_{}'.format(i) for i in pca_mh.columns]
    pca_ls.columns = ['ls_pc_{}'.format(i) for i in pca_ls.columns]
    pca_ls_mh_merge = pd.concat([pca_mh[pca_mh.columns[0:pca_mental_health_lifestyle_ncp]], pca_ls[pca_ls.columns[0:pca_mental_health_lifestyle_ncp]]], axis=1)
    pca_ls_mh_merge.index = mental_health_data.eid

    conf_data = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=lambda x: x in ['eid'] + conf_variables_list)
    dummy_cols = ['6142-2.0', '6138-2.0', '54-2.0', '31-0.0', '21000-2.0', '20160-2.0',
                  '20116-2.0', '1787-0.0', '1777-0.0', '1767-0.0', '1707-0.0',
                  '1697-0.0', '1687-0.0', '1677-0.0', '1647-0.0', '1558-2.0',
                  '1249-2.0', '1239-2.0', '728-2.0', '738-2.0', '826-2.0']
    conf_data_dummy = pd.get_dummies(data=conf_data, columns=dummy_cols)
    # add filter to avoid almost empty categories!!!
    filling = conf_data_dummy[[c for c in conf_data_dummy.columns if '_' in c]].sum(axis=0)
    dum_cols = filling.index[filling>=1000]
    conf_cols = [c for c in conf_data_dummy.columns if '_' not in c] + dum_cols.to_list()

    all_data = pd.merge(conf_data_dummy[conf_cols],
                        tension_data[['eid', 'diabetic', 'hypercholesterol',
                                      'hypertensive', 'self_declared_mental_exclude_2.0']],
                        on='eid')

    all_data_pca = pd.merge(mri_pca_df, all_data, left_index=True, right_on="eid", how='left')
    all_data_mental_t = pd.merge(all_data_pca, pca_ls_mh_merge, right_index=True, left_on='eid', how='left')
    all_data_gfactor = pd.merge(all_data_mental_t, cog_baseline_visit_2[['eid', 'g_factor']], on='eid')
    all_data_dep = pd.merge(all_data_gfactor,
                            mental_health_data[['eid', 'cMDD_final', 'life_long_depression', 'social_support']],
                            on='eid')
    all_data_met = pd.merge(all_data_dep,
                            metabolic_data[['eid', 'inflammation']],
                            on='eid')
    all_data_tab = pd.merge(all_data_met,
                            alcool_tobacco_data[['eid', 
                                                 'total_alcohol_weekly_units_2',
                                                 'alcohol_above_nhs_reco',
                                                 'current_smoker',
                                                 'current_never_smoker',
                                                 'previous_never_smoker']],
                            on='eid')
    all_data_final_nona = all_data_tab.dropna(subset=all_data_tab.columns.to_list()[:-2])
    present_names = [n for n in all_data_final_nona.columns if str(n).split('-')[0] in annotated_variable_df.field_id.astype(str).to_list()]
    absent_names = [n for n in all_data_final_nona.columns if str(n).split('-')[0] not in annotated_variable_df.field_id.astype(str).to_list()]
    dict_name_all = {n: str(n).replace(str(n).split('-')[0], str(annotated_variable_df[annotated_variable_df.field_id==str(n).split('-')[0]].nice_description.unique()[0])) for n in present_names}
    dict_name_all.update({'eid': 'eid'})
    for c in absent_names:
        dict_name_all.update({c: c})
    dict_name_all.update({'25756-2.0': 'Scanner_lateral_(X)_brain_position-2.0'})
    dict_name_all.update({'25757-2.0': 'Scanner_transverse_(Y)_brain_position-2.0'})
    dict_name_all.update({'25758-2.0': 'Scanner_longitudinal_(Z)_brain_position-2.0'})
    dict_name_all.update({'25759-2.0': 'Scanner_table_position-2.0'})
    all_data_final_nona.rename(columns=dict_name_all, inplace=True)
    all_data_final_nona = all_data_final_nona.assign(obesity=(all_data_final_nona['Body_mass_index_(BMI)-2.0']>30).astype(int))
    all_data_final_nona.to_csv('tmp/20230721_fixed_ukbb_dataset_ncp{}.csv'.format(pca_mental_health_lifestyle_ncp), sep='\t')

# adding hb1ac for the study of continuous treatments
hb1ac_ukbb = pd.read_csv(ukbb_path + '/ukb49582.csv', usecols=['eid', '30750-0.0'])
path_to_file = 'tmp/20230721_fixed_ukbb_dataset_ncp2.csv'
ukbb_data = pd.read_csv(path_to_file, sep='\t')
hb1ac_ukbb.rename(columns={'30750-0.0': 'hb1ac'}, inplace=True)
hb1ac_ukbb.head()
ukbb_data_hb1ac = pd.merge(ukbb_data, hb1ac_ukbb, on='eid', how='left')
ukbb_data_hb1ac.to_csv('tmp/20230721_fixed_ukbb_dataset_ncp2_bh1ac.csv', sep='\t', index=False)


########################
#### make table one ####
########################

potential_treatments = ['Qualifications-2.0_1.0', # higher degree
                        'diabetic',
                        'hypercholesterol',
                        'hypertensive',
                        'cMDD_final', # current depression
                        'life_long_depression',
                        'social_support',
                        'inflammation',
                        'current_smoker',
                        'current_never_smoker',
                        'previous_never_smoker',
                        'alcohol_above_nhs_reco',
                        'obesity']


for treatment_name in potential_treatments:
    if treatment_name in ['current_never_smoker', 'previous_never_smoker']:
        sub_df = all_data_final_nona.dropna(subset=[treatment_name])
    else:
        sub_df = all_data_final_nona.copy()
    mediator_columns = [c for c in all_data_final_nona.columns if 'mri_pc' in c]
    outcome_columns = ['g_factor']
    treatment_columns = [treatment_name]

    confounder_columns = set(sub_df.columns).difference(potential_mediator_cols)\
                                                                  .difference(confounders_to_remove[treatment_name])\
                                                                  .difference(outcome_columns)\
                                                                  .difference(treatment_columns)\
                                                                  .difference(['current_never_smoker', 'previous_never_smoker'])
    columns = sorted(mediator_columns + outcome_columns + list(confounder_columns))
    categorical_columns = [c for c in columns if (('0_' in c) or (c in potential_treatments))]
    other_columns = [c for c in columns if c not in categorical_columns]

    df = sub_df[columns + [treatment_name]]

    re_encoding = {'Adopted_as_a_child-0.0': {'0.0': 'No',
                                              '1.0': 'Yes'},
                   'Alcohol_intake_frequency.-2.0': {'1.0': 'Daily or almost daily',
                                                     '2.0': 'Three or four times a week',
                                                     '3.0': 'Once or twice a week',
                                                     '4.0': 'One to three times a month',
                                                     '5.0': 'Special occasions only',
                                                     '6.0': 'Never'},
                   'Average_total_household_income_before_tax-2.0': {'-1.0': 'Do not know',
                                                                     '-3.0': 'Prefer not to answer',
                                                                     '1.0': 'Less than 18,000',
                                                                     '2.0': '18,000 to 30,999',
                                                                     '3.0': '31,000 to 51,999',
                                                                     '4.0': '52,000 to 100,000',
                                                                     '5.0': 'Greater than 100,000'},
                   'Breastfed_as_a_baby-0.0': {'-1.0': 'Do not know',
                                               '0.0': 'No',
                                               '1.0': 'Yes'},
                   'Comparative_body_size_at_age_10-0.0': {'1.0': 'Thinner',
                                                           '2.0': 'Plumper',
                                                           '3.0': 'About average',
                                                           '-1.0': 'Do not know'},
                   'Comparative_height_size_at_age_10-0.0': {'1.0': 'Shorter',
                                                             '2.0': 'Taller',
                                                             '3.0': 'About average',
                                                             '-1.0': 'Do not know'},
                   'Country_of_birth_(UK/elsewhere)-0.0': {'1.0': 'England',
                                                           '2.0': 'Wales',
                                                           '3.0': 'Scotland',
                                                           '4.0': 'Northern Ireland',
                                                           '5.0': 'Republic of Ireland',
                                                           '6.0': 'Elsewhere'},
                   'Current_employment_status-2.0': {'1.0': 'In paid employment or self-employed',
                                                     '2.0': 'Retired'},
                   'Current_tobacco_smoking-2.0': {'1.0': 'Yes, on most or all days',
                                                   '0.0': 'No'},
                   'Ethnic_background-2.0': {'1001.0': 'British'},
                   'Ever_smoked-2.0': {'1.0': 'Yes',
                                       '0.0': 'No'},
                   'Handedness_(chirality/laterality)-0.0': {'1.0': 'Right-handed',
                                                             '2.0': 'Left-handed',
                                                             '3.0': 'Use both right and left hands equally'},
                   'Job_involves_shift_work-2.0': {'1.0': 'Never/rarely',
                                                   '2.0': 'Sometimes',
                                                   '4.0': 'Always'},
                   'Maternal_smoking_around_birth-0.0': {'-1.0': 'Do not know',
                                                         '0.0': 'No',
                                                         '1.0': 'Yes'},
                   'Number_of_vehicles_in_household-2.0': {'1.0': 'None',
                                                           '2.0': 'One',
                                                           '3.0': 'Two',
                                                           '4.0': 'Three',
                                                           '5.0': 'Four or more'},
                   'Part_of_a_multiple_birth-0.0': {'0.0': 'No',
                                                    '1.0': 'Yes'},
                   'Past_tobacco_smoking-2.0': {'1.0': 'Smoked on most or all days',
                                                '2.0': 'Smoked occasionally',
                                                '3.0': 'Just tried once or twice',
                                                '4.0': 'I have never smoked'},
                   'Qualifications-2.0': {'1.0': 'College or University degree',
                                          '2.0': 'A levels/AS levels or equivalent',
                                          '3.0': 'O levels/GCSEs or equivalent',
                                          '4.0': 'CSEs or equivalent',
                                          '5.0': 'NVQ or HND or HNC or equivalent',
                                          '6.0': 'Other professional qualifications eg: nursing, teaching',
                                          '-7.0': 'None of the above'}, 
                   'Sex-0.0': {'1.0': 'Male',
                               '0.0': 'Female'},
                   'Smoking_status-2.0': {'0.0': 'Never',
                                          '1.0': 'Previous',
                                          '2.0': 'Current'},
                   'UK_Biobank_assessment_centre-2.0': {'11025.0': 'Cheadle',
                                                        '11026.0': 'Reading',
                                                        '11027.0': 'Newcastle'}}
    
    df2 = df.copy()
    new_columns = list(columns)
    new_categorical_columns = list(categorical_columns)
    restr_cat_cols = set([c.split('0_')[0] + '0' for c in categorical_columns])
    for colname in re_encoding:
        df2 = df2.assign(**{colname: 'other'})
        code_dict = re_encoding[colname]
        if colname not in restr_cat_cols:
            continue
        for key in code_dict:
            dummy_colname = colname + '_' + key
            df2.loc[df2[dummy_colname]==1, colname] = code_dict[key]
            df2.drop(columns=[dummy_colname], inplace=True)
            new_columns.remove(dummy_colname)
            new_categorical_columns.remove(dummy_colname)
        new_categorical_columns.append(colname)
        new_columns.append(colname)

    final_new_columns = outcome_columns + sorted(mediator_columns) + sorted(list(set(new_columns).difference(mediator_columns + outcome_columns)))
    nice_writing_dict = {c: c.replace('_', ' ').replace('-0.0', ' - visit 0').replace('-2.0', ' - visit 2').replace('2.0', 'visit 2').capitalize() for c in final_new_columns + [treatment_name]}
    nice_final_new_columns = [nice_writing_dict[c] for c in outcome_columns] +\
                             sorted([nice_writing_dict[c] for c in mediator_columns]) +\
                             sorted([nice_writing_dict[c] for c in list(set(new_columns).difference(mediator_columns + outcome_columns))])

    nice_new_categorical_columns = [nice_writing_dict[c] for c in new_categorical_columns]
    df2.rename(columns=nice_writing_dict, inplace=True)

    table3 = TableOne(df2, columns=nice_final_new_columns,
                      categorical=nice_new_categorical_columns,
                      groupby=[nice_writing_dict[treatment_name]],
                      pval = True, smd=True, htest_name=True)
    print('\n\n\n', treatment_name, '\n', table3.tabulate(tablefmt = "latex"))


# prepare param file
j_idx = 1
output_file = "20250105_ukbb_params.csv"


with open(output_file, 'w') as param_file:
    for pca_mental_health_lifestyle_ncp in [2, 5]:
        for mri_data_type in ['smri', 'dmri', 'smri_dmri']:
        #for mri_data_type in ['dmri', 'smri_dmri']:
            for treatment_name in potential_treatments:
                param_file.write('{},{},{},{}\n'.format(j_idx, pca_mental_health_lifestyle_ncp, mri_data_type, treatment_name))
                j_idx += 1