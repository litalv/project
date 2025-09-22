import os
import re
import pandas as pd
import numpy as np

import duckdb
# Mount Google Drive
from google.colab import drive

from project.preprocessing.utils import load_file, sql_from_MIMICIII, age, ethnicity_to_ohe, generate_series_data, _word2pat, _map_drug_name_2_category, _site_from_specimen_type
from project.preprocessing.config import get_config
from project.preprocessing.sql import get_sql_queries
from project.preprocessing.name_keywords import get_keywords_cat


def _ICU_preprocess(con, icu_sql, subject_ids, config_dict):
	""" """
	icu = sql_from_MIMICIII(con, icu_sql, subject_ids)
	
	# exclude patients with less than 54 hours of hospitalization data
	MIN_TARGET_ONSET_H = config_dict['temporal_config']['EVAL_WINDOW_H'] + config_dict['temporal_config']['GAP_H']
	icu = icu[(icu["dischtime"]-icu["admittime"]) >= pd.Timedelta(hours=MIN_TARGET_ONSET_H)]
	# remove ICU readmission within the same hospital admission
	icu.drop_duplicates(inplace=True)
	
	icu['age'] = icu.apply(lambda row: age(row['admittime'], row['dob']), axis=1)
	icu['gender'] = np.where(icu['gender']=="M", 1, 0)
	icu = ethnicity_to_ohe(icu)
	
	# for each patient find the first hospital admission
	first_admission_df = icu.sort_values(by=['subject_id', 'admittime']).drop_duplicates(subset=['subject_id'], keep='first').set_index('subject_id')

	# lable data according to all 3 targets
	second_admission_df = icu.sort_values(by=['subject_id', 'admittime']).groupby('subject_id')[['subject_id','admittime']].nth(1).rename(columns={"admittime":'sec_admittime'}).set_index('subject_id')

	targets = pd.concat([first_admission_df,second_admission_df], axis=1)
	# mortality during hospitalization or up to 30 days after discharge
	targets["mortality"] = np.where((
		(targets["deathtime"].notna()) |  # died during hospitalization
		(targets["dod"].notna()) & ((targets["dod"]-targets["dischtime"]) <= pd.Timedelta(days=config_dict['targets_config']['mortality_d']))), 1, 0)
	# length of stay > 7 days
	targets["prolonged_stay"] = np.where((targets["dischtime"]-targets["admittime"]) > pd.Timedelta(days=config_dict['targets_config']['los_prolonged_d']), 1, 0)
	# hospital readmission in 30 days after discharge
	targets["readmission"] = np.where((
		(targets["sec_admittime"].notna()) &
		(targets["sec_admittime"] > targets["dischtime"]) &
		((targets["sec_admittime"] - targets["dischtime"]) <= pd.Timedelta(days=config_dict['targets_config']['readmission_d']))), 1, 0)

	targets = targets.reset_index()
	print("ICU data preprocessed successfully")
	return targets

def _filter_modalitys(mod, hadm_ids, mod_metadata=None):
	mod = mod[mod['hadm_id'].isin(hadm_ids)]
	if mod_metadata is not None:
		mod = pd.merge(mod,mod_metadata,on='itemid')
		mod = mod[mod['valuenum'].between(mod['min'],mod['max'], inclusive='both')]
	
	return mod

def _preprocess_lab_vit(con, config_dict, sql_queries, hadm_ids):
	""" """
	# load lab data and filter invalid mesurments 
	labevent_meatdata = load_file(config_dict['data_paths']['DATA_PATH'], config_dict['data_paths']['LABS_METADATA'])
	labs = sql_from_MIMICIII(con, sql_queries["LABQUERY"], labevent_meatdata['itemid'].tolist())
	labs = _filter_modalitys(labs, hadm_ids, labevent_meatdata)
	print("Laboratory test results data preprocessed successfully")
	
	# load vital data and filter invalid mesurments 
	vital_meatdata = load_file(config_dict['data_paths']['DATA_PATH'], config_dict['data_paths']['VITALS_METADATA'])
	vits = sql_from_MIMICIII(con, sql_queries["VITQUERY"], vital_meatdata['itemid'].tolist())
	vits = _filter_modalitys(vits, hadm_ids, vital_meatdata)
	vits.loc[(vits['feature name'] == 'TempF'),'valuenum'] = (vits[vits['feature name'] == 'TempF']['valuenum']-32)/1.8
	vits.loc[vits['feature name'] == 'TempF','feature name'] = 'TempC'
	print("Vital signs data preprocessed successfully")
	
	# combine labs + vitals
	all_measurements = pd.concat([labs, vits])

	# aggregate per patient, admission, day, and feature using grouper and pivot
	all_measurements_pivot = pd.pivot_table(all_measurements, index=['subject_id', 'hadm_id', pd.Grouper(key='charttime', freq=config_dict['temporal_config']['EVAL_FREQ_H'])], columns=['feature name'], values='valuenum', aggfunc=['mean', 'max', 'min', 'std'])

	# flatten MultiIndex columns
	all_measurements_pivot.columns = [f"{feat}_{stat}" for stat, feat in all_measurements_pivot.columns]
	all_measurements_pivot = all_measurements_pivot.reset_index()
	
	return all_measurements_pivot, labevent_meatdata['feature name'].unique().tolist(), vital_meatdata['feature name'].unique().tolist()

def _preprocess_prescriptions(con, sql, subject_ids, temp_freq):
	"""
	Build per-{temp_freq} prescription features for each (subject_id, hadm_id).
	
	* Load and clean PRESCRIPTIONS for the cohort.
	* Map each drug name to a drug_category via regex patterns.
	* Per {temp_freq} window:
	   - Binary flags: is_<category> (any use in window)
	   - num_distinct_drugs_get: unique drug_name count
	   - new_any_drug: 1 if any category appears for the first time in the admission
		 (via groupby-cummax + shift)
	* For key drugs (antibiotics,sedatives,opioids,steroids,vasopressors,glucose_correction and insulin):
	   - Normalize doses to base units (mg for mg-cats; unit for insulin else ignore)
	   - Compute dose sum and cumsum across windows"""
	
	drug_cat_keywords = get_keywords_cat()[0]
	mg_cats = ['antibiotics','sedatives','opioids','steroids','vasopressors','glucose_correction']
	patterns = {cat: re.compile(rf"(?i)(?<![a-z])(?:{'|'.join(_word2pat(w) for w in words)})(?![a-z])") for cat, words in drug_cat_keywords.items()}
	
	# load lab data and filter invalid mesurments 
	pres = sql_from_MIMICIII(con, sql["PRESCRIPTIONS"], subject_ids)
	
	# lowercase, collapse whitespace, and strip
	pres["drug_name"] = pres["drug_name"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
	pres["dose_unit_rx"] = pres["dose_unit_rx"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
	# match drug to drug category 
	pres["drug_category"] = pres["drug_name"].apply(lambda x: _map_drug_name_2_category(x, patterns))
	pres['dose_val_rx_num'] = pd.to_numeric(pres['dose_val_rx'], errors='coerce')

	# feature engineering for prescriptions
	pres_ = pres[['subject_id','hadm_id','charttime']].join(pd.get_dummies(pres["drug_category"], prefix='is').astype(int))
	drug_cols = [c for c in pres_.columns if c.startswith('is_') and c != 'is_other']
	
	# set 0/1 if any drug in the category was given in the 6h window
	pres_ = pres_.groupby(['subject_id','hadm_id',pd.Grouper(key='charttime', freq=temp_freq)], as_index=False)[drug_cols].max().rename(columns={'bin_time': 'charttime'})

	# count distinct drug types given
	num_of_drugs = pres.groupby(["subject_id","hadm_id",pd.Grouper(key='charttime', freq=temp_freq)], as_index=False)["drug_name"].nunique().rename({"drug_name":"num_distinct_drugs_get"})
	pres_ = pres_.merge(num_of_drugs, on=["subject_id","hadm_id","charttime"], how="left")

	# 0/1 if it’s the first time this category appears in the admission
	# cummax:seen up to *this* row -> shift:seen *before* this row
	pres_ = pres_.sort_values(['subject_id','hadm_id','charttime'])
	prev_seen = (pres_.groupby(['subject_id','hadm_id'])[drug_cols].cummax().shift(fill_value=0).astype(int))
	# marks if a new drug is given 
	pres_['new_drug'] = (((pres_[drug_cols] == 1) & (prev_seen == 0)).astype(int).sum(axis=1) > 0).astype(int)
	
	# for key categories - total dose in the 6h window and cumm dose in base unit
	check_dose = pres[(pres['drug_category'].isin(mg_cats+['insulin'])) & (pres['dose_unit_rx'].isin(['g','mg','mcg','unit','units','_unit'])) & (~pres['dose_val_rx_num'].isna())].copy()

	# Convert mg-based categories to mg
	mask_mg = check_dose['drug_category'].isin(mg_cats)
	check_dose.loc[mask_mg & check_dose['dose_unit_rx'].eq('g'), 'dose_val_rx_num'] *= 1000
	check_dose.loc[mask_mg & check_dose['dose_unit_rx'].eq('mcg'),'dose_val_rx_num'] *= 0.001
	check_dose.loc[mask_mg & check_dose['dose_unit_rx'].isin(['mg','g','mcg']), 'dose_unit_rx'] = 'mg'
	# Insulin → keep only unit
	check_dose.loc[(check_dose['drug_category'].eq('insulin')) & (check_dose['dose_unit_rx'].isin(['units','_unit'])), 'dose_unit_rx'] = 'unit'
	# total dose in the 6h window 
	check_dose = check_dose.groupby(["subject_id","hadm_id","drug_category",pd.Grouper(key='charttime', freq=temp_freq)], as_index=False)["dose_val_rx_num"].sum().rename(columns={'dose_val_rx_num':'dose_sum'})
	# cumulative dose up to this window
	check_dose = check_dose.sort_values(['subject_id','hadm_id','drug_category','charttime'])
	check_dose['dose_cumsum'] = check_dose.groupby(["subject_id","hadm_id","drug_category"])["dose_sum"].cumsum()
	check_dose = (check_dose.pivot(index=['subject_id','hadm_id','charttime'], columns='drug_category', values=['dose_sum','dose_cumsum']).reset_index())
	check_dose.columns = [f'{c[0]}_{c[1]}' if c[0].startswith('dose') else c[0] for c in check_dose.columns]
	check_dose = check_dose.fillna(0.0)
	
	print("Prescriptions data data uploaded and feature engineering done for it successfully")
	return pres_.merge(check_dose, on=['subject_id','hadm_id','charttime'], how='left')

def _preprocess_microbiology(con, sql, subject_ids):
	""" process microbiology for early-hospitalization modeling.
		Returns *one row per subject_id* (and hadm_id) with:
		  - is_site_* one-hots (fixed set from bio_specimen_keywords)
		  - is_r, is_s, is_i, is_p interpretation flags
		  - is_any_culture_pos - organism identified
		  - common organisms one-hots (fixed set from common_organisms_columns) """
	
	_, bio_specimen_keywords, common_organisms_columns = get_keywords_cat()
	_PATTERNS = [(label, re.compile(rx)) for label, rx in bio_specimen_keywords]
	
	# load lab data and filter invalid mesurments 
	bio = sql_from_MIMICIII(con, sql["MICROBIOLOGY"], subject_ids)
	
	# lowercase, collapse whitespace, and strip
	for col in ["specimen_type", "organism", "antibiotic", "interpretation"]:
		bio[col] = bio[col].astype("string").str.strip().str.lower()

	# Coarse site from specimen_type
	bio["site"] = bio["specimen_type"].apply(lambda x: _site_from_specimen_type(x, _PATTERNS))

	# is taken from site 
	site = pd.get_dummies(bio["site"], prefix='test_site').astype(int)
	# any R (resistant) S (susceptible) or I (intermediate) 
	interp = pd.get_dummies(bio["interpretation"], prefix='is').astype(int)
	interp = interp.reindex(columns=[f"is_{c}" for c in ("r","s","i","p")], fill_value=0)
	# positive culture/evidence 
	has_organism = ((bio["organism"].notna()) & (bio["organism"].ne(""))).astype(int)
	# positive culture/evidence for one of the top most commom organisms (found in more than 100 of the train subjects)
	is_common_organism = pd.get_dummies(bio["organism"], prefix='is').astype(int)
	is_common_organism = is_common_organism.reindex(columns=common_organisms_columns, fill_value=0)

	bio_ = pd.concat([bio[["subject_id",'hadm_id']],site,interp,has_organism,is_common_organism], axis=1)
	
	print("Microbiology events data uploaded and feature engineering done for it successfully")
	return bio_.groupby(['subject_id','hadm_id'], as_index=False).max()


def _feature_engineering_lab_vit(df, lab_cols, vits_col):
	# Labs: vt − v0
	lab_mean_cols = [f"{f}_mean" for f in lab_cols if f"{f}_mean" in df.columns]
	lab_baseline = df.groupby(['subject_id','hadm_id'])[lab_mean_cols].transform('first')
	lab_diff = df[lab_mean_cols] - lab_baseline
	lab_diff.columns = [col.split("_")[0]+"_diff" for col in lab_diff.columns]

	# Vitals: vt − vt−1
	vital_mean_cols = [f"{f}_mean" for f in vits_col if f"{f}_mean" in df.columns]
	vital_diff = df.groupby(['subject_id','hadm_id'])[vital_mean_cols].diff()
	vital_diff.columns = [col.split("_")[0]+"_diff" for col in vital_diff.columns]
	
	print("Vital signs and lab data feature engineering done successfully")
	return lab_diff, vital_diff
	

def preprocess_data(subject_ids, con):
	config_dict = get_config()
	sql_queries = get_sql_queries()
	
	# ICU data pre-processing
	icu = _ICU_preprocess(con, sql_queries["ICUQ"], subject_ids, config_dict)
	
	# laboratory test results and vital signs pre-processing
	lab_vit_measurements, lab_cols, vits_col = _preprocess_lab_vit(con, config_dict, sql_queries, icu['hadm_id'])
	
	# new data modalities pre-processing
	prescriptions = _preprocess_prescriptions(con, sql_queries, subject_ids, config_dict['temporal_config']['EVAL_FREQ_H'])
	pres_cols = [c for c in prescriptions.columns if c not in ('subject_id','hadm_id','charttime')]
	
	# merge all datasets
	timegrid = lab_vit_measurements.merge(prescriptions, on=['subject_id','hadm_id','charttime'], how='outer')
	
	microbiology = _preprocess_microbiology(con, sql_queries, subject_ids)
	
	merged_measurements = icu.merge(timegrid, on=['subject_id', 'hadm_id'], how='left')
	merged_measurements = merged_measurements.merge(microbiology, on=['subject_id', 'hadm_id'], how='left')
	merged_measurements[pres_cols] = merged_measurements[pres_cols].fillna(0)

	# filter to use only data collected in the first 48 hours of admittion
	merged_measurements = merged_measurements[merged_measurements["charttime"] - merged_measurements["admittime"] <= pd.Timedelta(hours=config_dict['temporal_config']['EVAL_WINDOW_H'])]
	
	# data imputation
	filled_measurements = merged_measurements.sort_values(['subject_id', 'hadm_id', 'charttime'])
	filled_measurements[lab_vit_measurements.columns] = filled_measurements.groupby(['subject_id', 'hadm_id'])[lab_vit_measurements.columns].ffill()
	
	# feature engineering
	lab_diff, vital_diff = _feature_engineering_lab_vit(filled_measurements, lab_cols, vits_col)
	
	df = pd.concat([filled_measurements, lab_diff, vital_diff], axis=1)
	
	print("Pre-process done")
	return df