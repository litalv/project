import os
import re
import pandas as pd
import numpy as np

import duckdb
# Mount Google Drive
from google.colab import drive

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# data processing 

def load_file(folder_path, file_name):
	""" help function for loading data from a CSV to a DataFrame """
	
	full_path = os.path.join(folder_path, file_name)
	
	try:
		csv_res = pd.read_csv(full_path)
		print(f"Successfully loaded {file_name}")
		return csv_res
	except FileNotFoundError:
		print(f"Error: '{file_name}' not found.")
		print("Please ensure the file exists and the path is correct.")
		return None
	
def sql_from_MIMICIII(con, sql, vars):
	return con.execute(sql, [vars]).fetchdf().rename(str.lower, axis='columns')

def split_data(X, y, groups_df): 
	""" Perform train-val-test (80%/10%-10%) split using GroupShuffleSplit """
	# Train / Temp (80% / 20%)
	gss_train = GroupShuffleSplit(n_splits=1, test_size=0.2)
	train_idx, temp_idx = next(gss_train.split(X, y, groups=groups_df))
	X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
	X_temp,  y_temp  = X.iloc[temp_idx], y.iloc[temp_idx]

	# Test / Val (50%/50% of the temp → 10% each of total)
	gss_test_val = GroupShuffleSplit(n_splits=1, test_size=0.5)
	val_idx, test_idx = next(gss_test_val.split(X_temp, y_temp, groups=groups_df.iloc[temp_idx]))
	X_test,  y_test  = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
	X_val,   y_val   = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
	
	return X_train, y_train, X_val, y_val, X_test, y_test

def data_norm(df, numeric_cols, scaler=None):
	""" Standardize numeric features 
		scaler is fitted over TRAIN only so if train->scaler=None else not None """
	if scaler is None:
		scaler = StandardScaler()
		scaler.fit(df[numeric_cols])
	
	df.loc[:, numeric_cols] = scaler.transform(df[numeric_cols])
	return df, scaler
	
def imputation_by_baseline(df, numeric_cols, baseline=None):
	""" Imputation numeric features by first day baseline 
		baseline calulated by TRAIN only so if train->baseline=None else not None """
	if baseline is None:
		baseline = df[df.charttime.dt.date == df.admittime.dt.date].mean(axis=0).fillna(0)
	
	df.loc[:, numeric_cols] = df[numeric_cols].fillna(baseline)
	return df, baseline

# basic ICU feature extraction

def age(admittime, dob):
	if admittime < dob:
		return 0
	return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

def ethnicity_to_ohe(hosps):
	# ethnicity to category
	hosps.ethnicity = hosps.ethnicity.str.lower()
	hosps.loc[(hosps.ethnicity.str.contains('^white')),'ethnicity'] = 'white'
	hosps.loc[(hosps.ethnicity.str.contains('^black')),'ethnicity'] = 'black'
	hosps.loc[(hosps.ethnicity.str.contains('^hisp')) | (hosps.ethnicity.str.contains('^latin')),'ethnicity'] = 'hispanic'
	hosps.loc[(hosps.ethnicity.str.contains('^asia')),'ethnicity'] = 'asian'
	hosps.loc[~(hosps.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))),'ethnicity'] = 'other'

	# ethnicity to one hot encoding
	hosps['eth_white'] = (hosps['ethnicity'] == 'white').astype(int)
	hosps['eth_black'] = (hosps['ethnicity'] == 'black').astype(int)
	hosps['eth_hispanic'] = (hosps['ethnicity'] == 'hispanic').astype(int)
	hosps['eth_asian'] = (hosps['ethnicity'] == 'asian').astype(int)
	hosps['eth_other'] = (hosps['ethnicity'] == 'other').astype(int)
	hosps.drop(['ethnicity'], inplace=True, axis=1)
	return hosps
	

# help functions for preprocess_prescriptions
def _word2pat(w):
	return re.escape(w).replace(r"\ ", r"[\s\-]*")

def _map_drug_name_2_category(name, patterns):
	""" Return matching category for a given drug name, If none match, return 'other' """
	s = name or ""
	for cat, rx in patterns.items():
		if rx.search(s):
			return cat
	return "other"

# help functions for preprocess_microbiology
def _site_from_specimen_type(s, PATTERNS):
	""" Map specimen_type to a microbiology category (or 'other') """
	if not s:
		return "other"
	s = str(s).strip().lower()

	for label, rx in PATTERNS:
		if rx.search(s):
			return label
	return "other"


# time series format
def generate_series_data(df_X, df_y, time_col="charttime"):
	""" padding function 
		expect input of df_X (contain: subject_id, charttime, features...) and df_y (one row per subject_id with labels)
		
		returns:
			- X_seq: a 3D array (N_subjects, T_max, D_features) of padded sequences
			- Y: a 2D array (N_subjects, 3) of subject-level labels aligned to X_seq
			- mask: a 2D array (N_subjects, T_max) with 1s where the sequence is real data and 0s where it’s padding
	"""
	# sort by subject&time, then group each subject’s time series
	g = df_X.sort_values(['subject_id', time_col]).groupby('subject_id')
	# build per-subject sequences keepping only feature columns
	seqs = [grp.drop(columns=['subject_id', time_col]).to_numpy(dtype='float32') for _, grp in g]
	lengths = [s.shape[0] for s in seqs] # true lengths (before padding)
	
	# pad all to the same length → shape (N, T_max, D)
	X_seq = pad_sequences(seqs, padding='post', dtype='float32')
	
	# mask
	mask = np.zeros((len(seqs), X_seq.shape[1]), dtype=np.float32)
	for i, L in enumerate(lengths): mask[i, :L] = 1.0
	
	# sequence-level labels
	order = [key for key, _ in g]
	df_y_ = df_y.drop_duplicates(inplace=False)
	Y = df_y_.set_index('subject_id').loc[order][['mortality','prolonged_stay','readmission']].to_numpy(dtype='float32')
	
	return X_seq, Y, mask, df_y_.set_index('subject_id').loc[order].index.to_list()