import pandas as pd
import numpy as np
import pickle as pkl

import duckdb
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

from preprocessing.utils import load_file, sql_from_MIMICIII, age, ethnicity_to_ohe, split_data, data_norm, imputation_by_baseline, generate_series_data
from preprocessing.config import get_config
from preprocessing.preprocess_pipeline import preprocess_data

from model.model import MultiTaskSeqGRUAE

def run_pipeline_on_unseen_data(subject_ids ,client):
	"""
	Run your full pipeline, from data loading to prediction.

	:param subject_ids: A list of subject IDs of an unseen test set.
	:type subject_ids: List[int]

	:param client: A BigQuery client object for accessing the MIMIC-III dataset.
	:type client: google.cloud.bigquery.client.Client

	:return: DataFrame with the following columns:
							- subject_id: Subject IDs, which in some cases can be different due to your analysis.
							- mortality_proba: Prediction probabilities for mortality.
							- prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
							- readmission_proba: Prediction probabilities for readmission.
	:rtype: pandas.DataFrame
	"""
	# import models configuration params
	config_dict = get_config()
	
	# read the full train_features list and Standardization and Imputation params from the train data 
	with open(f"pre-processing/{config_dict['data_paths']['DATA_PATH']}/models_params_dict.pkl", 'rb') as f:
		models_params = pkl.load(f)
	
	# preprocess_datax`
	df = preprocess_data(subject_ids, client)
	X = df.drop(columns=['hadm_id','dischtime','dod','dob','deathtime','mortality','prolonged_stay','readmission','sec_admittime'])
	y = df[['subject_id'] + config_dict['TASKS']]
	
	# add missing features to the data - fill with 0s
	missing_cols = list(set(models_params['train_features'])-set(X.columns))
	X[missing_cols] = 0
	
	# Standardization (scaler fitted by TRAIN only)
	X, _ = data_norm(X, models_params["numeric_cols"], scaler=models_params["scaler"])
	# Imputation by first day baseline (baseline calced by TRAIN only)
	X, _ = imputation_by_baseline(X, models_params["numeric_cols"], baseline=models_params["imputation_baseline"])
	X = X.fillna(0)
	X = X.drop(columns=['admittime']).reset_index(drop=True)
	
	# remove unknown featurs
	X = X[models_params['train_features'] + ['subject_id', 'charttime']]
	
	# Generate padded sequences + masks
	X, y, mask, fin_subject_ids = generate_series_data(X, y, time_col="charttime")
	
	# load trained model 
	model = MultiTaskSeqGRUAE(input_dim=X.shape[-1], latent_dim=64, SupCon_latent_dim=32, pooling="mean+max+final")
	model.load_state_dict(torch.load("model.pt", weights_only=True))
	model.eval()
	
	# run model 
	pred_logits = predict_logits(model, X, mask)
	probs_BCE_cal_test = predict_proba_calibrated(models_params['calibrator'], pred_logits)
	
	fin_df = pd.DataFrame(probs_BCE_cal_test, columns=['mortality_proba', 'prolonged_LOS_proba', 'readmission_proba'])
	fin_df['subject_id'] = fin_subject_ids
	return(fin_df)
