# config file for MLHC MIMIC-III preprocessing

# needed data file paths 
paths = {
	"DATA_PATH": "./data", # data root directory
	"INITIAL_COHORT": "initial_cohort.csv",
	"LABS_METADATA": "labs_metadata.csv",
	"VITALS_METADATA": "vital_metadata.csv",
	"DRIVE_PATH": "/content/drive/MyDrive/MIMIC-III", # MIMIC-III DB shortcut path in drive
	"MIMIC_DB_NAME": "mimiciii.duckdb"
}

# temporal configuration for first-admission prediction
temporal_config = {
	"EVAL_WINDOW_H": 48,   # use first 48h for features
	"GAP_H": 6,            # 6h prediction gap to avoid leakage
	"EVAL_FREQ_H": "6h",   # feature sampling frequency (e.g., aggregate hourly)
}

# target definitions
targets_config = {
	"mortality_d": 30, # mortality during admission or  post-discharge
	"los_prolonged_d": 7, # hospital prolonged stay
	"readmission_d": 30, # hospital readmission 
}

TASKS = ['mortality','prolonged_stay','readmission']

def get_config():
	return {"data_paths" : paths, "temporal_config" : temporal_config, "targets_config" : targets_config, "TASKS" : TASKS}