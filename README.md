# TRIAGE: Time-series Risk prediction from ICU data via an Autoencoder of GRUs with contrastive Embeddings

## Overview

TRIAGE is a multi-task deep learning model for early ICU outcome prediction. Using the first 48 hours of patient data, it jointly predicts three critical outcomes:
- **Mortality**: In-hospital or 30-day mortality
- **Prolonged Stay**: ICU stay longer than 7 days
- **Readmission**: Hospital readmission within 30 days

The model combines a GRU-based sequence autoencoder with supervised contrastive learning to create interpretable patient embeddings while achieving strong predictive performance on imbalanced clinical data.

## Key Features

- **Multi-task learning**: Simultaneous prediction of three ICU outcomes with shared representations
- **Temporal modeling**: GRU-based architecture captures time-series dynamics over 48-hour windows
- **Contrastive embeddings**: Supervised contrastive learning creates discriminative patient representations
- **Multimodal integration**: Incorporates demographics, vital signs, laboratory tests, medications, and microbiology data
- **Calibrated predictions**: Platt scaling ensures reliable probability estimates for clinical decision support
- **Interpretability**: SHAP-based feature importance analysis and embedding-based patient similarity

## Dataset

This project uses the MIMIC-III database (Medical Information Mart for Intensive Care III). The cohort includes:
- 27,636 unique ICU patients
- 190,963 observations (6-hour windows)
- 274 multimodal features across 4 categories:
  - Vital signs (36 features)
  - Laboratory tests (33 features)
  - Medications (77 features)
  - Microbiology (33 features)

## Installation

### Prerequisites
- Python 3.9
- CUDA-capable GPU (recommended)
- Access to MIMIC-III database

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/litalv/project.git
cd project
```

2. Create conda environment from the provided file:
```bash
conda env create -f environment.yml
conda activate triage
```

## Project Structure

```
project/
├── preprocessing/          # Data preprocessing pipeline
│   ├── config.py          # Configuration settings
│   ├── preprocess_pipeline.py  # Main preprocessing pipeline
│   ├── sql.py             # SQL query templates
│   ├── utils.py           # Helper functions
│   ├── name_keywords.py   # Drug/micro category mappings
│   ├── exploratory_data_analysis.ipynb  # EDA notebook
│   └── preprocess_training_data.ipynb   # Data preparation notebook
├── model/                 # Model implementation
│   ├── model.py          # GRU autoencoder architecture
│   ├── dataset.py        # PyTorch dataset classes
│   ├── loss.py           # Loss functions (BCE, SupCon, reconstruction)
│   ├── trainning.py      # Training loops and utilities
│   ├── prediction.py     # Inference and calibration
│   └── utils/            # Evaluation and visualization tools
├── model fine-tuning.ipynb  # Hyperparameter tuning notebook
├── full_evaluation.ipynb    # Complete model evaluation
└── unseen_data_evaluation.py # Script for new data evaluation
```

## Usage

### For Course Evaluation (MLHC Spring 2025)

This project implements the required `unseen_data_evaluation.py` function for automated evaluation:

```python
from unseen_data_evaluation import run_pipeline_on_unseen_data

# Run on unseen test data
predictions = run_pipeline_on_unseen_data(subject_ids, bigquery_client)
```

The function:
1. Takes subject IDs and BigQuery client as input
2. Runs the complete preprocessing pipeline on unseen data
3. Applies the pre-trained model
4. Returns calibrated prediction probabilities for all three outcomes

**Required Files for Evaluation:**
- `unseen_data_evaluation.py` - Main evaluation function
- `model/model.pt` - Pre-trained model weights
- `model/models_params_dict.pkl` - Training parameters and calibrators
- All preprocessing and model code

### For Research and Development

#### 1. Data Preprocessing

Prepare MIMIC-III data using the preprocessing pipeline:

```python
from preprocessing.preprocess_pipeline import preprocess_data
import duckdb

# Connect to MIMIC-III database
client = duckdb.connect('path/to/mimic.db')  # or BigQuery client

# Load subject IDs
subject_ids = pd.read_csv('initial_cohort.csv')['subject_id'].tolist()

# Run preprocessing (extracts first 48h, creates 6h windows)
df = preprocess_data(subject_ids, client)
```

#### 2. Model Training

Train the multi-task GRU autoencoder:

```python
from model.model import MultiTaskSeqGRUAE
from model.trainning import train_model

# See model fine-tuning.ipynb for complete training pipeline
# Key components:
# - Balanced positive-per-task sampling
# - Warm-up phase with reconstruction + contrastive loss
# - Multi-task loss with weighted BCE + SupCon + reconstruction
```

#### 3. Evaluation and Analysis

Use the provided notebooks:
- `full_evaluation.ipynb` - Complete model evaluation with metrics and plots
- `model fine-tuning.ipynb` - Hyperparameter tuning and ablation studies
- `preprocessing/exploratory_data_analysis.ipynb` - Data exploration and feature analysis

## Model Performance

Performance on held-out test set (mean ± std across bootstrap samples):

| Task | AUROC | AUPR |
|------|-------|------|
| Mortality | 0.792 ± 0.008 | 0.782 ± 0.010 |
| Prolonged Stay | 0.856 ± 0.005 | 0.431 ± 0.012 |
| Readmission | 0.691 ± 0.015 | 0.084 ± 0.008 |

## Key Findings

- **Shared features**: Age, polypharmacy, and vital sign variability are important across all tasks
- **Task-specific patterns**:
  - Mortality: Driven by acute physiology (electrolytes, acid-base)
  - Prolonged stay: Care intensity and persistent instability
  - Readmission: Treatment complexity and residual hemodynamic issues
- **Temporal dynamics**: Early features matter most for mortality, while late features are crucial for readmission