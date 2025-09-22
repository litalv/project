# Pre-processing (MLHC — ICU Outcomes)
This module builds the feature set and labels for the ICU prediction tasks, matching the instructioned timeline (first 48 h, 6 h prediction gap, min 54 h stay; first hospital admission only) and the three targets (mortality, LOS>7 d, 30-day readmission).

## Steps implemented

1. **Cohort + targets** (`_ICU_preprocess`)
   - First hospital admission only, keep stays >=54 h.
   - Demographics: age, gender, ethnicity one-hots.
   - Targets:
     - `mortality`: death in hospital or ≤30 d post-discharge.
     - `prolonged_stay`: LOS > 7 days.
     - `readmission`: next hospital admission within 30 days.

2. **Labs & Vitals** (`_preprocess_lab_vit`)
   - From `labevents` (labs) and `chartevents` (vitals), 0–48 h since admit.
   - Validate using metadata, convert TempF→TempC.
   - Resample to uniform 6 h grid, pivot with mean/max/min/std.
   - **Feature engineering**
      - Labs: baseline deltas (`<lab>_diff`).
      - Vitals: per-window deltas (`<vital>_diff`).

3. **Prescriptions** (`_preprocess_prescriptions`)
   - From `prescriptions`, 0–48 h.
   - Map drug names → categories (regex rules).
   - Features: binary flags (`is_<cat>`), new_drug, per-window and cumulative doses for chosen set of drugs with unit normalization (mg/g/mcg, insulin units).

4. **Microbiology** (`_preprocess_microbiology`)
   - From `microbiologyevents`, 0–48 h.
   - Specimen site one-hots, culture interpretation flags (`is_r/is_s/...`), organism one-hots,
     `is_any_culture_pos`.

5. **Merge + filter + impute**
   - Merge ICU + labs/vitals grid + prescriptions + microbiology.
   - Keep only 0–48 h.
   - Forward-fill labs/vitals on 6 h grid; fill missing prescription flags with 0.

6. 

7. **Output**
   - Final DataFrame with demographics, labels, and all engineered features.

## Files

- `preprocess_pipeline.py` — main pipeline (entry: `preprocess_data(subject_ids, con)`).
- `sql.py` — SQL templates for queries.
- `name_keywords.py` — regex definitions for drug categories and microbiology.
- `config.py` — paths and temporal config (EVAL_WINDOW_H=48, GAP_H=6, EVAL_FREQ_H=6h).
- `utils.py` — helpers for I/O, DB access, scaling, sequence generation.
- `unseen_data_evaluation.py` — placeholder to be completed (runs pipeline on unseen test set).
