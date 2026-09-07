# Zygomatic EEG model for non-invasive detection of medial temporal lobe epileptiform discharges

This repository contains the analysis pipeline used to develop, validate, and apply an XGBoost-based model for detecting medial temporal lobe (MTL) epileptiform discharges (EDs) from non-invasive zygomatic EEG (zEEG) recordings.

## Analysis overview

The code is organized according to the main study phases:

- **`phase_1.ipynb`**  
   In Phase 1, simultaneous intracranial and zEEG recordings are preprocessed, divided into 250-ms epochs, and used to generate depth-derived ED labels and corresponding zEEG features. These data are then used to train and evaluate the zEEG XGBoost classifier.

- **`phase_2_3.ipynb`**
   In Phases 2 and 3, the trained model is applied to independent non-invasive recordings. The notebooks include the preprocessing steps, feature extraction, and calculation of ED occurrence rates.

- **`zeeg_utils.py`**  
  Shared utilities for epoching, normalization, and depth/zEEG feature extraction.

- **`model_config.json`**  
  Contains the complete configuration of the XGBoost model used in the study.

- **`requirements.txt`**  
  Lists the Python packages and versions used for the analyses.
