# Zygomatic EEG model for epileptiform discharges detection in the medial temporal lobe

This notebook details the end-to-end pipeline for developing and validating an XGBoost machine learning model to detect medial temporal lobe(MTL) epileptiform discharges (ED) from zygomatic (zEEG) channels.

The process is divided into three main parts:

1.  **Part 1: Preprocessing and Label Generation**
    * Loads raw intracranial depth (MTL) data (`.fif` files).
    * Applies a pre-trained "depth model" to generate ground-truth labels for each 250ms window.
    * Extracts features from the corresponding zEEG channels for the same windows.
    * Saves the processed data (features + labels) to `zeeg_training_data.pkl`.

2.  **Part 2: zEEG Model Training and Validation**
    * Loads the preprocessed data from Part 1.
    * Performs data balancing and augmentation.
    * Trains and evaluates an XGBClassifier, and saves the final, trained model (`zeeg_model.pkl`).

3.  **Part 3: Application on Example Non-Invasive Data (HC vs EPI)**
    * Loads the final trained zEEG model.
    * Runs the model on independent cohorts of Healthy Controls (HC) and Epilepsy patients (EPI).
    * Calculates and compares the ED rate between the two groups.
