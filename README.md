Project Title: A Deep-Learning Approach to Identifying Alcohol Use Disorder Through Abnormal Reductions in Transient EEG Amplitudes Using a Multilayer Binary Classification Learning Model

---

## Overview

This repository contains code for a science fair project that uses deep learning to identify Alcohol Use Disorder (AUD) from EEG recordings.  

The core of the project is a set of Python scripts that:
- **Extract and clean raw EEG trials** from subject archives.
- **Build balanced datasets** and splits (train / validation / test).
- **Train and evaluate deep-learning models** (CNN, RNN/LSTM, and CNN–LSTM hybrids).

Tutorial notebooks and practice scripts (for Pandas, matplotlib, file I/O, etc.) live alongside the project but are **not** part of the experimental pipeline.

---

## Core project scripts

- **`CNN.py`**  
  - End‑to‑end pipeline for a **2D CNN classifier**.  
  - Can optionally:
    - Extract subject `.tar.gz` EEG archives into a `processed/` directory.
    - Clean individual trial text files into `new_*.csv` files (one per trial).
    - Build:
      - `trial_files_descriptions.csv` (maps each cleaned trial to its stimulus type and class label),
      - `groups.csv` (which dataset split a trial belongs to),
      - `targets.csv` (class labels per trial).
  - Loads trials according to `groups.csv` / `targets.csv`, trains the CNN, and writes:
    - Accuracy and loss plots (`*_model_plot_1.png`, `*_model_plot_2.png`),
    - A text model summary (`model_summaries.txt`),
    - A log line with experiment metadata (`experimentdata.txt`),
    - Copies of `groups.csv` and `targets.csv` for reference.

- **`RNN.py`**  
  - Uses the **same data layout and preprocessing functions** as `CNN.py`.  
  - Builds and trains an **LSTM-based time‑series classifier** (no convolutional layers).  
  - Useful as a **baseline RNN model** to compare against the CNN/CNN‑LSTM approaches.

- **`CNNLSTM.py`**  
  - Also shares the same preprocessing and data loading logic.  
  - Implements a **hybrid architecture**:
    - First applies several 2D convolution + max‑pool layers,
    - Then reshapes the feature maps and feeds them into one or more LSTM layers,
    - Finally uses dense layers for binary classification.  
  - Designed to test whether combining CNN feature extraction with temporal modeling improves performance.

- **`AccuracyTest.py`**  
  - A more specialized **hyperparameter / configuration test script**.  
  - Loops over different **batch sizes** to study their effect on model performance.  
  - Uses:
    - A **reduced channel set** (`['FP1', 'FP2', 'FPZ']`), and
    - A specific **stimulus subset** (e.g., `'S1 obj'`).  
  - Trains an LSTM-based model and records accuracy for each configuration.

These four scripts (`CNN.py`, `RNN.py`, `CNNLSTM.py`, and `AccuracyTest.py`) are the main files relevant to the project’s results.

---

## Data layout and paths

The scripts expect EEG data to live **outside** this repo, under a directory similar to:

- `C:/eeg/sciencefair22-23/data1/eeg_full/` (for `CNN.py` and `CNNLSTM.py`)
- `C:/eeg/sciencefair22-23/data/eeg_full/` (for `RNN.py`, `AccuracyTest.py`, and some utilities)

Within that directory, the scripts assume:
- **Raw subject archives**: `*.tar.gz` files in `eeg_full/`
- **Processed data directory**:  
  - `processed/` (created by the scripts)  
  - Contains subject folders like `co###a` / `co###c`, each with per‑trial text/CSV files.
- **Metadata CSVs** (generated during preprocessing):  
  - `processed/trial_files_descriptions.csv`  
  - `processed/groups.csv`  
  - `processed/targets.csv`
- **Experiments directory**:  
  - `eeg_full/experiments/`  
  - Each run creates timestamped files (plots, summaries, logs) under this folder.

If you move the data, you must update the `full_data_dir` variable near the top of each script to point to the correct location.

---

## Running the pipelines

1. **Set up a Python environment**
   - Install dependencies (versions will depend on your system):
     - `tensorflow` / `keras`
     - `pandas`
     - `numpy`
     - `matplotlib`
     - `scikit-learn`
     - `pydot` (only needed for model diagram plotting)

2. **Configure paths and options in the script**
   - Open one of the main scripts (for example, `CNN.py`).
   - At the top, verify or change:
     - `full_data_dir` (points to your EEG data root),
     - `do_initial_extraction` (`0` or `1`),
     - `process_raw_data` (`0` or `1`),
     - `datasubset` (e.g., `'S'`, `'S1 obj'`, `'S2 match'`, `'S2 nomatch'`),
     - Training hyperparameters: `lr`, `lstm_units`, `epochs`, `batch_size`, and train/validation split fractions.

3. **Initial extraction (first‑time setup only)**
   - In `CNN.py` (or `RNN.py`), set:
     - `do_initial_extraction = 1` to:
       - Create the `processed/` and `experiments/` directories,
       - Extract all subject `.tar.gz` archives into `processed/` and gunzip the trial files.
     - `process_raw_data = 1` to:
       - Create `new_*` cleaned trial files,
       - Build `trial_files_descriptions.csv`, `groups.csv`, and `targets.csv`.  
   - Run the script once, then:
     - Set `do_initial_extraction = 0` and `process_raw_data = 0` for future runs (to avoid re‑doing the same work).

4. **Train and evaluate a model**
   - With `do_initial_extraction` and `process_raw_data` set to `0`, simply run:
     - `CNN.py` for the CNN model,
     - `RNN.py` for the LSTM‑only model,
     - `CNNLSTM.py` for the hybrid CNN–LSTM model,
     - `AccuracyTest.py` for batch‑size / configuration sweeps with a small channel subset.
   - Each run will:
     - Load data according to `groups.csv` and `targets.csv`,
     - Train the model with early stopping or model checkpointing (depending on the script),
     - Save plots and logs under `eeg_full/experiments/`.

You can use your editor’s run configuration (for example, the VS Code entry in `.vscode/launch.json`) to run “Python: Current File” on any of these scripts.

---

## Non‑project / tutorial material

The following files and folders are **practice or tutorial resources** and are **not used** in the experiments:
- `PandasTutorials/`
- `DataScienceVSCode/`
- `FileObjects_Tutorial/`
- `OS_Tutorial.py`
- `W3Schools_numpyTutorial.py`

Utility scripts like `import gzip.py` (loading a specific saved model with `pandas.read_pickle`) and `krish.py` (creating experiment subfolders) are convenience helpers and not part of the main analysis pipeline.

