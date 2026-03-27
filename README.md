# Activity Classification Project
The project involves working on an open DAS(Distributed Acoustic Sensing) dataset. This dataset contains six types of Phi-OTDR events, including background noises, digging, knocking, shaking, watering and walking, in total of 15,419 samples.The dataset is split 80-20 between training and testing sets.
By extracting 64 features from the dataset , we trained various machine learning models such as Random Forest , SVM etc.
We also trained the a CNN that can acheive an accuracy of upto 95.73%.


## Project Overview

This project implements activity recognition using two approaches:
1. **Traditional ML Classifiers**: Scikit-learn models (Random Forest, SVM, Gradient Boosting, Logistic Regression)
2. **Deep Learning**: Convolutional Neural Networks (CNN)

### Activity Classes (6 categories)
- `01_background`: Background noise/no activity (3,094 samples)
- `02_dig`: Digging motion (2,512 samples)
- `03_knock`: Knocking sound (2,530 samples)
- `04_water`: Water-related activity (2,298 samples)
- `05_shake`: Shaking motion (2,728 samples)
- `06_walk`: Walking motion (2,450 samples)

**Total samples**: 15,612 (Train: 12,478 | Test: 3,134)

## Dataset Structure

```
70-30 Data/
├── train/                          # Training data (80%)
│   ├── features_dataset_train.csv  # Extracted features
│   ├── label.txt                   # Class labels
│   └── 01_background/ ... 06_walk/ # Raw .mat files by class
│
├── test/                           # Test data (20%)
│   ├── features_dataset_test.csv   # Extracted features
│   ├── label.txt                   # Class labels
│   └── 01_background/ ... 06_walk/ # Raw .mat files by class
│
└── [other directories and files]   # Results and scripts
```

## Files Description

### Main Scripts

| File | Purpose |
|------|---------|
| `train_classifier.py` | Train and evaluate traditional ML classifiers with MLflow logging |
| `train_classifier_mlflow.py` | MLflow-integrated classifier training |
| `train_cnn.py` | Build and train CNN models |
| `log_cnn_to_mlflow.py` | Log CNN model results to MLflow |
| `feature_extraction.py` | Extract features from raw .mat files |
| `check.py` | Data validation and inspection utility |

### Results Directories

- **`classifier_results/`**: Traditional ML model outputs
  - `model_comparison.csv`: Performance metrics comparison
  
- **`cnn_results/`**: Deep learning model outputs
  - `best_cnn_model.h5`: Best trained CNN model
  - `cnn_model.keras`: Keras format model
  - `cnn_metrics.csv`: Performance metrics
  - `training_history.csv`: Training/validation history

- **`mlruns/`**: MLflow experiment tracking
  - Stores experiment runs and model metadata

## Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install scikit-learn
pip install pandas numpy scipy
pip install matplotlib seaborn
pip install mlflow
```

### Usage

#### 1. Feature Extraction
Extract features from raw .mat files:
```bash
python feature_extraction.py
```

#### 2. Train Traditional Classifiers
```bash
python train_classifier.py
```
This will:
- Train multiple models (Random Forest, SVM, Gradient Boosting, Logistic Regression)
- Generate evaluation metrics (accuracy, precision, recall, F1-score)
- Create visualizations (confusion matrices, ROC curves)
- Log results to MLflow

#### 3. Train CNN
```bash
python train_cnn.py
```
This will:
- Build and train a CNN architecture
- Evaluate on test set
- Save best model
- Log metrics and training history

#### 4. Log CNN Results to MLflow
```bash
python log_cnn_to_mlflow.py
```

#### 5. Validate Data
```bash
python check.py
```

## Data Statistics

### Training Set (80%)
- 01_background: 2,475
- 02_dig: 2,010
- 03_knock: 2,024
- 04_water: 1,827
- 05_shake: 2,182
- 06_walk: 1,960
- **Total**: 12,478

### Test Set (20%)
- 01_background: 619
- 02_dig: 502
- 03_knock: 506
- 04_water: 471
- 05_shake: 546
- 06_walk: 490
- **Total**: 3,134

## Model Performance

Check the results in:
- `classifier_results/model_comparison.csv` - Traditional ML model metrics
- `cnn_results/cnn_metrics.csv` - CNN model metrics

### Typical Metrics Reported
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

## MLflow Integration

Experiment tracking is enabled via MLflow. View results:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

## File Specifications

### Feature Dataset Format
- CSV files with extracted feature columns
- Last column typically contains the activity label
- Rows represent individual samples

### Raw Data Format
- `.mat` files: MATLAB format containing raw signal/sensor data
- Organized by activity class subdirectories

## Notes

- All paths use absolute references to `/home/monish-m/Downloads/das/70-30 Data`
- Random seeds are set for reproducibility
- GPU support is detected and used if available (for CNN)
- StandardScaler is applied to features for traditional ML models

## Output Files Generated

After running the scripts, expect:
- CSV files with model metrics and predictions
- Serialized models (.pkl, .h5)
- Visualizations (PNG/PDF plots)
- MLflow experiment logs
- Training/validation history data
