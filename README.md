# Diabetes Readmission Classifier

This project implements machine learning models to predict early readmission (within 30 days) for diabetes patients using the Diabetes 130-US hospitals dataset.

## Project Structure
- `knn_testing.py`: Implementation of K-Nearest Neighbors classifier
- `diabetic_with_weight.csv`: Dataset including weight information
- `diabetic_without_weight.csv`: Dataset excluding weight information

## Dependencies

Required Python packages:
```
numpy>=1.17.0
pandas
scikit-learn
matplotlib
seaborn
lightgbm>=4.6.0
```

You can install all dependencies using pip:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn lightgbm
```

## Dataset
The project uses the Diabetes 130-US hospitals dataset, which contains:
- Patient demographics
- Hospital information
- Medical information
- Readmission status

Two versions of the dataset are used:
1. With weight information
2. Without weight information

## Models
Currently implemented:
- K-Nearest Neighbors (KNN) classifier
  - Uses cross-validation
  - Includes feature selection
  - Handles class imbalance
  - Evaluates performance using multiple metrics (accuracy, precision, recall, F1)

## Usage
To run the KNN experiments:
```bash
python knn_testing.py
```

This will:
1. Load and preprocess the data
2. Train and evaluate KNN models
3. Generate performance plots
4. Compare results with and without weight information

## Performance Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Learning Curves
- Confusion Matrices 