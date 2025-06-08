import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """Preprocess data for gradient boosting"""
    df = df.copy()
    
    # Create binary target
    df['early_readmission'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])
    
    # Separate features and target
    X = df.drop(columns=['early_readmission'])
    y = df['early_readmission']
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = X[col].fillna('Missing')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Fill remaining missing values
    X = X.fillna(0)
    
    cat_idx = [X.columns.get_loc(col) for col in categorical_columns]
    return X.values, y.values, cat_idx, X.columns.tolist()

def save_classifier_data(classifier_name, train_sizes, train_scores, val_scores, test_scores=None):
    """
    Save learning curve data for a classifier
    """
    filename = f'learning_curves_{classifier_name.lower().replace(" ", "_")}.npy'
    with open(filename, 'wb') as f:
        np.save(f, train_sizes)
        np.save(f, train_scores)
        np.save(f, val_scores)
        if test_scores is not None:
            np.save(f, test_scores)

def save_regularization_data(classifier_name, param_values, train_scores, val_scores):
    """
    Save regularization effect data for a classifier
    """
    filename = f'regularization_{classifier_name.lower().replace(" ", "_")}.npy'
    with open(filename, 'wb') as f:
        np.save(f, param_values)
        np.save(f, train_scores)
        np.save(f, val_scores)

def save_error_curves(classifier_name, train_sizes, train_errors, val_errors):
    """
    Save error curve data for a classifier
    """
    filename = f'error_curves_{classifier_name.lower().replace(" ", "_")}.npy'
    with open(filename, 'wb') as f:
        np.save(f, train_sizes)
        np.save(f, train_errors)
        np.save(f, val_errors)

def save_complexity_data(classifier_name, complexity_values, train_scores, val_scores):
    """
    Save model complexity data for a classifier
    """
    filename = f'complexity_{classifier_name.lower().replace(" ", "_")}.npy'
    with open(filename, 'wb') as f:
        np.save(f, complexity_values)
        np.save(f, train_scores)
        np.save(f, val_scores)

def run_lgbm_experiment(csv_file, label=''):
    print(f"\n{'='*50}")
    print(f"Running LightGBM Experiment: {label}")
    print(f"{'='*50}")
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")
    X, y, cat_idx, feature_names = preprocess_data(df)
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    # Apply SMOTE to training set only
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, class_weight=None, verbose=-1)
    grid = RandomizedSearchCV(lgbm, param_grid, scoring='f1', cv=2, n_jobs=-1, verbose=1, n_iter=50, random_state=42)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")
    # Threshold tuning on validation set
    val_probs = best_model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (val_probs > thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"Best threshold on validation set: {best_thresh:.2f} (F1={best_f1:.3f})")
    # Validation metrics with tuned threshold
    y_val_pred = (val_probs > best_thresh).astype(int)
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'threshold': best_thresh
    }
    # Test metrics with tuned threshold
    test_probs = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_probs > best_thresh).astype(int)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'threshold': best_thresh
    }
    print(f"\nValidation Metrics:")
    for metric, score in val_metrics.items():
        if metric != 'threshold':
            print(f"{metric.capitalize()}: {score:.3f}")
    print(f"\nTest Metrics:")
    for metric, score in test_metrics.items():
        if metric not in ['confusion_matrix', 'threshold']:
            print(f"{metric.capitalize()}: {score:.3f}")
    print(f"\nConfusion Matrix (Test Set):")
    print(test_metrics['confusion_matrix'])
    print(f"Threshold used: {best_thresh:.2f}")
    
    # Save learning curve data
    train_sizes = np.linspace(0.1, 0.9, 10)
    train_f1s = []
    val_f1s = []
    test_f1s = []
    
    for frac in train_sizes:
        X_train_part, _, y_train_part, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=42)
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_part, y_train_part, test_size=0.2, stratify=y_train_part, random_state=42)
        
        # Apply SMOTE to training set only
        smote = SMOTE(random_state=42)
        X_train_part, y_train_part = smote.fit_resample(X_train_part, y_train_part)
        
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_model.learning_rate,
            max_depth=best_model.max_depth,
            n_estimators=best_model.n_estimators,
            num_leaves=best_model.num_leaves,
            min_child_samples=best_model.min_child_samples,
            subsample=best_model.subsample,
            colsample_bytree=best_model.colsample_bytree,
            reg_alpha=best_model.reg_alpha,
            reg_lambda=best_model.reg_lambda,
            scale_pos_weight=best_model.scale_pos_weight
        )
        
        model.fit(X_train_part, y_train_part, categorical_feature=cat_idx)
        train_f1s.append(f1_score(y_train_part, model.predict(X_train_part)))
        val_f1s.append(f1_score(y_val, model.predict(X_val)))
        test_f1s.append(f1_score(y_test, model.predict(X_test)))
    
    save_classifier_data('LightGBM', train_sizes * len(X), train_f1s, val_f1s, test_f1s)
    
    # Save regularization data
    reg_alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    reg_lambda_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    # Alpha regularization
    alpha_train_scores = []
    alpha_val_scores = []
    for alpha in reg_alpha_values:
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_model.learning_rate,
            max_depth=best_model.max_depth,
            n_estimators=best_model.n_estimators,
            num_leaves=best_model.num_leaves,
            min_child_samples=best_model.min_child_samples,
            subsample=best_model.subsample,
            colsample_bytree=best_model.colsample_bytree,
            reg_alpha=alpha,
            reg_lambda=best_model.reg_lambda,
            scale_pos_weight=best_model.scale_pos_weight
        )
        model.fit(X_train, y_train, categorical_feature=cat_idx)
        alpha_train_scores.append(f1_score(y_train, model.predict(X_train)))
        alpha_val_scores.append(f1_score(y_val, model.predict(X_val)))
    
    # Lambda regularization
    lambda_train_scores = []
    lambda_val_scores = []
    for lambda_val in reg_lambda_values:
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_model.learning_rate,
            max_depth=best_model.max_depth,
            n_estimators=best_model.n_estimators,
            num_leaves=best_model.num_leaves,
            min_child_samples=best_model.min_child_samples,
            subsample=best_model.subsample,
            colsample_bytree=best_model.colsample_bytree,
            reg_alpha=best_model.reg_alpha,
            reg_lambda=lambda_val,
            scale_pos_weight=best_model.scale_pos_weight
        )
        model.fit(X_train, y_train, categorical_feature=cat_idx)
        lambda_train_scores.append(f1_score(y_train, model.predict(X_train)))
        lambda_val_scores.append(f1_score(y_val, model.predict(X_val)))
    
    save_regularization_data('LightGBM_alpha', reg_alpha_values, alpha_train_scores, alpha_val_scores)
    save_regularization_data('LightGBM_lambda', reg_lambda_values, lambda_train_scores, lambda_val_scores)
    
    # Save error curves
    train_errors = [1 - score for score in train_f1s]
    val_errors = [1 - score for score in val_f1s]
    save_error_curves('LightGBM', train_sizes * len(X), train_errors, val_errors)
    
    # Save complexity data
    max_depths = [1, 3, 5, 7, 9, 11]
    n_estimators = [50, 100, 200, 300, 400, 500]
    
    # Depth complexity
    depth_train_scores = []
    depth_val_scores = []
    for depth in max_depths:
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_model.learning_rate,
            max_depth=depth,
            n_estimators=best_model.n_estimators,
            num_leaves=best_model.num_leaves,
            min_child_samples=best_model.min_child_samples,
            subsample=best_model.subsample,
            colsample_bytree=best_model.colsample_bytree,
            reg_alpha=best_model.reg_alpha,
            reg_lambda=best_model.reg_lambda,
            scale_pos_weight=best_model.scale_pos_weight
        )
        model.fit(X_train, y_train, categorical_feature=cat_idx)
        depth_train_scores.append(f1_score(y_train, model.predict(X_train)))
        depth_val_scores.append(f1_score(y_val, model.predict(X_val)))
    
    # Estimators complexity
    est_train_scores = []
    est_val_scores = []
    for n_est in n_estimators:
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_model.learning_rate,
            max_depth=best_model.max_depth,
            n_estimators=n_est,
            num_leaves=best_model.num_leaves,
            min_child_samples=best_model.min_child_samples,
            subsample=best_model.subsample,
            colsample_bytree=best_model.colsample_bytree,
            reg_alpha=best_model.reg_alpha,
            reg_lambda=best_model.reg_lambda,
            scale_pos_weight=best_model.scale_pos_weight
        )
        model.fit(X_train, y_train, categorical_feature=cat_idx)
        est_train_scores.append(f1_score(y_train, model.predict(X_train)))
        est_val_scores.append(f1_score(y_val, model.predict(X_val)))
    
    save_complexity_data('LightGBM_depth', max_depths, depth_train_scores, depth_val_scores)
    save_complexity_data('LightGBM_estimators', n_estimators, est_train_scores, est_val_scores)
    
    return best_model, val_metrics, test_metrics, cat_idx, best_thresh

def plot_metrics_comparison(metrics_with, metrics_without):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    with_scores = [metrics_with[m] for m in metrics]
    without_scores = [metrics_without[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, with_scores, width, label='With Weight')
    plt.bar(x + width/2, without_scores, width, label='Without Weight')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('LightGBM Performance: With vs Without Weight')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_f1_learning_curve(X, y, best_params, test_set, cat_idx, label, threshold=0.5):
    train_sizes = np.linspace(0.1, 0.9, 10)
    train_f1s, val_f1s, test_f1s = [], [], []
    X_test, y_test = test_set
    for frac in train_sizes:
        X_train_part, _, y_train_part, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=42)
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_part, y_train_part, test_size=0.2, stratify=y_train_part, random_state=42)
        model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            class_weight=None,
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            num_leaves=best_params['num_leaves'],
            min_child_samples=best_params['min_child_samples'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            scale_pos_weight=best_params['scale_pos_weight']
        )
        model.fit(X_train_part, y_train_part, categorical_feature=cat_idx)
        # Training F1
        train_probs = model.predict_proba(X_train_part)[:, 1]
        train_preds = (train_probs > threshold).astype(int)
        train_f1s.append(f1_score(y_train_part, train_preds, zero_division=0))
        # Validation F1
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > threshold).astype(int)
        val_f1s.append(f1_score(y_val, val_preds, zero_division=0))
        # Test F1
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs > threshold).astype(int)
        test_f1s.append(f1_score(y_test, test_preds, zero_division=0))
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * len(X), train_f1s, marker='o', label='Training F1')
    plt.plot(train_sizes * len(X), val_f1s, marker='o', label='Validation F1')
    plt.plot(train_sizes * len(X), test_f1s, marker='o', label='Test F1')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves (F1) - LightGBM ({label})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # With Weight
    print("Loading dataset with weight...")
    model_with_weight, val_metrics_with, test_metrics_with, cat_idx_with, best_thresh_with = run_lgbm_experiment('diabetic_with_weight.csv', 'With Weight')
    # Without Weight
    print("\nLoading dataset without weight...")
    model_without_weight, val_metrics_without, test_metrics_without, cat_idx_without, best_thresh_without = run_lgbm_experiment('diabetic_without_weight.csv', 'Without Weight')
    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: LightGBM")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'With Weight':<15} {'Without Weight':<15} {'Difference':<12}")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        with_score = test_metrics_with[metric]
        without_score = test_metrics_without[metric]
        diff = with_score - without_score
        print(f"{metric:<12} {with_score:<15.3f} {without_score:<15.3f} {diff:<12.3f}")
    # Plot metrics comparison
    plot_metrics_comparison(test_metrics_with, test_metrics_without)
    # Learning curves
    df_with = pd.read_csv('diabetic_with_weight.csv')
    X_with, y_with, cat_idx_with, _ = preprocess_data(df_with)
    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with, y_with, test_size=0.2, stratify=y_with, random_state=42)
    plot_f1_learning_curve(X_train_with, y_train_with, model_with_weight.get_params(), (X_test_with, y_test_with), cat_idx_with, 'With Weight', threshold=best_thresh_with)
    df_without = pd.read_csv('diabetic_without_weight.csv')
    X_without, y_without, cat_idx_without, _ = preprocess_data(df_without)
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without, y_without, test_size=0.2, stratify=y_without, random_state=42)
    plot_f1_learning_curve(X_train_without, y_train_without, model_without_weight.get_params(), (X_test_without, y_test_without), cat_idx_without, 'Without Weight', threshold=best_thresh_without)

if __name__ == "__main__":
    main()
