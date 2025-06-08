import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    """
    Enhanced preprocessing with feature engineering for logistic regression
    """
    df = df.copy()
    
    # Create binary target variable
    df['early_readmission'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])
    
    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')
    
    # Handle missing values in categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna('Missing')
    
    # Handle missing values in numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if col != 'early_readmission':
            df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering for numerical columns
    if 'time_in_hospital' in df.columns:
        df['time_in_hospital_squared'] = df['time_in_hospital'] ** 2
    
    if 'num_lab_procedures' in df.columns:
        df['num_lab_procedures_squared'] = df['num_lab_procedures'] ** 2
    
    if 'num_medications' in df.columns:
        df['num_medications_squared'] = df['num_medications'] ** 2
    
    # Create interaction features
    if all(col in df.columns for col in ['time_in_hospital', 'num_lab_procedures']):
        df['time_lab_ratio'] = df['time_in_hospital'] / (df['num_lab_procedures'] + 1)
    
    if all(col in df.columns for col in ['num_medications', 'num_procedures']):
        df['med_proc_ratio'] = df['num_medications'] / (df['num_procedures'] + 1)
    
    # Separate features and target
    X = df.drop(columns=['early_readmission'])
    y = df['early_readmission']
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    return X, y, preprocessor

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

def run_logistic_regression_experiment(csv_file, label=''):
    """
    Comprehensive Logistic Regression experiment with cross-validation
    """
    print(f"\n{'='*50}")
    print(f"Running Logistic Regression Experiment: {label}")
    print(f"{'='*50}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")
    
    # Preprocess data
    X, y, preprocessor = preprocess_data(df)
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create pipeline with SMOTE
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear']
    }
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    print("\nTraining model with cross-validation...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nBest parameters: {best_params}")
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    # Print results
    print(f"\nTest Metrics:")
    for metric, score in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.capitalize()}: {score:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Get feature names after preprocessing
    feature_names = []
    for name, trans, cols in best_model.named_steps['preprocessor'].transformers_:
        if name == 'cat':
            feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    
    # Get coefficients
    coefficients = best_model.named_steps['classifier'].coef_[0]
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Feature Importance
    plt.subplot(2, 2, 3)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['coefficient'])
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Save learning curve data
    train_sizes = np.linspace(0.1, 0.9, 10)
    train_f1s = []
    val_f1s = []
    test_f1s = []
    
    for frac in train_sizes:
        X_train_part, _, y_train_part, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=42)
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_part, y_train_part, test_size=0.2, stratify=y_train_part, random_state=42)
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=best_params['classifier__C'],
                class_weight=best_params['classifier__class_weight'],
                penalty=best_params['classifier__penalty'],
                solver=best_params['classifier__solver'],
                max_iter=2000,
                random_state=42
            ))
        ])
        
        model.fit(X_train_part, y_train_part)
        train_f1s.append(f1_score(y_train_part, model.predict(X_train_part)))
        val_f1s.append(f1_score(y_val, model.predict(X_val)))
        test_f1s.append(f1_score(y_test, model.predict(X_test)))
    
    save_classifier_data('Logistic Regression', train_sizes * len(X), train_f1s, val_f1s, test_f1s)
    
    # Save regularization data
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores = []
    val_scores = []
    
    for C in C_values:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=C,
                class_weight=best_params['classifier__class_weight'],
                penalty=best_params['classifier__penalty'],
                solver=best_params['classifier__solver'],
                max_iter=2000,
                random_state=42
            ))
        ])
        
        model.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train)))
        val_scores.append(f1_score(y_val, model.predict(X_val)))
    
    save_regularization_data('Logistic Regression', C_values, train_scores, val_scores)
    
    # Save error curves
    train_errors = [1 - score for score in train_f1s]
    val_errors = [1 - score for score in val_f1s]
    save_error_curves('Logistic Regression', train_sizes * len(X), train_errors, val_errors)
    
    # Save complexity data (using C as complexity measure)
    save_complexity_data('Logistic Regression', C_values, train_scores, val_scores)
    
    return best_model, metrics

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
    plt.title('Logistic Regression Performance: With vs Without Weight')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curve(X, y, preprocessor, best_params, test_set, label='Without Weight'):
    from sklearn.base import clone
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_f1, val_f1, test_f1 = [], [], []
    X_test, y_test = test_set
    for frac in train_sizes:
        X_train_part, _, y_train_part, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=42)
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_part, y_train_part, test_size=0.2, stratify=y_train_part, random_state=42)
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=best_params['classifier__C'],
                class_weight=best_params['classifier__class_weight'],
                penalty=best_params['classifier__penalty'],
                solver=best_params['classifier__solver'],
                max_iter=2000,
                random_state=42
            ))
        ])
        model.fit(X_train_part, y_train_part)
        train_f1.append(f1_score(y_train_part, model.predict(X_train_part)))
        val_f1.append(f1_score(y_val, model.predict(X_val)))
        test_f1.append(f1_score(y_test, model.predict(X_test)))
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * len(X), train_f1, marker='o', label='Training F1')
    plt.plot(train_sizes * len(X), val_f1, marker='o', label='Validation F1')
    plt.plot(train_sizes * len(X), test_f1, marker='o', label='Test F1')
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - Logistic Regression ({label})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        model_with_weight, metrics_with = run_logistic_regression_experiment(
            'diabetic_with_weight.csv', 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        model_without_weight, metrics_without = run_logistic_regression_experiment(
            'diabetic_without_weight.csv', 
            'Without Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_without_weight.csv not found!")
        return
    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: LOGISTIC REGRESSION")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'With Weight':<15} {'Without Weight':<15} {'Difference':<12}")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        with_score = metrics_with[metric]
        without_score = metrics_without[metric]
        diff = with_score - without_score
        print(f"{metric:<12} {with_score:<15.3f} {without_score:<15.3f} {diff:<12.3f}")
    # Plot metrics comparison
    plot_metrics_comparison(metrics_with, metrics_without)
    # Learning curve for 'without weight' dataset
    df = pd.read_csv('diabetic_without_weight.csv')
    X, y, preprocessor = preprocess_data(df)
    # Use the best params from the last run (without weight)
    best_params = model_without_weight.named_steps['classifier'].get_params()
    # Get test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    plot_learning_curve(X_train, y_train, preprocessor, model_without_weight.get_params(), (X_test, y_test), label='Without Weight')

if __name__ == "__main__":
    main() 