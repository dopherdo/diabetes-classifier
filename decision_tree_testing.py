import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Fill remaining missing values
    X = X.fillna(0)
    
    return X.values, y.values, label_encoders, X.columns.tolist()

def run_gradient_boosting_experiment(csv_file, label=''):
    """Run Gradient Boosting experiment"""
    print(f"\n{'='*50}")
    print(f"Running Gradient Boosting Experiment: {label}")
    print(f"{'='*50}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")
    
    # Preprocess
    X, y, encoders, feature_names = preprocess_data(df)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Hyperparameter tuning for Gradient Boosting
    param_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 7},
    ]
    
    results = []
    best_f1 = 0
    best_params = {}
    
    print(f"\nTesting Gradient Boosting parameters...")
    
    for params in param_combinations:
        # Train Gradient Boosting with class balancing
        gb = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=42,
            subsample=0.8,  # Helps with overfitting
            min_samples_split=10,  # Prevents overfitting
            min_samples_leaf=5
        )
        
        # Use sample weights to handle class imbalance
        sample_weights = np.where(y_train == 1, 
                                len(y_train) / (2 * np.sum(y_train)), 
                                len(y_train) / (2 * np.sum(y_train == 0)))
        
        gb.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Validate
        y_val_pred = gb.predict(X_val)
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1])
        }
        
        # Test
        y_test_pred = gb.predict(X_test)
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        result = {
            'params': params,
            'validation': val_metrics,
            'test': test_metrics,
            'model': gb
        }
        results.append(result)
        
        # Track best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_params = params
            best_model = gb
        
        print(f"Params: {params}")
        print(f"  Val F1={val_metrics['f1']:.3f}, Test F1={test_metrics['f1']:.3f}")
        print(f"  Val Recall={val_metrics['recall']:.3f}, Test Recall={test_metrics['recall']:.3f}")
    
    # Find best result
    best_result = next(r for r in results if r['params'] == best_params)
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best validation F1: {best_result['validation']['f1']:.3f}")
    
    # Detailed results for best model
    print(f"\nDetailed Results for Best Model:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        val_score = best_result['validation'][metric]
        test_score = best_result['test'][metric]
        print(f"{metric.capitalize():>10}: Val={val_score:.3f}, Test={test_score:.3f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(best_result['test']['confusion_matrix'])
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<20}: {row['importance']:.4f}")
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Performance comparison
    plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    val_scores = [best_result['validation'][m] for m in metrics]
    test_scores = [best_result['test'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Gradient Boosting Performance ({label})')
    plt.xticks(x, [m.capitalize() for m in metrics], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Feature importance
    plt.subplot(2, 3, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    
    # Confusion Matrix
    plt.subplot(2, 3, 3)
    cm = best_result['test']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Learning curves (validation performance across parameters)
    plt.subplot(2, 3, 4)
    n_estimators_vals = [r['params']['n_estimators'] for r in results]
    f1_vals = [r['validation']['f1'] for r in results]
    plt.scatter(n_estimators_vals, f1_vals, alpha=0.7)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Validation F1 Score')
    plt.title('F1 vs Number of Estimators')
    
    # Precision-Recall tradeoff
    plt.subplot(2, 3, 5)
    precisions = [r['test']['precision'] for r in results]
    recalls = [r['test']['recall'] for r in results]
    plt.scatter(recalls, precisions, alpha=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Tradeoff')
    
    plt.tight_layout()
    plt.show()
    
    return results, best_params, feature_importance

# Run experiments on both datasets
def main():
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        results_with_weight, best_params_with, features_with = run_gradient_boosting_experiment(
            'diabetic_with_weight.csv', 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        results_without_weight, best_params_without, features_without = run_gradient_boosting_experiment(
            'diabetic_without_weight.csv', 
            'Without Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_without_weight.csv not found!")
        return
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: GRADIENT BOOSTING")
    print(f"{'='*60}")
    
    # Find best results
    best_with = max(results_with_weight, key=lambda x: x['validation']['f1'])
    best_without = max(results_without_weight, key=lambda x: x['validation']['f1'])
    
    print(f"{'Metric':<12} {'With Weight':<15} {'Without Weight':<15} {'Difference':<12}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        with_score = best_with['test'][metric]
        without_score = best_without['test'][metric]
        diff = with_score - without_score
        
        print(f"{metric.capitalize():<12} {with_score:<15.3f} {without_score:<15.3f} {diff:+.3f}")
    
    print(f"\nBest Parameters:")
    print(f"With Weight: {best_params_with}")
    print(f"Without Weight: {best_params_without}")

if __name__ == "__main__":
    main()
