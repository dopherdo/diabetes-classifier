import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    """
    Preprocess the data by handling categorical variables and creating target
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Create binary target variable
    df['early_readmission'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])
    
    # Separate features and target
    X = df.drop(columns=['early_readmission'])
    y = df['early_readmission']
    
    # Handle categorical variables with Label Encoding
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Create label encoders for categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Handle missing values by treating them as a separate category
        X[col] = X[col].fillna('Missing')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Convert to numpy array and handle any remaining missing values
    X = X.fillna(0)  # Fill any remaining NaN with 0
    
    return X.values, y.values, label_encoders, X.columns.tolist()

def run_logistic_regression_experiment(csv_file, label=''):
    """
    Run Logistic Regression experiment on the given CSV file
    """
    print(f"\n{'='*50}")
    print(f"Running Logistic Regression Experiment: {label}")
    print(f"{'='*50}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")
    
    # Preprocess data
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
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Validation set size: {X_val_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Test different regularization strengths
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []
    
    print(f"\nTesting regularization strengths (C values): {C_values}")
    
    for C in C_values:
        # Train Logistic Regression
        lr = LogisticRegression(
            C=C,
            max_iter=1000,  # Increase max iterations for convergence
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        lr.fit(X_train_scaled, y_train)
        
        # Validate
        y_val_pred = lr.predict(X_val_scaled)
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:, 1])
        }
        
        # Test
        y_test_pred = lr.predict(X_test_scaled)
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        results.append({
            'C': C,
            'validation': val_metrics,
            'test': test_metrics,
            'model': lr,
            'coefficients': lr.coef_[0]
        })
        
        print(f"C={C:.3f}: Val F1={val_metrics['f1']:.3f}, Test F1={test_metrics['f1']:.3f}")
    
    # Find best C based on validation F1 score
    best_result = max(results, key=lambda x: x['validation']['f1'])
    best_C = best_result['C']
    
    print(f"\nBest C: {best_C:.3f}")
    print(f"Best validation F1: {best_result['validation']['f1']:.3f}")
    
    # Print detailed results for best model
    print(f"\nDetailed Results for C={best_C:.3f}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        val_score = best_result['validation'][metric]
        test_score = best_result['test'][metric]
        print(f"{metric.capitalize():>10}: Val={val_score:.3f}, Test={test_score:.3f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(best_result['test']['confusion_matrix'])
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': best_result['coefficients']
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 Most Important Features (by absolute coefficient):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<20}: {row['coefficient']:.4f}")
    
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
    plt.title(f'Logistic Regression Performance ({label})')
    plt.xticks(x, [m.capitalize() for m in metrics], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Feature importance
    plt.subplot(2, 3, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['coefficient'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 Feature Coefficients')
    plt.gca().invert_yaxis()
    
    # Confusion Matrix
    plt.subplot(2, 3, 3)
    cm = best_result['test']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Learning curves (validation performance across C values)
    plt.subplot(2, 3, 4)
    C_vals = [r['C'] for r in results]
    f1_vals = [r['validation']['f1'] for r in results]
    plt.semilogx(C_vals, f1_vals, 'o-', alpha=0.7)
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Validation F1 Score')
    plt.title('F1 vs Regularization Strength')
    plt.grid(True)
    
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
    
    return results, best_C, feature_importance

def main():
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        results_with_weight, best_C_with, features_with = run_logistic_regression_experiment(
            'diabetic_with_weight.csv', 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        results_without_weight, best_C_without, features_without = run_logistic_regression_experiment(
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
    
    print(f"\nBest C values:")
    print(f"With Weight: {best_C_with:.3f}")
    print(f"Without Weight: {best_C_without:.3f}")

    # Create comparison visualization
    plt.figure(figsize=(12, 6))
    
    # Prepare data for comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    with_scores = [best_with['test'][m] for m in metrics]
    without_scores = [best_without['test'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(x - width/2, with_scores, width, label='With Weight', alpha=0.8)
    plt.bar(x + width/2, without_scores, width, label='Without Weight', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Logistic Regression Performance: With vs Without Weight')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for i, (with_val, without_val) in enumerate(zip(with_scores, without_scores)):
        plt.text(i - width/2, with_val + 0.01, f'{with_val:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, without_val + 0.01, f'{without_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 