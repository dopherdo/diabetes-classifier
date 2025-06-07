import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
    return X.values, y.values, label_encoders

def run_knn_experiment(csv_file, label=''):
    """
    Run KNN experiment on the given CSV file
    """
    print(f"\n{'='*50}")
    print(f"Running KNN Experiment: {label}")
    print(f"{'='*50}")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['readmitted'].value_counts()}")
    
    # Preprocess data
    X, y, encoders = preprocess_data(df)
    
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
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Validation set size: {X_val_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Test different K values
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    results = []
    
    print(f"\nTesting K values: {k_values}")
    
    for k in k_values:
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        
        # Validate
        y_val_pred = knn.predict(X_val_scaled)
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0)
        }
        
        # Test
        y_test_pred = knn.predict(X_test_scaled)
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        results.append({
            'k': k,
            'validation': val_metrics,
            'test': test_metrics
        })
        
        print(f"K={k}: Val F1={val_metrics['f1']:.3f}, Test F1={test_metrics['f1']:.3f}")
    
    # Find best K based on validation F1 score
    best_k = max(results, key=lambda x: x['validation']['f1'])['k']
    best_result = next(r for r in results if r['k'] == best_k)
    
    print(f"\nBest K: {best_k}")
    print(f"Best validation F1: {best_result['validation']['f1']:.3f}")
    
    # Print detailed results for best K
    print(f"\nDetailed Results for K={best_k}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        val_score = best_result['validation'][metric]
        test_score = best_result['test'][metric]
        print(f"{metric.capitalize():>10}: Val={val_score:.3f}, Test={test_score:.3f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(best_result['test']['confusion_matrix'])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot validation and test metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        val_scores = [r['validation'][metric] for r in results]
        test_scores = [r['test'][metric] for r in results]
        
        plt.plot(k_values, val_scores, 'o-', label='Validation', alpha=0.7)
        plt.plot(k_values, test_scores, 's-', label='Test', alpha=0.7)
        plt.xlabel('K Value')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs K ({label})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results, best_k

# Run experiments on both datasets
def main():
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        results_with_weight, best_k_with = run_knn_experiment(
            'diabetic_with_weight.csv', 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        results_without_weight, best_k_without = run_knn_experiment(
            'diabetic_without_weight.csv', 
            'Without Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_without_weight.csv not found!")
        return
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    best_with = next(r for r in results_with_weight if r['k'] == best_k_with)
    best_without = next(r for r in results_without_weight if r['k'] == best_k_without)
    
    print(f"{'Metric':<12} {'With Weight':<15} {'Without Weight':<15} {'Difference':<12}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        with_score = best_with['test'][metric]
        without_score = best_without['test'][metric]
        diff = with_score - without_score
        
        print(f"{metric.capitalize():<12} {with_score:<15.3f} {without_score:<15.3f} {diff:+.3f}")
    
    print(f"\nBest K values: With Weight = {best_k_with}, Without Weight = {best_k_without}")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    with_scores = [best_with['test'][m] for m in metrics]
    without_scores = [best_without['test'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, with_scores, width, label='With Weight', alpha=0.8)
    plt.bar(x + width/2, without_scores, width, label='Without Weight', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Performance Comparison: With vs Without Weight')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    for i, (with_val, without_val) in enumerate(zip(with_scores, without_scores)):
        plt.text(i - width/2, with_val + 0.01, f'{with_val:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, without_val + 0.01, f'{without_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
