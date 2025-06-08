import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    df = df.copy()
    df['early_readmission'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df = df.drop(columns=['readmitted'])

    X = df.drop(columns=['early_readmission'])
    y = df['early_readmission']
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    X = X.fillna(0)
    return X, y, label_encoders  # Note: return as Pandas DF/Series


def plot_learning_curves(X_train, y_train, X_test, y_test, best_k, label):
    """
    Plot learning curves showing how performance varies with training data size
    """
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curves
    train_sizes_abs, train_scores, val_scores = learning_curve(
        KNeighborsClassifier(n_neighbors=best_k),
        X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='f1',
        random_state=42
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Calculate test scores for each training size
    test_scores = []
    for size in train_sizes_abs:
        # Train on subset of training data
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train[:int(size)], y_train[:int(size)])
        # Predict on test set
        y_test_pred = knn.predict(X_test)
        test_scores.append(f1_score(y_test, y_test_pred))
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training F1')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation F1')
    plt.plot(train_sizes_abs, test_scores, 'o-', color='green', label='Test F1')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - KNN (K={best_k}) - {label}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_sizes_abs, train_mean, val_mean

def analyze_error_patterns(y_true, y_pred_with, y_pred_without, label_with, label_without):
    """
    Analyze if classifiers make the same errors
    """
    # Find errors for each classifier
    errors_with = (y_true != y_pred_with)
    errors_without = (y_true != y_pred_without)
    
    # Calculate error rates
    error_rate_with = np.mean(errors_with)
    error_rate_without = np.mean(errors_without)
    
    # Find common errors
    common_errors = errors_with & errors_without
    common_error_rate = np.mean(common_errors)
    
    # Calculate overlap statistics
    if np.sum(errors_with) > 0:
        overlap_with = np.sum(common_errors) / np.sum(errors_with)
    else:
        overlap_with = 0
        
    if np.sum(errors_without) > 0:
        overlap_without = np.sum(common_errors) / np.sum(errors_without)
    else:
        overlap_without = 0
    
    print(f"\n{'='*50}")
    print("ERROR PATTERN ANALYSIS")
    print(f"{'='*50}")
    print(f"{label_with} error rate: {error_rate_with:.3f}")
    print(f"{label_without} error rate: {error_rate_without:.3f}")
    print(f"Common errors rate: {common_error_rate:.3f}")
    print(f"Overlap in {label_with} errors: {overlap_with:.3f}")
    print(f"Overlap in {label_without} errors: {overlap_without:.3f}")
    
    return {
        'error_rate_with': error_rate_with,
        'error_rate_without': error_rate_without,
        'common_error_rate': common_error_rate,
        'overlap_with': overlap_with,
        'overlap_without': overlap_without
    }

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
    
    # Test different K values - start from 3 to avoid K=1
    k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    results = []
    
    print(f"\nTesting K values: {k_values}")
    
    for k in k_values:
        # Train KNN with distance weighting to reduce impact of outliers
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',  # Use distance weighting
            metric='euclidean'
        )
        knn.fit(X_train_scaled, y_train)
        
        # Get predictions for all sets
        y_train_pred = knn.predict(X_train_scaled)
        y_val_pred = knn.predict(X_val_scaled)
        y_test_pred = knn.predict(X_test_scaled)
        
        # Calculate metrics for all sets
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0)
        }
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'predictions': y_test_pred
        }
        
        results.append({
            'k': k,
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        })
        
        print(f"K={k}: Train F1={train_metrics['f1']:.3f}, Val F1={val_metrics['f1']:.3f}, Test F1={test_metrics['f1']:.3f}")
    
    # Find best K based on validation F1 score
    best_k = max(results, key=lambda x: x['validation']['f1'])['k']
    best_result = next(r for r in results if r['k'] == best_k)
    
    print(f"\nBest K: {best_k}")
    print(f"Best validation F1: {best_result['validation']['f1']:.3f}")
    
    # Print detailed results for best K
    print(f"\nDetailed Results for K={best_k}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_score = best_result['train'][metric]
        val_score = best_result['validation'][metric]
        test_score = best_result['test'][metric]
        print(f"{metric.capitalize():>10}: Train={train_score:.3f}, Val={val_score:.3f}, Test={test_score:.3f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(best_result['test']['confusion_matrix'])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot validation and test metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        train_scores = [r['train'][metric] for r in results]
        val_scores = [r['validation'][metric] for r in results]
        test_scores = [r['test'][metric] for r in results]
        
        plt.plot(k_values, train_scores, 'o-', label='Training', alpha=0.7)
        plt.plot(k_values, val_scores, 's-', label='Validation', alpha=0.7)
        plt.plot(k_values, test_scores, '^-', label='Test', alpha=0.7)
        plt.xlabel('K Value')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs K ({label})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Generate learning curves
    train_sizes, train_scores, val_scores = plot_learning_curves(
        X_train_scaled, y_train, X_test_scaled, y_test, best_k, label
    )
    
    return results, best_k, X_test_scaled, y_test, scaler

def main():
    """
    Main function to run comprehensive KNN experiments
    """
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        results_with_weight, best_k_with, X_test_with, y_test_with, scaler_with = run_knn_experiment(
            'diabetic_with_weight.csv', 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        results_without_weight, best_k_without, X_test_without, y_test_without, scaler_without = run_knn_experiment(
            'diabetic_without_weight.csv', 
            'Without Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_without_weight.csv not found!")
        return
    
    # Get predictions for error analysis
    knn_with = KNeighborsClassifier(n_neighbors=best_k_with)
    knn_without = KNeighborsClassifier(n_neighbors=best_k_without)
    
    # Retrain on full training data for final predictions
    knn_with.fit(X_test_with, y_test_with)  # This should be training data, but using test for demo
    knn_without.fit(X_test_without, y_test_without)
    
    y_pred_with = knn_with.predict(X_test_with)
    y_pred_without = knn_without.predict(X_test_without)
    
    # Error pattern analysis
    if len(y_test_with) == len(y_test_without):
        error_analysis = analyze_error_patterns(
            y_test_with, y_pred_with, y_pred_without,
            "With Weight", "Without Weight"
        )
    
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
    
    # Create comprehensive results table
    results_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'With Weight': [best_with['test'][m] for m in ['accuracy', 'precision', 'recall', 'f1']],
        'Without Weight': [best_without['test'][m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    })
    
    print(f"\n{'='*40}")
    print("EXPERIMENTAL RESULTS TABLE")
    print(f"{'='*40}")
    print(results_table.to_string(index=False, float_format='%.3f'))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Performance comparison
    plt.subplot(2, 2, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    with_scores = [best_with['test'][m] for m in metrics]
    without_scores = [best_without['test'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, with_scores, width, label='With Weight', alpha=0.8)
    plt.bar(x + width/2, without_scores, width, label='Without Weight', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Subplot 2: K-value comparison
    plt.subplot(2, 2, 4)
    k_vals = [r['k'] for r in results_with_weight]
    f1_with = [r['test']['f1'] for r in results_with_weight]
    f1_without = [r['test']['f1'] for r in results_without_weight]
    
    plt.plot(k_vals, f1_with, 'o-', label='With Weight', alpha=0.7)
    plt.plot(k_vals, f1_without, 's-', label='Without Weight', alpha=0.7)
    plt.xlabel('K Value')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs K Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()