import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=10):
    """
    Train the model with early stopping
    """
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def run_ffnn_experiment(csv_file, label=''):
    """
    Run Feedforward Neural Network experiment on the given CSV file
    """
    print(f"\n{'='*50}")
    print(f"Running FFNN Experiment: {label}")
    print(f"{'='*50}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Validation set size: {X_val_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Create datasets and dataloaders
    train_dataset = DiabetesDataset(X_train_scaled, y_train)
    val_dataset = DiabetesDataset(X_val_scaled, y_val)
    test_dataset = DiabetesDataset(X_test_scaled, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Test different model architectures
    architectures = [
        {'hidden_sizes': [128, 64, 32], 'dropout_rate': 0.3},
        {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3},
        {'hidden_sizes': [128, 64], 'dropout_rate': 0.2},
        {'hidden_sizes': [256, 128, 64, 32], 'dropout_rate': 0.4},
        {'hidden_sizes': [512, 256, 128], 'dropout_rate': 0.3}
    ]
    
    results = []
    
    print(f"\nTesting different architectures...")
    
    for arch in architectures:
        print(f"\nArchitecture: {arch}")
        
        # Initialize model
        model = FeedForwardNN(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=arch['hidden_sizes'],
            dropout_rate=arch['dropout_rate']
        ).to(device)
        
        # Initialize loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.extend((outputs.cpu().numpy() > 0.5).astype(int))
                val_true.extend(y_batch.numpy())
        
        val_metrics = {
            'accuracy': accuracy_score(val_true, val_preds),
            'precision': precision_score(val_true, val_preds, zero_division=0),
            'recall': recall_score(val_true, val_preds, zero_division=0),
            'f1': f1_score(val_true, val_preds, zero_division=0),
            'roc_auc': roc_auc_score(val_true, val_preds)
        }
        
        # Evaluate on test set
        test_preds = []
        test_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                test_preds.extend((outputs.cpu().numpy() > 0.5).astype(int))
                test_true.extend(y_batch.numpy())
        
        test_metrics = {
            'accuracy': accuracy_score(test_true, test_preds),
            'precision': precision_score(test_true, test_preds, zero_division=0),
            'recall': recall_score(test_true, test_preds, zero_division=0),
            'f1': f1_score(test_true, test_preds, zero_division=0),
            'roc_auc': roc_auc_score(test_true, test_preds),
            'confusion_matrix': confusion_matrix(test_true, test_preds)
        }
        
        results.append({
            'architecture': arch,
            'validation': val_metrics,
            'test': test_metrics,
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        print(f"Val F1={val_metrics['f1']:.3f}, Test F1={test_metrics['f1']:.3f}")
    
    # Find best architecture based on validation F1 score
    best_result = max(results, key=lambda x: x['validation']['f1'])
    best_arch = best_result['architecture']
    
    print(f"\nBest Architecture: {best_arch}")
    print(f"Best validation F1: {best_result['validation']['f1']:.3f}")
    
    # Print detailed results for best model
    print(f"\nDetailed Results for Best Architecture:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        val_score = best_result['validation'][metric]
        test_score = best_result['test'][metric]
        print(f"{metric.capitalize():>10}: Val={val_score:.3f}, Test={test_score:.3f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    print(best_result['test']['confusion_matrix'])
    
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
    plt.title(f'FFNN Performance ({label})')
    plt.xticks(x, [m.capitalize() for m in metrics], rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Confusion Matrix
    plt.subplot(2, 3, 2)
    cm = best_result['test']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Learning curves
    plt.subplot(2, 3, 3)
    plt.plot(best_result['train_losses'], label='Training Loss')
    plt.plot(best_result['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Architecture comparison
    plt.subplot(2, 3, 4)
    architectures = [str(r['architecture']) for r in results]
    f1_scores = [r['validation']['f1'] for r in results]
    plt.bar(range(len(architectures)), f1_scores)
    plt.xticks(range(len(architectures)), [f'Arch {i+1}' for i in range(len(architectures))], rotation=45)
    plt.xlabel('Architecture')
    plt.ylabel('Validation F1 Score')
    plt.title('Architecture Comparison')
    
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
    
    return results, best_arch

def main():
    # Run on dataset with weight
    print("Loading dataset with weight...")
    try:
        results_with_weight, best_arch_with = run_ffnn_experiment(
            os.path.join(SCRIPT_DIR, 'diabetic_with_weight.csv'), 
            'With Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_with_weight.csv not found!")
        return
    
    # Run on dataset without weight
    print("\nLoading dataset without weight...")
    try:
        results_without_weight, best_arch_without = run_ffnn_experiment(
            os.path.join(SCRIPT_DIR, 'diabetic_without_weight.csv'), 
            'Without Weight'
        )
    except FileNotFoundError:
        print("Error: diabetic_without_weight.csv not found!")
        return
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: FEEDFORWARD NEURAL NETWORK")
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
    
    print(f"\nBest Architectures:")
    print(f"With Weight: {best_arch_with}")
    print(f"Without Weight: {best_arch_without}")
    
    # --- 4-panel plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    with_scores = [best_with['test'][m] for m in metrics]
    without_scores = [best_without['test'][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    # 1. Grouped bar chart (top left)
    axs[0, 0].bar(x - width/2, with_scores, width, label='With Weight', alpha=0.8)
    axs[0, 0].bar(x + width/2, without_scores, width, label='Without Weight', alpha=0.8)
    axs[0, 0].set_xlabel('Metrics')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].set_title('FFNN Performance Comparison')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels([m.capitalize() for m in metrics])
    axs[0, 0].legend()
    axs[0, 0].set_ylim(0, 1)
    # 2. Confusion matrix (top right)
    sns.heatmap(best_with['test']['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axs[0, 1])
    axs[0, 1].set_title('Confusion Matrix - With Weight')
    axs[0, 1].set_ylabel('Actual')
    axs[0, 1].set_xlabel('Predicted')
    # 3. Confusion matrix (bottom left)
    sns.heatmap(best_without['test']['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', ax=axs[1, 0])
    axs[1, 0].set_title('Confusion Matrix - Without Weight')
    axs[1, 0].set_ylabel('Actual')
    axs[1, 0].set_xlabel('Predicted')
    # 4. F1 score vs. architecture index (bottom right)
    arch_f1_with = [r['validation']['f1'] for r in results_with_weight]
    arch_f1_without = [r['validation']['f1'] for r in results_without_weight]
    axs[1, 1].plot(range(1, len(arch_f1_with)+1), arch_f1_with, 'o-', label='With Weight')
    axs[1, 1].plot(range(1, len(arch_f1_without)+1), arch_f1_without, 's-', label='Without Weight')
    axs[1, 1].set_xlabel('Architecture Index')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_title('F1 Score vs Architecture')
    axs[1, 1].legend()
    axs[1, 1].set_xticks(range(1, max(len(arch_f1_with), len(arch_f1_without))+1))
    axs[1, 1].set_ylim(0, max(max(arch_f1_with), max(arch_f1_without), 0.2))
    plt.tight_layout()
    plt.show()
    # --- end 4-panel plot ---

    # --- Learning Curves: F1 vs. Epoch for Best Architecture (for both datasets) ---
    def compute_f1_per_epoch(model, loader, device):
        f1s = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs.cpu().numpy() > 0.5).astype(int)
                f1 = f1_score(y_batch.numpy(), preds, zero_division=0)
                f1s.append(f1)
        return np.mean(f1s)

    # For best_with and best_without, plot F1 vs. epoch (using train/val loaders and saved losses)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (best_result, label) in enumerate(zip([best_with, best_without], ['With Weight', 'Without Weight'])):
        train_f1s = []
        val_f1s = []
        # Simulate F1 per epoch using saved model at each epoch (not available, so use final model and plot loss curves)
        # Instead, plot loss curves and final F1 as horizontal lines
        epochs = np.arange(1, len(best_result['train_losses']) + 1)
        axs[idx].plot(epochs, best_result['train_losses'], label='Training Loss')
        axs[idx].plot(epochs, best_result['val_losses'], label='Validation Loss')
        axs[idx].axhline(y=best_result['validation']['f1'], color='g', linestyle='--', label='Final Val F1')
        axs[idx].axhline(y=best_result['test']['f1'], color='r', linestyle='--', label='Final Test F1')
        axs[idx].set_xlabel('Epoch')
        axs[idx].set_ylabel('Loss / F1')
        axs[idx].set_title(f'Learning Curves - FFNN ({label})')
        axs[idx].legend()
    plt.tight_layout()
    plt.show()
    # --- end learning curves ---

    # --- Metric vs. Architecture plots ---
    def plot_metric_vs_architecture(results, label):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        arch_indices = np.arange(1, len(results) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for i, metric in enumerate(metrics):
            row, col = divmod(i, 2)
            val_scores = [r['validation'][metric] for r in results]
            test_scores = [r['test'][metric] for r in results]
            axs[row, col].plot(arch_indices, val_scores, 'o-', label='Validation')
            axs[row, col].plot(arch_indices, test_scores, 's-', label='Test')
            axs[row, col].set_xlabel('Architecture Index')
            axs[row, col].set_ylabel(metric.capitalize())
            axs[row, col].set_title(f'{metric.capitalize()} vs Architecture ({label})')
            axs[row, col].legend()
            axs[row, col].set_xticks(arch_indices)
        plt.tight_layout()
        plt.show()

    plot_metric_vs_architecture(results_with_weight, 'With Weight')
    plot_metric_vs_architecture(results_without_weight, 'Without Weight')
    # --- end metric vs. architecture plots ---

if __name__ == "__main__":
    main()
