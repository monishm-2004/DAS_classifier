import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set paths
base_path = Path('/home/monish-m/Downloads/das/70-30 Data/train')
dataset_path = base_path / 'features_dataset.csv'
output_dir = base_path / 'classifier_results'
output_dir.mkdir(exist_ok=True)

def load_and_prepare_data(test_size=0.2, random_state=42):
    """Load data and split into train/test sets"""
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Prepare features and labels
    X = df.drop(['label', 'activity'], axis=1)
    y = df['label']
    activity_names = df['activity']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {sorted(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK']))
    
    return model, y_pred, y_pred_proba, acc

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting classifier"""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING CLASSIFIER")
    print("="*60)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK']))
    
    return model, y_pred, y_pred_proba, acc

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM classifier"""
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*60)
    
    model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK']))
    
    return model, y_pred, y_pred_proba, acc

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression classifier"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK']))
    
    return model, y_pred, y_pred_proba, acc

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK'],
                yticklabels=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Plot and save feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Features - {model_name}')
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("DAS ACTIVITY CLASSIFIER TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data()
    
    # Train classifiers
    models_results = {}
    
    # Random Forest
    rf_model, rf_pred, rf_proba, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    models_results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_acc,
        'predictions': rf_pred,
        'probabilities': rf_proba
    }
    plot_confusion_matrix(y_test, rf_pred, 'Random Forest')
    plot_feature_importance(rf_model, feature_names, 'Random Forest')
    
    # Gradient Boosting
    gb_model, gb_pred, gb_proba, gb_acc = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models_results['Gradient Boosting'] = {
        'model': gb_model,
        'accuracy': gb_acc,
        'predictions': gb_pred,
        'probabilities': gb_proba
    }
    plot_confusion_matrix(y_test, gb_pred, 'Gradient Boosting')
    plot_feature_importance(gb_model, feature_names, 'Gradient Boosting')
    
    # SVM
    svm_model, svm_pred, svm_proba, svm_acc = train_svm(X_train, y_train, X_test, y_test)
    models_results['SVM'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'predictions': svm_pred,
        'probabilities': svm_proba
    }
    plot_confusion_matrix(y_test, svm_pred, 'SVM')
    
    # Logistic Regression
    lr_model, lr_pred, lr_proba, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    models_results['Logistic Regression'] = {
        'model': lr_model,
        'accuracy': lr_acc,
        'predictions': lr_pred,
        'probabilities': lr_proba
    }
    plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression')
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison = []
    for model_name, results in models_results.items():
        acc = results['accuracy']
        y_pred = results['predictions']
        
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        comparison.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print("\n✓ Comparison saved to: model_comparison.csv")
    
    # Find best model
    best_model_name = max(models_results, key=lambda x: models_results[x]['accuracy'])
    best_model = models_results[best_model_name]['model']
    best_accuracy = models_results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Save best model
    model_save_path = output_dir / f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'model_name': best_model_name,
            'accuracy': best_accuracy
        }, f)
    print(f"✓ Best model saved to: {model_save_path}")
    
    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
