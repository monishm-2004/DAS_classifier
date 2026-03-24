import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import scipy.io

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set paths
base_path = Path('/home/monish-m/Downloads/das/70-30 Data')
train_base_path = base_path / 'train'
test_base_path = base_path / 'test'
output_dir = base_path / 'cnn_results'
output_dir.mkdir(exist_ok=True)

# Activity labels mapping
ACTIVITY_LABELS = {
    '01_background': 0,
    '02_dig': 1,
    '03_knock': 2,
    '04_water': 3,
    '05_shake': 4,
    '06_walk': 5
}

ACTIVITY_NAMES = {v: k.split('_')[1].upper() for k, v in ACTIVITY_LABELS.items()}

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001


def build_cnn_model(input_shape):
    """Build CNN model optimized for raw .mat data (10000x12)"""
    model = models.Sequential([
        # Input: (10000, 12, 1)
        # Conv1 Layer
        layers.Conv2D(
            filters=16,
            kernel_size=(200, 3),
            strides=(50, 1),
            padding='same',
            activation='relu',
            input_shape=input_shape
        ),
        layers.MaxPooling2D(pool_size=(2, 1), padding='same'),
        
        # Conv2 Layer
        layers.Conv2D(
            filters=32,
            kernel_size=(50, 3),
            strides=(10, 1),
            padding='same',
            activation='relu'
        ),
        layers.MaxPooling2D(pool_size=(2, 1), padding='same'),
        
        # Conv3 Layer
        layers.Conv2D(
            filters=64,
            kernel_size=(20, 1),
            strides=(5, 1),
            padding='same',
            activation='relu'
        ),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(6, activation='softmax')  # 6 classes
    ])
    
    return model

class DataGenerator(keras.utils.Sequence):
    """Custom data generator for loading .mat files in batches"""
    
    def __init__(self, file_list, labels_list, batch_size=16, shuffle=True):
        """
        Args:
            file_list: List of file paths
            labels_list: List of labels corresponding to files
            batch_size: Batch size for training
            shuffle: Whether to shuffle data after each epoch
        """
        self.file_list = file_list
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_list))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, index):
        """Get batch at index"""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            try:
                # Load .mat file
                data = scipy.io.loadmat(self.file_list[idx])
                if 'data' in data:
                    raw_data = data['data'].astype(np.float32)  # Shape: (10000, 12)
                    
                    # Normalize per sample (z-score normalization)
                    raw_data = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-8)
                    
                    # Reshape for CNN: (10000, 12, 1)
                    raw_data = raw_data.reshape(raw_data.shape[0], raw_data.shape[1], 1)
                    
                    X_batch.append(raw_data)
                    y_batch.append(self.labels_list[idx])
            except Exception as e:
                print(f"Error loading {self.file_list[idx]}: {e}")
                continue
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
        
        # Convert labels to categorical (one-hot encoding)
        y_batch_cat = to_categorical(y_batch, 6)
        
        return X_batch, y_batch_cat
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data(max_files_per_activity=4000):
    """Collect file paths and labels for data generator
    
    Args:
        max_files_per_activity: Maximum number of files per activity folder
    
    Returns:
        train_files, train_labels, test_files, test_labels
    """
    print("\n" + "="*60)
    print("PREPARING DATA GENERATORS")
    print(f"Max files per activity: {max_files_per_activity}")
    print("="*60)
    
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []
    
    # Collect training file paths and labels
    print("\nCollecting TRAIN file paths...")
    for activity_folder, label in ACTIVITY_LABELS.items():
        train_activity_path = train_base_path / activity_folder
        
        if not train_activity_path.exists():
            print(f"  ⚠ {activity_folder} not found")
            continue
        
        mat_files = sorted(list(train_activity_path.glob('*.mat')))[:max_files_per_activity]
        print(f"  {activity_folder}: {len(mat_files)} files")
        
        for mat_file in mat_files:
            train_files.append(str(mat_file))
            train_labels.append(label)
    
    # Collect test file paths and labels
    print("\nCollecting TEST file paths...")
    for activity_folder, label in ACTIVITY_LABELS.items():
        test_activity_path = test_base_path / activity_folder
        
        if not test_activity_path.exists():
            print(f"  ⚠ {activity_folder} not found")
            continue
        
        mat_files = sorted(list(test_activity_path.glob('*.mat')))[:max_files_per_activity]
        print(f"  {activity_folder}: {len(mat_files)} files")
        
        for mat_file in mat_files:
            test_files.append(str(mat_file))
            test_labels.append(label)
    
    print(f"\nTotal TRAIN files: {len(train_files)}")
    print(f"Total TEST files: {len(test_files)}")
    
    # Print class distribution
    train_labels_arr = np.array(train_labels)
    test_labels_arr = np.array(test_labels)
    print("\nClass distribution:")
    print("  Train:", {ACTIVITY_NAMES[i]: np.sum(train_labels_arr == i) for i in range(6)})
    print("  Test:", {ACTIVITY_NAMES[i]: np.sum(test_labels_arr == i) for i in range(6)})
    
    return train_files, train_labels, test_files, test_labels

def plot_training_history(history):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()
    print("✓ Training history saved")

def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK'],
                yticklabels=['BG', 'DIG', 'KNOCK', 'WATER', 'SHAKE', 'WALK'])
    plt.title('Confusion Matrix - CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_cnn.png', dpi=150)
    plt.close()
    print("✓ Confusion matrix saved")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("CNN TRAINING PIPELINE (TensorFlow/Keras)")
    print("="*60)
    
    # Load file paths and labels
    train_files, train_labels, test_files, test_labels = load_data(max_files_per_activity=4000)
    
    # Create data generators
    print("\n" + "="*60)
    print("CREATING DATA GENERATORS")
    print("="*60)
    
    train_generator = DataGenerator(
        file_list=train_files,
        labels_list=train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_generator = DataGenerator(
        file_list=test_files,
        labels_list=test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Train batches per epoch: {len(train_generator)}")
    print(f"Test batches: {len(test_generator)}")
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    input_shape = (10000, 12, 1)
    model = build_cnn_model(input_shape)
    
    print(f"Input shape: {input_shape}")
    print("\nModel architecture:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model checkpoint callback (saves best model)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        output_dir / 'best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Training with data generators - NO early stopping, full 50 epochs
    print("\n" + "="*60)
    print(f"TRAINING FOR {EPOCHS} EPOCHS")
    print("="*60)
    
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=[model_checkpoint],
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Collect all test predictions
    y_pred_all = []
    y_true_all = np.array(test_labels, dtype=np.int32)
    
    print("Generating predictions on test set...")
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred_all = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nClassification Report:")
    class_names = [ACTIVITY_NAMES[i] for i in range(6)]
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    plot_confusion_matrix(y_true_all, y_pred_all)
    plot_training_history(history)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv(output_dir / 'cnn_metrics.csv', index=False)
    print("✓ Metrics saved to cnn_metrics.csv")
    
    # Save model (SavedModel format)
    model.save(output_dir / 'cnn_model.keras')
    print(f"✓ Model saved to {output_dir / 'cnn_model.keras'}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print("✓ Training history saved")
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return model, accuracy, precision, recall, f1

if __name__ == '__main__':
    model, accuracy, precision, recall, f1 = main()
