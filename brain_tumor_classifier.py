import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
from pathlib import Path

class ImprovedBrainTumorClassifier:
    def __init__(self, data_dir, img_size=(300, 300), batch_size=32):  # Increased batch size
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.class_names = None
        self._setup_gpu()
    
    def _setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU DETECTED: {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU Warning: {e}")
        else:
            print("No GPU found - using CPU")
    
    def create_dataframe(self, data_path):
        print(f"Loading data from: {data_path}")
        filepaths = []
        labels = []
        
        classes = [d for d in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, d))]
        
        for cls in classes:
            cls_path = os.path.join(data_path, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(cls_path, fname))
                    labels.append(cls)
        
        df = pd.DataFrame({'filepaths': filepaths, 'label': labels})
        print(f"Loaded {len(df)} images from {len(classes)} classes")
        
        # Check class distribution
        print("Class distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def create_train_val_split(self, test_size=0.2):
        """Create train/validation split from training data"""
        train_path = self.data_dir / 'Training'
        self.train_df = self.create_dataframe(train_path)
        
        # Split training data into train and validation
        train_df, val_df = train_test_split(
            self.train_df, 
            test_size=test_size, 
            stratify=self.train_df['label'],
            random_state=42
        )
        
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        
        print(f"Training samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
    
    def create_advanced_data_generators(self):
        print("Creating advanced data generators...")
        
        # Enhanced data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,  # Increased
            width_shift_range=0.3,  # Increased
            height_shift_range=0.3,  # Increased
            horizontal_flip=True,
            vertical_flip=True,  # Added
            zoom_range=0.3,  # Increased
            shear_range=0.2,  # Added
            brightness_range=[0.8, 1.2],  # Added
            fill_mode='nearest',
            channel_shift_range=0.2  # Added
            # Add more advanced augmentations
            augmentation_layers = tf.keras.Sequential([
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
])
        )
        
        # For validation and testing
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load test data
        test_path = self.data_dir / 'Testing'
        self.test_df = self.create_dataframe(test_path)
        
        print(f"Test samples: {len(self.test_df)}")
        print(f"Classes: {self.train_df['label'].unique().tolist()}")
    
    def create_generator(self, dataframe, generator, shuffle=False):
        return generator.flow_from_dataframe(
            dataframe,
            x_col='filepaths',
            y_col='label',
            target_size=self.img_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def build_improved_model(self):
        print("Building improved model...")
        
        # Load pre-trained EfficientNetB0 with better initialization
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3),
            pooling='avg'  # Use average pooling directly
        )
        
        # Fine-tuning: Unfreeze last layers gradually
        base_model.trainable = False  # Start with frozen base
        
        # Build improved model architecture
        self.model = Sequential([
            base_model,
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(self.train_gen.class_indices), activation='softmax')
        ])
        
        self.class_names = list(self.train_gen.class_indices.keys())
        print(f"Model built for {len(self.class_names)} classes: {self.class_names}")
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalance"""
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.train_df['label']),
            y=self.train_df['label']
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("Class weights:", class_weight_dict)
        return class_weight_dict
    
    def train_improved(self, epochs=30):
        """Improved training strategy"""
        print("Starting improved training...")
        
        # Create generators
        self.train_gen = self.create_generator(self.train_df, self.train_datagen, shuffle=True)
        self.val_gen = self.create_generator(self.val_df, self.test_datagen, shuffle=False)
        
        # Build model
        self.build_improved_model()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights()
        
        # Enhanced optimizer with learning rate scheduling
        optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        print(f"Training for {epochs} epochs...")
        history = self.model.fit(
            self.train_gen,
            epochs=epochs,
            validation_data=self.val_gen,
            callbacks=callbacks,
            class_weight=class_weights,  # Use class weights
            verbose=1
        )
        
        # Load best model
        self.model.load_weights('best_model.h5')
        
        return history
    
    def unfreeze_and_finetune(self, epochs=10):
        """Fine-tune the model by unfreezing base layers"""
        print("Starting fine-tuning...")
        
        # Unfreeze base model layers
        self.model.layers[0].trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.model.layers[0].layers) // 2
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),  # Very low learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning for {epochs} epochs...")
        history = self.model.fit(
            self.train_gen,
            epochs=epochs,
            validation_data=self.val_gen,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Enhanced evaluation"""
        print("\nEvaluating model...")
        
        test_gen = self.create_generator(self.test_df, self.test_datagen, shuffle=False)
        
        # Get predictions
        predictions = self.model.predict(test_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        # Test accuracy
        test_loss, test_accuracy = self.model.evaluate(test_gen, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Classification Report
        self.print_classification_report(y_true, y_pred)
        
        # Sample predictions
        self.show_sample_predictions(test_gen)
        
        return test_accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_improved.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, y_true, y_pred):
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
    
    def show_sample_predictions(self, test_gen, num_samples=6):
        print(f"\nShowing {num_samples} sample predictions...")
        
        test_images, test_labels = next(test_gen)
        predictions = self.model.predict(test_images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(test_images))):
            axes[i].imshow(test_images[i])
            true_label = self.class_names[true_classes[i]]
            pred_label = self.class_names[predicted_classes[i]]
            confidence = np.max(predictions[i])
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions_improved.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_improved_pipeline(self):
        """Run the complete improved pipeline"""
        print("STARTING IMPROVED BRAIN TUMOR CLASSIFICATION PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Create train/validation split
            self.create_train_val_split()
            
            # Step 2: Create data generators
            self.create_advanced_data_generators()
            
            # Step 3: Initial training
            print("\n--- PHASE 1: Initial Training ---")
            history1 = self.train_improved(epochs=30)
            
            # Step 4: Fine-tuning
            print("\n--- PHASE 2: Fine-tuning ---")
            history2 = self.unfreeze_and_finetune(epochs=10)
            
            # Step 5: Evaluation
            print("\n--- PHASE 3: Evaluation ---")
            accuracy = self.evaluate_model()
            
            # Step 6: Save model
            self.model.save('improved_brain_tumor_model.h5')
            
            print(f"\nIMPROVED PIPELINE COMPLETED! Final Accuracy: {accuracy:.4f}")
            print("="*60)
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

# MAIN EXECUTION
if __name__ == "__main__":
    DATA_PATH = r"D:\Suba Projects\P1-Brain-Tumor-Classification-EfficientNetB3\DataSet Brain Tumor"
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at: {DATA_PATH}")
        print("Please download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    else:
        # Run improved classifier
        classifier = ImprovedBrainTumorClassifier(data_dir=DATA_PATH)
        classifier.run_improved_pipeline()