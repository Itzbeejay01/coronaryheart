import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# --- Directory Setup ---
base_dir = 'data/splits'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')  # only for evaluation, NOT for training

# --- Parameters ---
img_size = (224, 224)
batch_size = 32
epochs = 30

# --- Data Augmentation and Generators ---
train_datagen = ImageDataGenerator(
    preprocessing_function=efficientnet_preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print("Class distribution in training set:", np.bincount(train_generator.classes))
print("Class distribution in validation set:", np.bincount(validation_generator.classes))

# --- Load Pretrained EfficientNetB0 Model ---
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # Freeze base convolutional layers

# --- Build model ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Compute class weights to address imbalance ---
unique_classes = np.unique(train_generator.classes)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_generator.classes)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# --- Setup checkpoint path for weights only (resuming training) ---
checkpoint_dir = 'model_training/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'efficientnet_best_model.weights.h5')

# --- Load weights if checkpoint exists (resume training) ---
if os.path.exists(checkpoint_path):
    print("Loading weights from checkpoint to resume training...")
    model.load_weights(checkpoint_path)

# --- Callbacks ---
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- Train ---
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping]
)

# --- After training: Save full model in .keras format ---
final_model_path = 'models/efficientnet_final_model.keras'
model.save(final_model_path)
print(f"Full model saved after training at: {final_model_path}")

# --- Load best weights for evaluation ---
model.load_weights(checkpoint_path)

# --- Predict on test set ---
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Calculate metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)
specificity = recall_score(y_true, y_pred, pos_label=0)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print("\nTest Set Performance Metrics:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score   : {f1:.4f}")
print(f"AUC-ROC    : {auc:.4f}")

# --- Classification report and confusion matrix ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --- Plot ROC curve ---
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'EfficientNetB0 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show() 