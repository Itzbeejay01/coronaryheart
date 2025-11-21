import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

class EnsembleModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        
        # Initialize individual models
        self.vgg16 = self._create_vgg16()
        self.resnet50 = self._create_resnet50()
        self.efficientnet = self._create_efficientnet()
        
        self.models = [self.vgg16, self.resnet50, self.efficientnet]
    
    def _create_vgg16(self):
        """Create and compile VGG16 model"""
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=self.input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
            
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def _create_resnet50(self):
        """Create and compile ResNet50 model"""
        base_model = ResNet50(weights='imagenet', include_top=False,
                            input_shape=self.input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
            
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def _create_efficientnet(self):
        """Create and compile EfficientNet model"""
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                  input_shape=self.input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
            
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """Train each model with standard data and save checkpoints"""
        checkpoint_paths = [
            'model_training/checkpoints/vgg16_checkpoint.keras',
            'model_training/checkpoints/resnet50_checkpoint.keras',
            'model_training/checkpoints/efficientnet_checkpoint.keras'
        ]
        for model, ckpt_path in zip(self.models, checkpoint_paths):
            checkpoint_cb = ModelCheckpoint(
                ckpt_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=False,
                verbose=1
            )
            model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[checkpoint_cb]
            )
    
    def predict(self, x):
        """Make predictions using ensemble averaging"""
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Average the predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate(self, x_test, y_test):
        """Evaluate the ensemble model"""
        predictions = self.predict(x_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        return accuracy 