import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.class_names = ['no_disease', 'disease']
    
    def load_inbreast(self, data_dir):
        """Load INbreast dataset"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, 'inbreast', class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    img = load_img(img_path, target_size=self.input_shape[:2])
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(self.class_names.index(class_name))
        
        return np.array(images), np.array(labels)
    
    def load_cbis_ddsm(self, data_dir):
        """Load CBIS-DDSM dataset"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, 'cbis-ddsm', class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    img = load_img(img_path, target_size=self.input_shape[:2])
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(self.class_names.index(class_name))
        
        return np.array(images), np.array(labels)
    
    def load_mri(self, data_dir):
        """Load breast MRI dataset"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, 'mri', class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    img = load_img(img_path, target_size=self.input_shape[:2])
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(self.class_names.index(class_name))
        
        return np.array(images), np.array(labels)
    
    def load_all_datasets(self, data_dir):
        """Load and combine all user-provided datasets for coronary heart disease"""
        inbreast_images, inbreast_labels = self.load_inbreast(data_dir)
        cbis_images, cbis_labels = self.load_cbis_ddsm(data_dir)
        mri_images, mri_labels = self.load_mri(data_dir)
        images = np.concatenate([inbreast_images, cbis_images, mri_images])
        labels = np.concatenate([inbreast_labels, cbis_labels, mri_labels])
        images = preprocess_input(images)
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.class_names))
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        return x_train, y_train, x_test, y_test
    
    def get_data_generator(self, augmentation=True):
        """Get data generator with augmentation"""
        if augmentation:
            return tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest'
            )
        return tf.keras.preprocessing.image.ImageDataGenerator() 