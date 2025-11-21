import os
import shutil
import random

# Configurations
PROCESSED_DIR = 'data/train'  # Updated path to cropped_train
SPLIT_DIR = 'data/splits'
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def create_split_folders():
    """Create train and val directories."""
    for split in ['train', 'val']:
        split_path = os.path.join(SPLIT_DIR, split)
        os.makedirs(split_path, exist_ok=True)

def get_image_paths():
    """Get all image paths from the processed directory."""
    image_paths = []
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for file in files:
            if file.lower().endswith(VALID_IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file))
    return image_paths

def split_dataset(image_paths):
    """Split the dataset into train and val only."""
    random.shuffle(image_paths)
    total_images = len(image_paths)
    train_end = int(TRAIN_RATIO * total_images)
    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:]
    return train_images, val_images

def move_images(image_paths, split_type):
    """Move images to the corresponding split directory."""
    for image_path in image_paths:
        class_folder = os.path.basename(os.path.dirname(image_path))
        dest_dir = os.path.join(SPLIT_DIR, split_type, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, os.path.basename(image_path))
        shutil.move(image_path, dest_path)
        print(f"Moved: {image_path} -> {dest_path}")

def main():
    """Main function to perform the dataset split."""
    create_split_folders()
    image_paths = get_image_paths()
    train_images, val_images = split_dataset(image_paths)
    
    move_images(train_images, 'train')
    move_images(val_images, 'val')
    
    print("Dataset splitting completed successfully.")

if __name__ == "__main__":
    main()
