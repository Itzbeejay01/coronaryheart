import cv2
import numpy as np
import os
import logging
import multiprocessing as mp
from typing import List, Tuple

# Number of processes to use (3 cores as requested)
NUM_PROCESSES = 3

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ensure_directory_exists(directory: str):
    """Ensure the output directory exists."""
    os.makedirs(directory, exist_ok=True)

def load_image(image_path: str) -> np.ndarray:
    """Load image in grayscale mode."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"❌ ERROR: Could not load image {image_path}. Check the file path.")
        raise FileNotFoundError(f"Image not found: {image_path}")
    logging.info(f"✅ Image loaded successfully: {image_path} - Size: {image.shape[1]}x{image.shape[0]}")
    return image

def resize_image(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Resize image to target size using high-quality interpolation."""
    logging.info("🔄 Resizing image to 224x224...")
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to range [0, 255] for consistent processing."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_clahe(image: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    logging.info("🔄 Applying CLAHE...")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def save_image(image: np.ndarray, output_path: str):
    """Save the processed image."""
    success = cv2.imwrite(output_path, image)
    if not success:
        logging.error(f"❌ ERROR: Failed to save image at {output_path}")
        raise IOError(f"Failed to save image: {output_path}")
    logging.info(f"✅ Image saved successfully: {output_path}")

def process_image(image_path: str, output_path: str):
    """Complete image preprocessing pipeline."""
    setup_logging()
    ensure_directory_exists(os.path.dirname(output_path))
    
    image = load_image(image_path)
    image_resized = resize_image(image)
    image_normalized = normalize_image(image_resized)
    image_enhanced = apply_clahe(image_normalized)
    
    save_image(image_enhanced, output_path)
    logging.info("🎉 Image processing completed successfully!")
    return output_path

def process_batch_images(input_paths: List[str], output_paths: List[str]) -> List[str]:
    """
    Process a batch of images in parallel using multiprocessing.
    
    Args:
        input_paths: List of input image paths
        output_paths: List of output image paths
        
    Returns:
        List of output paths for successfully processed images
    """
    if len(input_paths) != len(output_paths):
        logging.error("Input and output paths lists must have the same length")
        return []
    
    # Create a pool of workers
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Process images in parallel
        results = pool.starmap(process_image, zip(input_paths, output_paths))
    
    return results

# Example Usage (Replace with actual paths during deployment)
if __name__ == "__main__":
    # Single image processing
    input_image_path = "media/uploads/jpeg/input.jpg"  # Change to actual path
    output_image_path = "processed_images/output.png"
    process_image(input_image_path, output_image_path)
    
    # Batch processing example
    input_images = [
        "media/uploads/jpeg/image1.jpg",
        "media/uploads/jpeg/image2.jpg",
        "media/uploads/jpeg/image3.jpg"
    ]
    output_images = [
        "processed_images/output1.png",
        "processed_images/output2.png",
        "processed_images/output3.png"
    ]
    processed_paths = process_batch_images(input_images, output_images)
    logging.info(f"Processed {len(processed_paths)} images successfully")
