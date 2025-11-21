import pydicom
import numpy as np
from PIL import Image
import cv2
import os
import logging
from typing import Tuple, List, Dict, Optional
import multiprocessing as mp

# Number of processes to use (3 cores as requested)
NUM_PROCESSES = 3

def get_dicom_info(dicom_data):
    """
    Extract relevant information from DICOM file
    """
    info = {
        'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
        'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
        'modality': getattr(dicom_data, 'Modality', 'Unknown'),
        'manufacturer': getattr(dicom_data, 'Manufacturer', 'Unknown'),
        'slice_thickness': getattr(dicom_data, 'SliceThickness', 'Unknown'),
        'pixel_spacing': getattr(dicom_data, 'PixelSpacing', ['Unknown', 'Unknown']),
        'is_3d': hasattr(dicom_data, 'NumberOfFrames')
    }
    return info

def apply_window_level(image, window_center, window_width):
    """
    Apply windowing to better visualize different tissue types
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = np.clip(image, img_min, img_max)
    window_image = ((window_image - img_min) / (img_max - img_min) * 255).astype('uint8')
    return window_image

def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Force resize image to 224x224 using high-quality interpolation.
    This is a fixed-size resize function specifically for the dataset requirements.
    """
    target_size = (224, 224)
    if image.shape[:2] != target_size:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image

def convert_dicom_to_png(dicom_path, output_path=None, for_visualization=True):
    """
    Convert a DICOM file to PNG format with forced 224x224 resizing
    
    Args:
        dicom_path (str): Path to the input DICOM file
        output_path (str, optional): Path where the PNG file should be saved. If None, will save in media/converted
        for_visualization (bool): If True, enhance for visualization
    
    Returns:
        tuple: (success (bool), message (str), dicom_info (dict))
    """
    try:
        # If no output path specified, create one in media/converted
        if output_path is None:
            filename = os.path.splitext(os.path.basename(dicom_path))[0]
            output_path = os.path.join('media', 'converted', f'{filename}.png')
            
        # Create debug directory
        debug_dir = os.path.join('media', 'debug', filename)
        os.makedirs(debug_dir, exist_ok=True)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # TEST 1: Load and verify DICOM
        logging.info(f"🔄 Loading DICOM: {dicom_path}")
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        dicom_info = get_dicom_info(dicom_data)
        
        # TEST 2: Extract pixel data
        logging.info("🔄 Extracting pixel data...")
        if not hasattr(dicom_data, 'PixelData'):
            logging.error("❌ No pixel data found in DICOM file")
            return False, "No pixel data found in DICOM file", {}
            
        try:
            image = dicom_data.pixel_array
            logging.info("✅ Successfully extracted pixel array")
        except Exception as e:
            logging.error(f"❌ Failed to read pixel_array, trying manual extraction: {str(e)}")
            try:
                raw_pixels = dicom_data.PixelData
                dtype = np.uint16 if getattr(dicom_data, 'BitsAllocated', 8) == 16 else np.uint8
                if hasattr(dicom_data, 'Rows') and hasattr(dicom_data, 'Columns'):
                    image = np.frombuffer(raw_pixels, dtype=dtype)
                    image = image.reshape((dicom_data.Rows, dicom_data.Columns))
                    logging.info("✅ Successfully extracted pixel data manually")
                else:
                    logging.error("❌ Missing image dimensions in DICOM file")
                    return False, "Missing image dimensions in DICOM file", {}
            except Exception as e2:
                logging.error(f"❌ Failed manual pixel data extraction: {str(e2)}")
                return False, f"Failed to extract pixel data: {str(e2)}", {}

        # Store original size
        original_size = image.shape
        dicom_info['original_size'] = f"{original_size[1]}x{original_size[0]}"
        logging.info(f"✅ Original image size: {original_size[1]}x{original_size[0]}")
        
        # TEST 3: Handle 3D DICOM
        if len(image.shape) > 2:
            logging.info("🔄 Processing 3D DICOM...")
            if for_visualization:
                middle_slice = image.shape[0] // 2
                image = image[middle_slice]
            else:
                dicom_info['number_of_slices'] = image.shape[0]
                dicom_info['slice_dimensions'] = f"{image.shape[1]}x{image.shape[2]}"
                image = image[image.shape[0] // 2]
            logging.info("✅ Extracted 2D slice from 3D DICOM")
            
        # Save original image for debugging
        cv2.imwrite(os.path.join(debug_dir, "1_original.png"), image)
        
        # TEST 4: Force resize to 224x224
        logging.info("🔄 Resizing to 224x224...")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        if image.shape[:2] != (224, 224):
            logging.error(f"❌ Resizing failed! Got size: {image.shape[:2]}")
            return False, f"Failed to resize image to 224x224. Current size: {image.shape[:2]}", dicom_info
            
        logging.info("✅ Resizing successful")
        cv2.imwrite(os.path.join(debug_dir, "2_resized.png"), image)
        
        # TEST 5: Normalize pixel values
        logging.info("🔄 Normalizing pixel values...")
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = np.uint8(image)
        cv2.imwrite(os.path.join(debug_dir, "3_normalized.png"), image)
        
        # Verify size after normalization
        if image.shape[:2] != (224, 224):
            logging.error(f"❌ Size changed after normalization! Got size: {image.shape[:2]}")
            return False, f"Image size changed during normalization", dicom_info
        logging.info("✅ Normalization complete")

        if for_visualization:
            # TEST 6: Apply enhancements
            logging.info("🔄 Applying image enhancements...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            cv2.imwrite(os.path.join(debug_dir, "4_clahe.png"), image)
            
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            cv2.imwrite(os.path.join(debug_dir, "5_contrast.png"), image)
            logging.info("✅ Enhancements complete")

        # TEST 7: Final verification and save
        if image.shape[:2] != (224, 224):
            logging.error(f"❌ Final size check failed! Got size: {image.shape[:2]}")
            return False, f"Final image size incorrect: {image.shape[:2]}", dicom_info
            
        # Save final image
        success = cv2.imwrite(output_path, image)
        if not success or not os.path.exists(output_path):
            return False, f"Failed to save PNG to {output_path}", {}
            
        # Verify saved image
        final_check = cv2.imread(output_path)
        if final_check is None or final_check.shape[:2] != (224, 224):
            return False, "Saved image verification failed", dicom_info
            
        # Update info
        dicom_info['resized_size'] = '224x224'
        dicom_info['converted_path'] = output_path
        dicom_info['debug_dir'] = debug_dir
        
        logging.info(f"✅ Successfully converted and saved image to: {output_path}")
        return True, "Successfully converted DICOM to PNG", dicom_info
        
    except Exception as e:
        logging.error(f"❌ Error converting DICOM to PNG: {str(e)}")
        logging.exception(e)
        return False, f"Error converting DICOM to PNG: {str(e)}", {}

def convert_batch_dicom_to_png(dicom_paths: List[str], output_paths: Optional[List[str]] = None, for_visualization: bool = True) -> List[Tuple[bool, str, Dict]]:
    """
    Convert a batch of DICOM files to PNG format in parallel using multiprocessing.
    
    Args:
        dicom_paths: List of input DICOM file paths
        output_paths: List of output PNG file paths (optional, will auto-generate if None)
        for_visualization: Whether to enhance images for visualization
        
    Returns:
        List of conversion results (success, message, dicom_info) for each DICOM file
    """
    # If output paths are not provided, generate them
    if output_paths is None:
        output_paths = []
        for dicom_path in dicom_paths:
            filename = os.path.splitext(os.path.basename(dicom_path))[0]
            output_path = os.path.join('media', 'converted', f'{filename}.png')
            output_paths.append(output_path)
    
    if len(dicom_paths) != len(output_paths):
        logging.error("DICOM paths and output paths lists must have the same length")
        return [(False, "DICOM paths and output paths lists must have the same length", {})] * len(dicom_paths)
    
    # Create a pool of workers
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Process DICOM files in parallel
        args = [(dicom_path, output_path, for_visualization) for dicom_path, output_path in zip(dicom_paths, output_paths)]
        results = pool.starmap(convert_dicom_to_png, args)
    
    return results

if __name__ == "__main__":
    # Example usage of batch processing
    dicom_files = [
        "path/to/dicom1.dcm",
        "path/to/dicom2.dcm",
        "path/to/dicom3.dcm"
    ]
    output_files = [
        "path/to/output1.png",
        "path/to/output2.png",
        "path/to/output3.png"
    ]
    
    results = convert_batch_dicom_to_png(dicom_files, output_files)
    
    for i, (success, message, info) in enumerate(results):
        if success:
            logging.info(f"DICOM {i+1} converted successfully: {message}")
        else:
            logging.error(f"DICOM {i+1} conversion failed: {message}") 