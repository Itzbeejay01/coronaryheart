import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import multiprocessing as mp
from typing import List, Dict, Tuple

# Number of processes to use (3 cores as requested)
NUM_PROCESSES = 3

def calculate_quality_metrics(original_image, processed_image):
    """
    Calculate enhanced quality metrics with improved accuracy
    """
    metrics = {}
    
    # Convert images to grayscale if they're not already
    if len(original_image.shape) > 2:
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_image
        processed_gray = processed_image
    
    # Normalize images to [0,1] range for metric calculations
    original_norm = original_gray.astype(float) / 255
    processed_norm = processed_gray.astype(float) / 255
    
    # Calculate enhanced contrast score
    contrast_orig = measure_local_contrast(original_norm)
    contrast_proc = measure_local_contrast(processed_norm)
    contrast_improvement = max(0, (contrast_proc - contrast_orig) / contrast_orig * 100)
    metrics['contrast_score'] = min(100, contrast_improvement)
    
    # Calculate enhanced edge score
    edge_score = calculate_edge_score(original_norm, processed_norm)
    metrics['edge_score'] = edge_score
    
    # Calculate enhanced noise reduction score
    noise_reduction = calculate_noise_reduction(original_norm, processed_norm)
    metrics['noise_reduction_score'] = noise_reduction
    
    # Calculate structural similarity with multi-scale approach
    ssim_score = calculate_msssim(original_norm, processed_norm)
    metrics['structural_similarity'] = ssim_score * 100
    
    # Calculate detail preservation score
    detail_score = calculate_detail_preservation(original_norm, processed_norm)
    metrics['detail_preservation'] = detail_score
    
    # Calculate weighted overall score
    weights = {
        'contrast_score': 0.2,
        'edge_score': 0.25,
        'noise_reduction_score': 0.15,
        'structural_similarity': 0.25,
        'detail_preservation': 0.15
    }
    
    overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    metrics['overall_score'] = min(100, max(0, overall_score))
    
    return metrics

def measure_local_contrast(image):
    """
    Measure local contrast using improved method
    """
    # Apply Gaussian blur for local mean
    local_mean = cv2.GaussianBlur(image, (11,11), 1.5)
    
    # Calculate local standard deviation
    local_std = np.sqrt(cv2.GaussianBlur(np.square(image - local_mean), (11,11), 1.5))
    
    # Calculate local contrast
    local_contrast = local_std / (local_mean + 0.01)
    
    # Return mean local contrast
    return np.mean(local_contrast) * 100

def calculate_edge_score(original, processed):
    """
    Calculate edge score using multi-scale approach
    """
    def get_edges(img):
        # Multi-scale edge detection
        edges_fine = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
        edges_medium = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
        
        # Combine edge maps
        edges = np.maximum(np.abs(edges_fine), np.abs(edges_medium))
        return edges > 0.1  # Threshold
    
    # Get edge maps
    edges_orig = get_edges(original)
    edges_proc = get_edges(processed)
    
    # Calculate edge preservation ratio
    edge_preservation = np.sum(edges_proc & edges_orig) / (np.sum(edges_orig) + 1e-6)
    
    # Calculate new edge detection ratio
    new_edges = np.sum(edges_proc & ~edges_orig) / (np.sum(edges_orig) + 1e-6)
    
    # Combine scores with weights
    edge_score = 70 * edge_preservation + 30 * new_edges
    return min(100, edge_score * 100)

def calculate_noise_reduction(original, processed):
    """
    Calculate noise reduction score using advanced method
    """
    # Estimate noise in both images
    noise_orig = estimate_noise_level(original)
    noise_proc = estimate_noise_level(processed)
    
    # Calculate noise reduction percentage
    if noise_orig > 0:
        reduction = max(0, (noise_orig - noise_proc) / noise_orig * 100)
    else:
        reduction = 0
    
    return min(100, reduction)

def estimate_noise_level(image):
    """
    Estimate noise level using multiple methods
    """
    # Laplacian method
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    noise_lap = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
    
    # Local variance method
    local_var = cv2.GaussianBlur(np.square(image - cv2.GaussianBlur(image, (7,7), 1.5)), (7,7), 1.5)
    noise_var = np.sqrt(np.mean(local_var))
    
    # Combine estimates
    return (noise_lap + noise_var) / 2

def calculate_msssim(original, processed):
    """
    Calculate multi-scale structural similarity
    """
    # Parameters for MS-SSIM
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    scales = len(weights)
    
    mssim = []
    mcs = []
    for _ in range(scales):
        # Calculate SSIM
        ssim_score, cs = ssim(original, processed, full=True)
        mssim.append(ssim_score)
        mcs.append(cs)
        
        # Downsample images
        if _ < scales - 1:
            original = cv2.pyrDown(original)
            processed = cv2.pyrDown(processed)
    
    # Calculate final MS-SSIM score
    overall_mssim = np.prod(np.power(mcs[:-1], weights[:-1])) * np.power(mssim[-1], weights[-1])
    return overall_mssim

def calculate_detail_preservation(original, processed):
    """
    Calculate detail preservation score
    """
    # Extract high-frequency components
    high_freq_orig = original - cv2.GaussianBlur(original, (5,5), 1.0)
    high_freq_proc = processed - cv2.GaussianBlur(processed, (5,5), 1.0)
    
    # Calculate correlation between high-frequency components
    correlation = np.corrcoef(high_freq_orig.flatten(), high_freq_proc.flatten())[0,1]
    
    # Calculate energy preservation
    energy_orig = np.sum(np.square(high_freq_orig))
    energy_proc = np.sum(np.square(high_freq_proc))
    energy_ratio = min(energy_proc / (energy_orig + 1e-6), 1.0)
    
    # Combine metrics
    detail_score = 100 * (0.7 * correlation + 0.3 * energy_ratio)
    return max(0, min(100, detail_score))

def ssim(img1, img2, full=False):
    """
    Calculate SSIM with optimized parameters
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)
    
    # Calculate variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11,11), 1.5) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if full:
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return np.mean(ssim_map), np.mean(cs_map)
    else:
        return np.mean(ssim_map)

def calculate_quality_metrics_from_paths(original_path: str, processed_path: str) -> Tuple[bool, str, Dict]:
    """
    Calculate quality metrics from image file paths
    
    Args:
        original_path: Path to the original image
        processed_path: Path to the processed image
        
    Returns:
        Tuple of (success, message, metrics)
    """
    try:
        # Load images
        original_image = cv2.imread(original_path)
        processed_image = cv2.imread(processed_path)
        
        if original_image is None:
            return False, f"Failed to load original image: {original_path}", {}
            
        if processed_image is None:
            return False, f"Failed to load processed image: {processed_path}", {}
        
        # Calculate metrics
        metrics = calculate_quality_metrics(original_image, processed_image)
        
        return True, "Quality metrics calculated successfully", metrics
        
    except Exception as e:
        return False, f"Error calculating quality metrics: {str(e)}", {}

def calculate_batch_quality_metrics(original_paths: List[str], processed_paths: List[str]) -> List[Tuple[bool, str, Dict]]:
    """
    Calculate quality metrics for a batch of image pairs in parallel using multiprocessing
    
    Args:
        original_paths: List of paths to original images
        processed_paths: List of paths to processed images
        
    Returns:
        List of results (success, message, metrics) for each image pair
    """
    if len(original_paths) != len(processed_paths):
        return [(False, "Original and processed paths lists must have the same length", {})] * len(original_paths)
    
    # Create a pool of workers
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Process image pairs in parallel
        results = pool.starmap(calculate_quality_metrics_from_paths, zip(original_paths, processed_paths))
    
    return results

if __name__ == "__main__":
    # Example usage of batch processing
    original_images = [
        "path/to/original1.png",
        "path/to/original2.png",
        "path/to/original3.png"
    ]
    processed_images = [
        "path/to/processed1.png",
        "path/to/processed2.png",
        "path/to/processed3.png"
    ]
    
    results = calculate_batch_quality_metrics(original_images, processed_images)
    
    for i, (success, message, metrics) in enumerate(results):
        if success:
            print(f"Image pair {i+1} metrics: {metrics}")
        else:
            print(f"Image pair {i+1} calculation failed: {message}") 