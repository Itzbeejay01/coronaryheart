import os
import cv2
import numpy as np
import logging
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
from skimage.filters import unsharp_mask
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from typing import Tuple, Dict, Optional, List
import multiprocessing as mp

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_PROCESSES = 3


def resize_image(image: np.ndarray) -> np.ndarray:
    """ Resize to 224x224 """
    target_size = (224, 224)
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image


def apply_clahe_lab(image: np.ndarray) -> np.ndarray:
    """ Apply CLAHE to L channel only """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def apply_unsharp(image: np.ndarray) -> np.ndarray:
    """ Apply unsharp masking per channel """
    for i in range(3):
        channel = image[:, :, i]
        sharpened = unsharp_mask(channel, radius=1.5, amount=1.0)
        image[:, :, i] = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    return image


def apply_denoise(image: np.ndarray) -> np.ndarray:
    """ Denoise each channel """
    denoised = np.zeros_like(image)
    for i in range(3):
        sigma = np.mean(estimate_sigma(image[:, :, i]))
        d = denoise_nl_means(
            image[:, :, i],
            h=0.8 * sigma,
            fast_mode=True,
            patch_size=7,
            patch_distance=5,
            channel_axis=None
        )
        denoised[:, :, i] = np.clip(d * 255, 0, 255).astype(np.uint8)
    return denoised


def calculate_metrics(original: np.ndarray, enhanced: np.ndarray) -> Dict:
    """ Calculate PSNR, SSIM, MSE """
    psnr_val = psnr(original, enhanced, data_range=255)
    ssim_val = ssim(original, enhanced, channel_axis=-1, data_range=255)
    mse_val = mean_squared_error(original, enhanced)
    return {'psnr': psnr_val, 'ssim': ssim_val, 'mse': mse_val}


def process_image(input_path: str, output_path: str) -> Tuple[bool, str, Optional[Dict]]:
    logging.info(f"Processing {input_path} → {output_path}")

    if not os.path.exists(input_path):
        return False, "Input not found", None

    # Load as BGR
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        return False, "Failed to read image", None

    # Convert to RGB
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize early
    image = resize_image(image)
    original = image.copy()

    # Denoise
    image = apply_denoise(image)

    # CLAHE (LAB)
    image = apply_clahe_lab(image)

    # Unsharp
    image = apply_unsharp(image)

    # Final resize check
    image = resize_image(image)

    # Save as grayscale
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        return False, f"Failed to save processed image to {output_path}", None
    if not os.path.exists(output_path):
        return False, f"Processed image was not saved to {output_path}", None
    logging.info(f"Successfully saved processed image to: {output_path}")

    metrics = calculate_metrics(original, image)

    logging.info(f"Saved: {output_path} | PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")

    return True, "Done", metrics


def process_batch(input_paths: List[str], output_paths: List[str]) -> List[Tuple[bool, str, Optional[Dict]]]:
    if len(input_paths) != len(output_paths):
        return [(False, "Input/output length mismatch", None)] * len(input_paths)
    
    with mp.Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(process_image, zip(input_paths, output_paths))
    
    return results


if __name__ == "__main__":
    input_images = ["data/input1.jpg", "data/input2.jpg"]
    output_images = ["data/output1.jpg", "data/output2.jpg"]

    results = process_batch(input_images, output_images)
    for res in results:
        print(res)
