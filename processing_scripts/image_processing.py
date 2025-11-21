import os
import logging
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _ensure_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _calculate_uqi(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))

    denominator = (var1 + var2) * (mu1 ** 2 + mu2 ** 2)
    if denominator == 0:
        return 0.0

    return float((4 * cov * mu1 * mu2) / denominator)


def process_image(input_path: str, output_path: str) -> Tuple[bool, str, Dict[str, float]]:
    """
    Enhance a medical image and persist the processed version.

    Returns (success_flag, message, metrics_dict)
    """
    if not os.path.exists(input_path):
        return False, f"Input file not found: {input_path}", {}

    try:
        original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            return False, "Failed to read input image.", {}

        denoised = cv2.fastNlMeansDenoising(original)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        target_size = (224, 224)
        processed = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)
        normalized = cv2.resize(original, target_size, interpolation=cv2.INTER_CUBIC)

        mse = float(np.mean((normalized - processed) ** 2))
        psnr_value = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0
        ssim_value = float(ssim(normalized, processed))
        uqi_value = _calculate_uqi(normalized, processed)

        _ensure_directory(output_path)
        cv2.imwrite(output_path, processed)

        metrics = {
            'mse': mse,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'uqi': uqi_value,
        }

        logging.info(
            "Processed image %s -> %s | PSNR: %.2f, SSIM: %.4f, MSE: %.4f, UQI: %.4f",
            input_path, output_path, psnr_value, ssim_value, mse, uqi_value
        )
        return True, "Processing complete.", metrics

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Error processing image %s", input_path)
        return False, str(exc), {}
