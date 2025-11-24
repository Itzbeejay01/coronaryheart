from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.utils import timezone
from .models import DicomImage, JpegImage, ProcessingBatch
from processing_scripts.image_processing import process_image
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import pydicom
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import Counter

# Import the new form and the prediction function


# Global storage for processing progress
processing_status = {}

SEVERITY_THRESHOLDS = {
    'low': 0.34,
    'moderate': 0.67,
}


def classify_severity(probability: float):
    """
    Map a probability (0-1) to a qualitative severity label and key.
    """
    if probability < SEVERITY_THRESHOLDS['low']:
        return "Low Severity", "low"
    if probability < SEVERITY_THRESHOLDS['moderate']:
        return "Moderate Severity", "moderate"
    return "High Severity", "high"


def infer_jpeg_image_type(filename: str) -> str:
    """
    Determine image type from the file extension so we can track PNG vs JPEG uploads.
    """
    extension = os.path.splitext(filename.lower())[1]
    if extension == '.png':
        return 'png'
    if extension in {'.jpg', '.jpeg'}:
        return 'jpeg'
    return 'other'

def update_progress(image_id, step, progress):
    """Update processing progress for an image"""
    if image_id not in processing_status:
        processing_status[image_id] = {
            'current_step': '',
            'progress': 0,
            'completed_steps': [],
            'status': 'processing'
        }
    
    processing_status[image_id].update({
        'current_step': step,
        'progress': progress,
        'status': 'completed' if progress == 100 else 'processing'
    })
    
    if progress == 100:
        if step not in processing_status[image_id]['completed_steps']:
            processing_status[image_id]['completed_steps'].append(step)

def process_image_async(dicom_image):
    """Process image asynchronously with progress tracking"""
    try:
        logging.info(f"Starting processing for image {dicom_image.id}")
        
        # Convert DICOM to PNG
        update_progress(dicom_image.id, "Converting DICOM", 0)
        
        # Read DICOM and convert to PNG
        ds = pydicom.dcmread(dicom_image.dicom_file.path)
        pixel_array = ds.pixel_array
        
        # Extract DICOM metadata
        dicom_info = {
            'PatientID': getattr(ds, 'PatientID', 'Unknown'),
            'StudyDate': getattr(ds, 'StudyDate', 'Unknown'),
            'Modality': getattr(ds, 'Modality', 'Unknown'),
            'Manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),
            'ImageLaterality': getattr(ds, 'ImageLaterality', 'Unknown'),
            'ViewPosition': getattr(ds, 'ViewPosition', 'Unknown'),
            'PixelSpacing': getattr(ds, 'PixelSpacing', ['Unknown', 'Unknown']),
            'Dimensions': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}"
        }
        
        # Normalize to 8-bit range
        pixel_min = float(pixel_array.min())
        pixel_max = float(pixel_array.max())
        if pixel_min == pixel_max:
            raise Exception("Image has no contrast")
            
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        
        # Save visualization image
        visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', f"{dicom_image.id}_visual.png")
        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
        cv2.imwrite(visualization_path, normalized)
        dicom_image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
        
        # Save converted image for processing
        converted_path = os.path.join(settings.MEDIA_ROOT, 'converted', f"{dicom_image.id}_converted.png")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        cv2.imwrite(converted_path, normalized)
        
        update_progress(dicom_image.id, "Converting DICOM", 100)
        logging.info("DICOM conversion complete")
        
        # Store DICOM info
        dicom_image.dicom_info = json.dumps(dicom_info)
        dicom_image.save()
        
        # Process the image
        update_progress(dicom_image.id, "Enhancing image", 0)
        enhanced_path = os.path.join(settings.MEDIA_ROOT, 'processed', f"{dicom_image.id}_enhanced.png")
        
        success, message, processing_info = process_image(converted_path, enhanced_path)
        
        if not success:
            raise Exception(f"Image enhancement failed: {message}")
            
        # Store the metrics and processing info
        dicom_image.preprocessing_steps = {
            'psnr': processing_info.get('psnr', 0),
            'ssim': processing_info.get('ssim', 0),
            'mse': processing_info.get('mse', 0),
            'uqi': processing_info.get('uqi', 0),
            'original_dimensions': dicom_info['Dimensions'],
            'processed_dimensions': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}",
            'processing_steps': [
                'DICOM to PNG conversion',
                'Noise reduction using Non-local Means',
                'Contrast enhancement using CLAHE',
                'Edge sharpening',
                'Quality metrics calculation'
            ]
        }
        
        # Save the enhanced image
        relative_path = os.path.relpath(enhanced_path, settings.MEDIA_ROOT)
        dicom_image.processed_image = relative_path
        dicom_image.processed_date = timezone.now()
        dicom_image.status = 'completed'
        dicom_image.save()
            
        logging.info(f"Saved processed image path: {relative_path}")
        
        update_progress(dicom_image.id, "Enhancing image", 100)
        logging.info(f"Processing completed for DICOM image {dicom_image.id}")
        
    except Exception as e:
        logging.error(f"Error processing DICOM image {dicom_image.id}: {str(e)}")
        logging.error(traceback.format_exc())
        dicom_image.status = 'error'
        dicom_image.save()
        update_progress(dicom_image.id, "Error", 100)

def ensure_media_dirs():
    """Ensure all required media directories exist"""
    media_dirs = [
        'uploads', 
        'converted', 
        'processed',
        'uploads/dicom',
        'uploads/dicom/preview',
        'uploads/jpeg',
        'uploads/jpeg/preview'
    ]
    for dir_name in media_dirs:
        dir_path = os.path.join(settings.MEDIA_ROOT, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")

# Ensure all required media directories exist
ensure_media_dirs()

@csrf_exempt
def upload(request):
    if request.method == 'POST':
        dicom_files = request.FILES.getlist('dicom_files')
        jpeg_files = request.FILES.getlist('jpeg_files')
        
        if not dicom_files and not jpeg_files:
            return JsonResponse({'status': 'error', 'message': 'No files were uploaded'})

        processed_images = []
        batch = ProcessingBatch.objects.create(
            total_files=len(dicom_files) + len(jpeg_files)
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Process DICOM files
            for dicom_file in dicom_files:
                try:
                    logging.info(f"Processing DICOM file: {dicom_file.name}")
                    
                    # Save the DICOM file
                    dicom_image = DicomImage.objects.create(
                        title=dicom_file.name,
                        dicom_file=dicom_file,
                        upload_date=timezone.now(),
                        status='processing'
                    )
                    logging.info(f"Created DicomImage object with ID: {dicom_image.id}")
                    
                    batch.dicom_images.add(dicom_image)
                    processed_images.append(dicom_image)
                    
                    # Process the image asynchronously
                    executor.submit(process_dicom_image_async, dicom_image)
                    
                except Exception as e:
                    logging.error(f"Error processing DICOM file {dicom_file.name}: {str(e)}")
                    logging.exception(e)
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Error processing {dicom_file.name}: {str(e)}'
                    })

            # Process JPEG files
            for jpeg_file in jpeg_files:
                try:
                    logging.info(f"Processing JPEG file: {jpeg_file.name}")
                    
                    # Determine image type from file extension (PNG vs JPEG)
                    image_type = infer_jpeg_image_type(jpeg_file.name)
                    
                    # Save the JPEG file
                    jpeg_image = JpegImage.objects.create(
                        title=jpeg_file.name,
                        image_file=jpeg_file,
                        image_type=image_type,
                        upload_date=timezone.now(),
                        status='processing'
                    )
                    logging.info(f"Created JpegImage object with ID: {jpeg_image.id}")
                    
                    # Add to batch and process
                    batch.jpeg_images.add(jpeg_image)
                    processed_images.append(jpeg_image)
                    
                    # Process the image asynchronously
                    executor.submit(process_jpeg_image_async, jpeg_image)
                    
                except Exception as e:
                    logging.error(f"Error processing JPEG file {jpeg_file.name}: {str(e)}")
                    logging.exception(e)
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Error processing {jpeg_file.name}: {str(e)}'
                    })

        # Update batch information
        batch.save()

        # Always redirect to the result page of the first image
        if processed_images:
            image = processed_images[0]
            logging.info(f"Upload complete. Redirecting to result page for image {image.id}")
            return JsonResponse({
                'status': 'success',
                'redirect_url': f'/result/{image.id}/',
                'image_id': image.id,
                'batch_id': batch.batch_id
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No images were processed'
            })

    return render(request, 'preprocessing_app/upload.html')

def process_dicom_image_async(image):
    try:
        logging.info(f"Starting processing for DICOM image {image.id}")
        
        # Convert DICOM to PNG
        update_progress(image.id, "Converting DICOM", 0)
        
        # Read DICOM and convert to PNG
        ds = pydicom.dcmread(image.dicom_file.path)
        pixel_array = ds.pixel_array
        
        # Extract DICOM metadata
        dicom_info = {
            'PatientID': getattr(ds, 'PatientID', 'Unknown'),
            'StudyDate': getattr(ds, 'StudyDate', 'Unknown'),
            'Modality': getattr(ds, 'Modality', 'Unknown'),
            'Manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),
            'ImageLaterality': getattr(ds, 'ImageLaterality', 'Unknown'),
            'ViewPosition': getattr(ds, 'ViewPosition', 'Unknown'),
            'PixelSpacing': getattr(ds, 'PixelSpacing', ['Unknown', 'Unknown']),
            'Dimensions': f"{pixel_array.shape[1]}x{pixel_array.shape[0]}"
        }
        
        # Normalize to 8-bit range
        pixel_min = float(pixel_array.min())
        pixel_max = float(pixel_array.max())
        if pixel_min == pixel_max:
            raise Exception("Image has no contrast")
            
        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        
        # Save visualization image
        visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dicom', 'preview', f"{image.id}_visual.png")
        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
        cv2.imwrite(visualization_path, normalized)
        image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
        
        # Save converted image for processing
        converted_path = os.path.join(settings.MEDIA_ROOT, 'converted', f"{image.id}_converted.png")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        cv2.imwrite(converted_path, normalized)
        
        update_progress(image.id, "Converting DICOM", 100)
        logging.info(f"DICOM conversion complete for image {image.id}")
        
        # Store DICOM info - Convert to JSON string
        image.dicom_info = json.dumps(dicom_info)
        image.save()
        
        # Process the image
        update_progress(image.id, "Enhancing image", 0)
        enhanced_path = os.path.join(settings.MEDIA_ROOT, 'processed', f"{image.id}_enhanced.png")
        os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
        
        # Apply image processing
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(normalized)
        update_progress(image.id, "Enhancing image", 30)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        update_progress(image.id, "Enhancing image", 60)
        
        # 3. Sharpen edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        update_progress(image.id, "Enhancing image", 80)
        
        # 4. Resize to 224x224 for standardization
        target_size = (224, 224)
        resized = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)
        # Also resize the normalized image to match
        normalized_resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC)
        update_progress(image.id, "Enhancing image", 90)
        
        # Save processed image as PNG
        cv2.imwrite(enhanced_path, resized)
        logging.info(f"Saved processed image to: {enhanced_path}")
        
        # Verify file existence
        if os.path.exists(enhanced_path):
            logging.info(f"Verified processed file exists: {enhanced_path}")
            file_size = os.path.getsize(enhanced_path)
            logging.info(f"File size: {file_size} bytes")
        else:
            logging.error(f"Failed to save processed file: {enhanced_path}")
        
        # Calculate quality metrics using the resized versions
        mse = np.mean((normalized_resized - resized) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Calculate SSIM using the resized versions
        ssim_value = ssim(normalized_resized, resized)
        
        # Calculate UQI (Universal Quality Index)
        def calculate_uqi(img1, img2):
            # Convert to float
            img1 = img1.astype(float)
            img2 = img2.astype(float)
            
            # Calculate means
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            # Calculate variances and covariance
            var1 = np.var(img1)
            var2 = np.var(img2)
            cov = np.mean((img1 - mu1) * (img2 - mu2))
            
            # Calculate UQI
            numerator = 4 * cov * mu1 * mu2
            denominator = (var1 + var2) * (mu1**2 + mu2**2)
            
            # Avoid division by zero
            if denominator == 0:
                return 0
                
            return numerator / denominator
        
        uqi_value = calculate_uqi(normalized_resized, resized)
        
        # Store processing information - Convert to JSON string
        preprocessing_steps = {
            'psnr': float(psnr),
            'mse': float(mse),
            'ssim': float(ssim_value),
            'uqi': float(uqi_value),
            'original_dimensions': dicom_info['Dimensions'],
            'processed_dimensions': f"{target_size[0]}x{target_size[1]}",
            'processing_steps': [
                'DICOM to PNG conversion',
                'Noise reduction using Non-local Means',
                'Contrast enhancement using CLAHE',
                'Edge sharpening',
                'Quality metrics calculation (PSNR, MSE, SSIM, UQI)'
            ]
        }
        # Store as a dictionary, not a JSON string
        image.preprocessing_steps = preprocessing_steps
        
        # Save the enhanced image
        relative_path = os.path.relpath(enhanced_path, settings.MEDIA_ROOT)
        image.processed_image = relative_path
        image.processed_date = timezone.now()
        image.status = 'completed'
        image.save()
            
        logging.info(f"Saved processed image path: {relative_path}")
        
        update_progress(image.id, "Enhancing image", 100)
        logging.info(f"Processing completed for DICOM image {image.id}")
        
    except Exception as e:
        logging.error(f"Error processing DICOM image {image.id}: {str(e)}")
        logging.error(traceback.format_exc())
        image.status = 'error'
        image.save()
        update_progress(image.id, "Error", 100)

def process_jpeg_image_async(image):
    """Process a JPEG image asynchronously"""
    try:
        logging.info(f"Starting processing for JPEG image {image.id}")
        
        # Read the JPEG image
        update_progress(image.id, "Reading image", 0)
        img = cv2.imread(image.image_file.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Failed to read image file")
        
        # Store original dimensions
        dimensions = f"{img.shape[1]}x{img.shape[0]}"
        
        # Save visualization image (original JPEG)
        visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'jpeg', 'preview', f"{image.id}_visual.png")
        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
        cv2.imwrite(visualization_path, img)
        image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
        
        # Convert to PNG for processing
        converted_path = os.path.join(settings.MEDIA_ROOT, 'converted', f"{image.id}_converted.png")
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        cv2.imwrite(converted_path, img)
        
        update_progress(image.id, "Converting to PNG", 100)
        logging.info(f"JPEG conversion complete for image {image.id}")
        
        # Process the image
        update_progress(image.id, "Enhancing image", 0)
        enhanced_path = os.path.join(settings.MEDIA_ROOT, 'processed', f"{image.id}_enhanced.png")
        os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
        
        # Apply same image processing as DICOM
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(img)
        update_progress(image.id, "Enhancing image", 30)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        update_progress(image.id, "Enhancing image", 60)
        
        # 3. Sharpen edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        update_progress(image.id, "Enhancing image", 80)
        
        # 4. Resize to 224x224 for standardization
        target_size = (224, 224)
        resized = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)
        # Also resize the normalized image to match
        normalized_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        update_progress(image.id, "Enhancing image", 90)
        
        # Save processed image as PNG
        cv2.imwrite(enhanced_path, resized)
        logging.info(f"Saved JPEG processed image to: {enhanced_path}")
        
        # Verify file existence
        if os.path.exists(enhanced_path):
            logging.info(f"Verified JPEG processed file exists: {enhanced_path}")
            file_size = os.path.getsize(enhanced_path)
            logging.info(f"File size: {file_size} bytes")
        
        # Calculate quality metrics using the resized versions
        mse = np.mean((normalized_resized - resized) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
        
        # Calculate SSIM using the resized versions
        ssim_value = ssim(normalized_resized, resized)
        
        # Calculate UQI (Universal Quality Index)
        def calculate_uqi(img1, img2):
            # Convert to float
            img1 = img1.astype(float)
            img2 = img2.astype(float)
            
            # Calculate means
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            # Calculate variances and covariance
            var1 = np.var(img1)
            var2 = np.var(img2)
            cov = np.mean((img1 - mu1) * (img2 - mu2))
            
            # Calculate UQI
            numerator = 4 * cov * mu1 * mu2
            denominator = (var1 + var2) * (mu1**2 + mu2**2)
            
            # Avoid division by zero
            if denominator == 0:
                return 0
            
            return numerator / denominator
        
        uqi_value = calculate_uqi(normalized_resized, resized)
        
        # Log metric values before storing
        logging.info(f"Metrics for image {image.id}: PSNR={psnr:.2f}, SSIM={ssim_value:.4f}, MSE={mse:.4f}, UQI={uqi_value:.4f}")
        
        # Update the image record with the processed image path
        image.processed_image = os.path.relpath(enhanced_path, settings.MEDIA_ROOT)
        image.status = 'completed'
        
        # Store preprocessing steps and metrics
        image.preprocessing_steps = {
            'psnr': float(psnr),
            'ssim': float(ssim_value),
            'mse': float(mse),
            'uqi': float(uqi_value),
            'original_dimensions': dimensions,
            'processed_dimensions': f"{target_size[0]}x{target_size[1]}",
            'processing_steps': [
                'Noise reduction using Non-local Means',
                'Contrast enhancement using CLAHE',
                'Edge sharpening',
                'Resizing to 224x224',
                'Quality metrics calculation'
            ]
        }
        
        image.save()
        logging.info(f"Updated JPEG image record {image.id} with processed image path and metrics")
        
        update_progress(image.id, "Processing complete", 100)
        logging.info(f"Processing completed for JPEG image {image.id}")
        
    except Exception as e:
        logging.error(f"Error processing JPEG image {image.id}: {str(e)}")
        logging.error(traceback.format_exc())
        image.status = 'error'
        image.save()
        update_progress(image.id, "Error", 100)

@csrf_exempt
def batch_status(request, batch_id=None):
    """View for showing batch processing status"""
    if not batch_id:
        return redirect('upload')
        
    try:
        batch = ProcessingBatch.objects.get(batch_id=batch_id)
    except ProcessingBatch.DoesNotExist:
        return redirect('upload')
        
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # AJAX request - return JSON status
        dicom_statuses = []
        jpeg_statuses = []
        
        # Get DICOM image statuses
        for image in batch.dicom_images.all():
            if str(image.id) in processing_status:
                status_info = processing_status[str(image.id)]
            else:
                status_info = {
                    'status': image.status,
                    'progress': 100 if image.status == 'completed' else 0,
                    'current_step': '',
                    'type': 'dicom'
                }
            dicom_statuses.append(status_info)
            
        # Get JPEG image statuses
        for image in batch.jpeg_images.all():
            if str(image.id) in processing_status:
                status_info = processing_status[str(image.id)]
            else:
                status_info = {
                    'status': image.status,
                    'progress': 100 if image.status == 'completed' else 0,
                    'current_step': '',
                    'type': 'jpeg'
                }
            jpeg_statuses.append(status_info)
        
        # Calculate overall progress
        total_files = len(dicom_statuses) + len(jpeg_statuses)
        completed_files = len([s for s in dicom_statuses + jpeg_statuses 
                             if s['status'] in ['completed', 'error']])
        
        # Calculate validation progress
        validation_complete = any(s['current_step'] != '' for s in dicom_statuses + jpeg_statuses)
        validation_progress = 100 if validation_complete else (
            (len(dicom_statuses) + len(jpeg_statuses)) / batch.total_files * 100
        )
        
        # Calculate processing progress
        if total_files > 0:
            total_progress = sum(s.get('progress', 0) for s in dicom_statuses + jpeg_statuses)
            processing_progress = total_progress / total_files
        else:
            processing_progress = 0
            
        # Determine overall status
        all_completed = completed_files == total_files
        
        # Generate status message
        if not validation_complete:
            status_message = f"Validating files... ({len(dicom_statuses) + len(jpeg_statuses)}/{batch.total_files})"
        elif all_completed:
            status_message = "Processing complete!"
        else:
            status_message = f"Processing images... ({completed_files}/{total_files} complete)"
        
        return JsonResponse({
            'validation_status': 'complete' if validation_complete else 'in_progress',
            'validation_progress': validation_progress,
            'processing_status': 'complete' if all_completed else 'in_progress',
            'processing_progress': processing_progress,
            'status_message': status_message,
            'all_completed': all_completed,
            'total_files': total_files,
            'processed_files': completed_files,
            'statuses': {
                'dicom': dicom_statuses,
                'jpeg': jpeg_statuses
            }
        })
    else:
        # Regular request - show batch status page
        context = {
            'batch': batch,
            'dicom_images': batch.dicom_images.all(),
            'jpeg_images': batch.jpeg_images.all()
        }
        return render(request, 'preprocessing_app/batch_status.html', context)

def processing_status_view(request, image_id):
    """Display processing status page"""
    try:
        image = DicomImage.objects.get(id=image_id)
        return render(request, 'preprocessing_app/processing_status.html', {
            'image': image,
            'image_id': image_id
        })
    except DicomImage.DoesNotExist:
        return redirect('upload')

@csrf_exempt
def get_processing_progress(request, image_id):
    """API endpoint to get current processing progress"""
    try:
        # First try to find the image
        try:
            dicom_image = DicomImage.objects.get(id=image_id)
            image = dicom_image
            image_type = 'dicom'
        except DicomImage.DoesNotExist:
            try:
                jpeg_image = JpegImage.objects.get(id=image_id)
                image = jpeg_image
                image_type = 'jpeg'
            except JpegImage.DoesNotExist:
                return JsonResponse({'status': 'not_found', 'message': f'Image with ID {image_id} not found'}, status=404)
        
        # Get the status from the model if it's in the database
        if image.status == 'completed':
            # Get dimensions for redirecting to result
            dimensions = None
            processed_dimensions = None
            if image_type == 'dicom' and image.dicom_info:
                try:
                    dicom_info = json.loads(image.dicom_info)
                    dimensions = dicom_info.get('Dimensions', None)
                except json.JSONDecodeError:
                    pass
            
            if image.preprocessing_steps:
                try:
                    # Handle both dictionary and JSON string formats
                    if isinstance(image.preprocessing_steps, str):
                        preprocessing_data = json.loads(image.preprocessing_steps)
                    else:
                        preprocessing_data = image.preprocessing_steps
                    logging.info(f"Preprocessing data: {preprocessing_data}")
                    
                    # Extract metrics with proper error handling
                    psnr = preprocessing_data.get('psnr', 'N/A')
                    mse = preprocessing_data.get('mse', 'N/A')
                    ssim = preprocessing_data.get('ssim', 'N/A')
                    uqi = preprocessing_data.get('uqi', 'N/A')
                    
                    # Get dimensions based on image type
                    if image_type == 'dicom':
                        dicom_info = json.loads(image.dicom_info) if image.dicom_info else {}
                        dimensions = dicom_info.get('Dimensions', 'N/A')
                    else:
                        dimensions = preprocessing_data.get('original_dimensions', 'N/A')
                    
                    # Get processed dimensions
                    processed_dimensions = preprocessing_data.get('processed_dimensions', '224x224')
                    
                    logging.info(f"Extracted metrics - PSNR: {psnr}, MSE: {mse}, SSIM: {ssim}, UQI: {uqi}")
                    
                    # Check if the processed image file exists
                    processed_url = None
                    if image.processed_image:
                        processed_path = os.path.join(settings.MEDIA_ROOT, str(image.processed_image))
                        if os.path.exists(processed_path):
                            processed_url = image.processed_image.url
                            logging.info(f"Processed image found at: {processed_path}")
                        else:
                            logging.warning(f"Processed image file does not exist: {processed_path}")
                    
                    # For DICOM images, ensure we have a visualization image
                    original_image_url = None
                    if image_type == 'dicom':
                        if image.visualization_image:
                            original_image_url = image.visualization_image.url
                        else:
                            # If no visualization image exists, create one
                            try:
                                ds = pydicom.dcmread(image.dicom_file.path)
                                pixel_array = ds.pixel_array
                                
                                # Normalize to 8-bit range
                                pixel_min = float(pixel_array.min())
                                pixel_max = float(pixel_array.max())
                                if pixel_min != pixel_max:
                                    normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                                    
                                    # Save visualization image
                                    visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dicom', 'preview', f"{image.id}_visual.png")
                                    os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                                    cv2.imwrite(visualization_path, normalized)
                                    
                                    # Update the model with the visualization image path
                                    image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
                                    image.save()
                                    
                                    original_image_url = image.visualization_image.url
                            except Exception as e:
                                logging.error(f"Error creating DICOM visualization: {str(e)}")
                                original_image_url = None
                    else:
                        if image.visualization_image:
                            original_image_url = image.visualization_image.url
                        else:
                            original_image_url = image.image_file.url
                    
                    # Create image data dictionary
                    image_data = {
                        'id': image.id,
                        'title': image.title,
                        'type': 'DICOM' if image_type == 'dicom' else 'JPEG',
                        'original_image': original_image_url,
                        'processed_image': processed_url,
                        'original_dimensions': dimensions,
                        'processed_dimensions': processed_dimensions,
                        'psnr': f"{float(psnr):.2f}" if psnr != 'N/A' else 'N/A',
                        'mse': f"{float(mse):.4f}" if mse != 'N/A' else 'N/A',
                        'ssim': f"{float(ssim):.4f}" if ssim != 'N/A' else 'N/A',
                        'uqi': f"{float(uqi):.4f}" if uqi != 'N/A' else 'N/A',
                        'processing_steps': preprocessing_data.get('processing_steps', [])
                    }
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing preprocessing_steps JSON for image {image.id}: {str(e)}")

            return JsonResponse({
                'status': 'completed',
                'progress': 100,
                'current_step': 'Processing complete',
                'redirect': f'/result/{image_id}/',
                'dimensions': dimensions,  # Original dimensions
                'processed_dimensions': processed_dimensions  # Processed dimensions
            })
        elif image.status == 'error':
            return JsonResponse({
                'status': 'error',
                'progress': 100,
                'current_step': 'Error occurred',
                'message': 'An error occurred during processing'
            })
        
        # Check if we have status in memory (try both string and int versions of the ID)
        status_key = str(image_id)
        if status_key in processing_status:
            status = processing_status[status_key]
            if status.get('status') == 'completed':
                # Get dimensions for redirecting to result
                dimensions = None
                processed_dimensions = None
                if image_type == 'dicom' and image.dicom_info:
                    try:
                        dicom_info = json.loads(image.dicom_info)
                        dimensions = dicom_info.get('Dimensions', None)
                    except json.JSONDecodeError:
                        pass
                
                if image.preprocessing_steps:
                    try:
                        # Handle both dictionary and JSON string formats
                        if isinstance(image.preprocessing_steps, str):
                            preprocessing_data = json.loads(image.preprocessing_steps)
                        else:
                            preprocessing_data = image.preprocessing_steps
                        logging.info(f"Preprocessing data: {preprocessing_data}")
                        
                        # Extract metrics with proper error handling
                        psnr = preprocessing_data.get('psnr', 'N/A')
                        mse = preprocessing_data.get('mse', 'N/A')
                        ssim = preprocessing_data.get('ssim', 'N/A')
                        uqi = preprocessing_data.get('uqi', 'N/A')
                        
                        # Get dimensions based on image type
                        if image_type == 'dicom':
                            dicom_info = json.loads(image.dicom_info) if image.dicom_info else {}
                            dimensions = dicom_info.get('Dimensions', 'N/A')
                        else:
                            dimensions = preprocessing_data.get('original_dimensions', 'N/A')
                        
                        # Get processed dimensions
                        processed_dimensions = preprocessing_data.get('processed_dimensions', '224x224')
                        
                        logging.info(f"Extracted metrics - PSNR: {psnr}, MSE: {mse}, SSIM: {ssim}, UQI: {uqi}")
                        
                        # Check if the processed image file exists
                        processed_url = None
                        if image.processed_image:
                            processed_path = os.path.join(settings.MEDIA_ROOT, str(image.processed_image))
                            if os.path.exists(processed_path):
                                processed_url = image.processed_image.url
                                logging.info(f"Processed image found at: {processed_path}")
                            else:
                                logging.warning(f"Processed image file does not exist: {processed_path}")
                        
                        # For DICOM images, ensure we have a visualization image
                        original_image_url = None
                        if image_type == 'dicom':
                            if image.visualization_image:
                                original_image_url = image.visualization_image.url
                            else:
                                # If no visualization image exists, create one
                                try:
                                    ds = pydicom.dcmread(image.dicom_file.path)
                                    pixel_array = ds.pixel_array
                                    
                                    # Normalize to 8-bit range
                                    pixel_min = float(pixel_array.min())
                                    pixel_max = float(pixel_array.max())
                                    if pixel_min != pixel_max:
                                        normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                                        
                                        # Save visualization image
                                        visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dicom', 'preview', f"{image.id}_visual.png")
                                        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                                        cv2.imwrite(visualization_path, normalized)
                                        
                                        # Update the model with the visualization image path
                                        image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
                                        image.save()
                                        
                                        original_image_url = image.visualization_image.url
                                except Exception as e:
                                    logging.error(f"Error creating DICOM visualization: {str(e)}")
                                    original_image_url = None
                        else:
                            if image.visualization_image:
                                original_image_url = image.visualization_image.url
                            else:
                                original_image_url = image.image_file.url
                        
                        # Create image data dictionary
                        image_data = {
                            'id': image.id,
                            'title': image.title,
                            'type': 'DICOM' if image_type == 'dicom' else 'JPEG',
                            'original_image': original_image_url,
                            'processed_image': processed_url,
                            'original_dimensions': dimensions,
                            'processed_dimensions': processed_dimensions,
                            'psnr': f"{float(psnr):.2f}" if psnr != 'N/A' else 'N/A',
                            'mse': f"{float(mse):.4f}" if mse != 'N/A' else 'N/A',
                            'ssim': f"{float(ssim):.4f}" if ssim != 'N/A' else 'N/A',
                            'uqi': f"{float(uqi):.4f}" if uqi != 'N/A' else 'N/A',
                            'processing_steps': preprocessing_data.get('processing_steps', [])
                        }
                        
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing preprocessing_steps JSON for image {image.id}: {str(e)}")
                else:
                    logging.warning(f"No preprocessing_steps found for {image_type} image {image.id}")

                return JsonResponse({
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'Processing complete',
                    'redirect': f'/result/{image_id}/',
                    'dimensions': dimensions,  # Original dimensions
                    'processed_dimensions': processed_dimensions  # Processed dimensions
                })
            return JsonResponse(status)
        
        # If no stored status but it's processing
        if image.status == 'processing':
            return JsonResponse({
                'status': 'processing',
                'progress': 50,  # Arbitrary progress value
                'current_step': 'Processing image'
            })
        
        # Default response for unknown status
        return JsonResponse({
            'status': image.status if image.status else 'unknown',
            'progress': 0,
            'current_step': 'Pending processing'
        })
        
    except Exception as e:
        logging.error(f"Error in get_processing_progress: {str(e)}")
        logging.error(traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def result(request, image_id):
    try:
        # Try to get DICOM image first
        try:
            image = DicomImage.objects.get(id=image_id)
            image_type = 'dicom'
            logging.info(f"Found DICOM image with ID {image_id}")
        except DicomImage.DoesNotExist:
            # If not DICOM, try JPEG
            image = JpegImage.objects.get(id=image_id)
            image_type = 'jpeg'
            logging.info(f"Found JPEG image with ID {image_id}")
        
        # Get batch information based on image type
        if image_type == 'dicom':
            batch = ProcessingBatch.objects.filter(dicom_images=image).first()
        else:
            batch = ProcessingBatch.objects.filter(jpeg_images=image).first()
        
        if not batch:
            logging.error(f"No batch found for image {image_id}")
            return render(request, 'preprocessing_app/error.html', {'error': 'No batch found for this image'})
            
        processed_images = []
        
        # Process all DICOM images in the batch
        for dicom_image in batch.dicom_images.all():
            try:
                if dicom_image.preprocessing_steps:
                    # Handle both dictionary and JSON string formats
                    if isinstance(dicom_image.preprocessing_steps, str):
                        preprocessing_data = json.loads(dicom_image.preprocessing_steps)
                    else:
                        preprocessing_data = dicom_image.preprocessing_steps
                    
                    # Extract metrics
                    psnr = preprocessing_data.get('psnr', 'N/A')
                    mse = preprocessing_data.get('mse', 'N/A')
                    ssim = preprocessing_data.get('ssim', 'N/A')
                    uqi = preprocessing_data.get('uqi', 'N/A')
                    
                    # Get dimensions from DICOM info
                    dicom_info = json.loads(dicom_image.dicom_info) if dicom_image.dicom_info else {}
                    dimensions = dicom_info.get('Dimensions', 'N/A')
                    processed_dimensions = preprocessing_data.get('processed_dimensions', '224x224')
                    
                    # Get image URLs
                    processed_url = None
                    if dicom_image.processed_image:
                        processed_path = os.path.join(settings.MEDIA_ROOT, str(dicom_image.processed_image))
                        if os.path.exists(processed_path):
                            processed_url = dicom_image.processed_image.url
                    
                    # For DICOM images, ensure we have a visualization image
                    original_image_url = None
                    if dicom_image.visualization_image:
                        original_image_url = dicom_image.visualization_image.url
                    else:
                        # If no visualization image exists, create one
                        try:
                            ds = pydicom.dcmread(dicom_image.dicom_file.path)
                            pixel_array = ds.pixel_array
                            
                            # Normalize to 8-bit range
                            pixel_min = float(pixel_array.min())
                            pixel_max = float(pixel_array.max())
                            if pixel_min != pixel_max:
                                normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                                
                                # Save visualization image
                                visualization_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dicom', 'preview', f"{dicom_image.id}_visual.png")
                                os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
                                cv2.imwrite(visualization_path, normalized)
                                
                                # Update the model with the visualization image path
                                dicom_image.visualization_image = os.path.relpath(visualization_path, settings.MEDIA_ROOT)
                                dicom_image.save()
                                
                                original_image_url = dicom_image.visualization_image.url
                        except Exception as e:
                            logging.error(f"Error creating DICOM visualization: {str(e)}")
                            original_image_url = None
                    
                    # Create image data dictionary
                    image_data = {
                        'id': dicom_image.id,
                        'title': dicom_image.title,
                        'type': 'DICOM',
                        'original_image': original_image_url,
                        'processed_image': processed_url,
                        'original_dimensions': dimensions,
                        'processed_dimensions': processed_dimensions,
                        'psnr': f"{float(psnr):.2f}" if psnr != 'N/A' else 'N/A',
                        'mse': f"{float(mse):.4f}" if mse != 'N/A' else 'N/A',
                        'ssim': f"{float(ssim):.4f}" if ssim != 'N/A' else 'N/A',
                        'uqi': f"{float(uqi):.4f}" if uqi != 'N/A' else 'N/A',
                        'processing_steps': preprocessing_data.get('processing_steps', [])
                    }
                    processed_images.append(image_data)
            except Exception as e:
                logging.error(f"Error processing DICOM image {dicom_image.id}: {str(e)}")
                continue
        
        # Process all JPEG images in the batch
        for jpeg_image in batch.jpeg_images.all():
            try:
                if jpeg_image.preprocessing_steps:
                    # Handle both dictionary and JSON string formats
                    if isinstance(jpeg_image.preprocessing_steps, str):
                        preprocessing_data = json.loads(jpeg_image.preprocessing_steps)
                    else:
                        preprocessing_data = jpeg_image.preprocessing_steps
                    
                    # Extract metrics
                    psnr = preprocessing_data.get('psnr', 'N/A')
                    mse = preprocessing_data.get('mse', 'N/A')
                    ssim = preprocessing_data.get('ssim', 'N/A')
                    uqi = preprocessing_data.get('uqi', 'N/A')
                    
                    # Get dimensions
                    dimensions = preprocessing_data.get('original_dimensions', 'N/A')
                    processed_dimensions = preprocessing_data.get('processed_dimensions', '224x224')
                    
                    # Get image URLs
                    processed_url = None
                    if jpeg_image.processed_image:
                        processed_path = os.path.join(settings.MEDIA_ROOT, str(jpeg_image.processed_image))
                        if os.path.exists(processed_path):
                            processed_url = jpeg_image.processed_image.url
                    
                    original_image_url = None
                    if jpeg_image.visualization_image:
                        original_image_url = jpeg_image.visualization_image.url
                    else:
                        original_image_url = jpeg_image.image_file.url
                    
                    # Create image data dictionary
                    image_data = {
                        'id': jpeg_image.id,
                        'title': jpeg_image.title,
                        'type': 'JPEG',
                        'original_image': original_image_url,
                        'processed_image': processed_url,
                        'original_dimensions': dimensions,
                        'processed_dimensions': processed_dimensions,
                        'psnr': f"{float(psnr):.2f}" if psnr != 'N/A' else 'N/A',
                        'mse': f"{float(mse):.4f}" if mse != 'N/A' else 'N/A',
                        'ssim': f"{float(ssim):.4f}" if ssim != 'N/A' else 'N/A',
                        'uqi': f"{float(uqi):.4f}" if uqi != 'N/A' else 'N/A',
                        'processing_steps': preprocessing_data.get('processing_steps', [])
                    }
                    processed_images.append(image_data)
            except Exception as e:
                logging.error(f"Error processing JPEG image {jpeg_image.id}: {str(e)}")
                continue
        
        context = {
            'processed_images': processed_images,
            'total_images': len(processed_images),
            'batch_id': batch.batch_id
        }
        
        logging.info(f"Sending {len(processed_images)} processed images to template")
        return render(request, 'preprocessing_app/result.html', context)
        
    except Exception as e:
        logging.error(f"Error in result view: {str(e)}")
        logging.error(traceback.format_exc())
        return render(request, 'preprocessing_app/error.html', {'error': str(e)})

def index(request):
    """
    Render the welcome page
    """
    return render(request, 'preprocessing_app/index.html')



def predict_diagnosis_view(request):
    """
    View for predicting diagnosis using ensemble model on preprocessed images
    """
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from PIL import Image
    import cv2
    import tempfile
    import json
    from datetime import datetime
    
    # Get list of preprocessed images for dropdown
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    available_images = []
    
    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                available_images.append(filename)
    
    context = {
        'available_images': available_images,
        'selected_image': None,
        'predictions': None,
        'error': None,
        'uploaded_image': None
    }
    
    if request.method == 'POST':
        selected_image_name = request.POST.get('selected_image')
        uploaded_file = request.FILES.get('uploaded_image')
        
        image_path = None
        
        # Handle uploaded file
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                image_path = temp_path
                context['uploaded_image'] = uploaded_file.name
                
            except Exception as e:
                context['error'] = f"Error uploading file: {str(e)}"
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'error': context['error']})
                return render(request, 'preprocessing_app/predict_diagnosis.html', context)
        
        # Handle selected existing image
        elif selected_image_name:
            image_path = os.path.join(processed_dir, selected_image_name)
            if not os.path.exists(image_path):
                context['error'] = f"Image {selected_image_name} not found"
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'error': context['error']})
                return render(request, 'preprocessing_app/predict_diagnosis.html', context)
            context['selected_image'] = selected_image_name
        
        # Perform prediction if we have an image
        if image_path:
            try:
                # Load and preprocess image
                img = load_img(image_path, target_size=(224, 224))
                img_array = img_to_array(img)
                
                # Ensure image is RGB (3 channels)
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 1:  # Single channel
                    img_array = np.concatenate([img_array] * 3, axis=2)
                
                # Prepare image for different models
                # VGG16 and ResNet50 use rescale=1./255
                img_vgg_resnet = img_array / 255.0
                img_vgg_resnet = np.expand_dims(img_vgg_resnet, axis=0)
                
                # EfficientNet uses its own preprocessing
                img_efficientnet = efficientnet_preprocess(img_array)
                img_efficientnet = np.expand_dims(img_efficientnet, axis=0)
                
                # Load models
                vgg16_model = load_model('models/vgg16_final_model.keras')
                resnet50_model = load_model('models/resnet50_final_model.keras')
                efficientnet_model = load_model('models/efficientnet_final_model.keras')
                
                # Get predictions
                vgg16_pred_prob = vgg16_model.predict(img_vgg_resnet)[0][0]
                resnet50_pred_prob = resnet50_model.predict(img_vgg_resnet)[0][0]
                efficientnet_pred_prob = efficientnet_model.predict(img_efficientnet)[0][0]
                
                # Convert to binary predictions
                vgg16_pred = 1 if vgg16_pred_prob > 0.5 else 0
                resnet50_pred = 1 if resnet50_pred_prob > 0.5 else 0
                efficientnet_pred = 1 if efficientnet_pred_prob > 0.5 else 0
                
                # Hard voting ensemble
                votes = vgg16_pred + resnet50_pred + efficientnet_pred
                ensemble_pred = 1 if votes >= 2 else 0
                ensemble_prob = (vgg16_pred_prob + resnet50_pred_prob + efficientnet_pred_prob) / 3
                
                def get_confidence(prob, pred):
                    return prob if pred == 1 else 1 - prob

                severity_votes = Counter()

                def build_model_result(probability_value, binary_prediction):
                    label, level = classify_severity(float(probability_value))
                    severity_votes[level] += 1
                    confidence = get_confidence(probability_value, binary_prediction)
                    return {
                        'prediction': label,
                        'severity_level': level,
                        'probability': float(probability_value) * 100,
                        'confidence': float(confidence) * 100
                    }

                vgg16_result = build_model_result(vgg16_pred_prob, vgg16_pred)
                resnet50_result = build_model_result(resnet50_pred_prob, resnet50_pred)
                efficientnet_result = build_model_result(efficientnet_pred_prob, efficientnet_pred)

                ensemble_conf = get_confidence(ensemble_prob, ensemble_pred)
                ensemble_label, ensemble_level = classify_severity(float(ensemble_prob))
                
                predictions = {
                    'vgg16': vgg16_result,
                    'resnet50': resnet50_result,
                    'efficientnet': efficientnet_result,
                    'ensemble': {
                        'prediction': ensemble_label,
                        'severity_level': ensemble_level,
                        'probability': float(ensemble_prob) * 100,
                        'confidence': float(ensemble_conf) * 100,
                        'votes': {
                            'low': severity_votes.get('low', 0),
                            'moderate': severity_votes.get('moderate', 0),
                            'high': severity_votes.get('high', 0)
                        }
                    },
                    'timestamp': datetime.now().isoformat(),
                    'image_name': context['uploaded_image'] or context['selected_image']
                }
                
                context.update({
                    'predictions': predictions
                })
                
                # Clean up temporary file if it was uploaded
                if uploaded_file and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass  # Ignore cleanup errors
                
                # Return JSON for AJAX requests
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': True,
                        'predictions': predictions
                    })
                
                # Store results in session and redirect
                request.session['prediction_results'] = predictions
                return redirect('predict_diagnosis_result')
                
            except Exception as e:
                context['error'] = f"Error during prediction: {str(e)}"
                # Clean up temporary file on error
                if uploaded_file and image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'error': context['error']})
    
    return render(request, 'preprocessing_app/predict_diagnosis.html', context)

def predict_diagnosis_result_view(request):
    """
    Displays the prediction results stored in the session.
    """
    predictions = request.session.get('prediction_results')
    
    if not predictions:
        # Redirect back if no results are found
        return redirect('predict_diagnosis')
        
    context = {
        'predictions': predictions
    }
    
    # Clear the session variable after use
    if 'prediction_results' in request.session:
        del request.session['prediction_results']
        
    return render(request, 'preprocessing_app/predict_diagnosis_result.html', context)
