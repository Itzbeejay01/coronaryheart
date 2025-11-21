#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import json
import shutil

# Import processing modules
from jpeg_to_png import process_image as convert_jpeg_to_png
from image_processing import (
    process_image,
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_uqi,
    resize_image,
    apply_non_local_means,
    apply_guided_filter,
    apply_clahe,
    normalize_image
)
from dicom_to_png import convert_dicom_to_png

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("terminal_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class TerminalProcessor:
    """
    Terminal-based image processor for handling medical images
    """
    def __init__(self, input_dir, output_dir, temp_dir, max_workers=4, batch_size=10):
        """
        Initialize the terminal processor
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for processed images
            temp_dir: Directory for temporary files
            max_workers: Maximum number of parallel workers
            batch_size: Number of images to process in each batch
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.converted_dir = Path(output_dir) / 'converted'
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.start_time = None
        
        # Initialize metrics storage
        self.metrics = []
        self.processed_files = []
        self.failed_files = []
        
        logging.info(f"Initialized terminal processor with:")
        logging.info(f"Input directory: {input_dir}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Temporary directory: {temp_dir}")
        logging.info(f"Maximum workers: {max_workers}")
        logging.info(f"Batch size: {batch_size}")
    
    def find_images(self):
        """
        Find all supported image files in the input directory
        """
        supported_extensions = {'.dcm', '.jpg', '.jpeg', '.png'}
        image_files = []
        
        for ext in supported_extensions:
            image_files.extend(list(self.input_dir.glob(f"**/*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"**/*{ext.upper()}")))
        
        logging.info(f"Found {len(image_files)} images to process")
        return image_files
    
    def process_dicom(self, file_path):
        """
        Process a DICOM file
        """
        try:
            # Create output path
            output_filename = f"{file_path.stem}.png"
            output_path = self.converted_dir / output_filename
            
            # Convert DICOM to PNG
            success, message, dicom_info = convert_dicom_to_png(
                str(file_path), 
                str(output_path),
                for_visualization=True
            )
            
            if not success:
                return False, message, None
                
            # Process the converted PNG
            return self.process_png(output_path, dicom_info)
            
        except Exception as e:
            logging.error(f"Error processing DICOM {file_path}: {str(e)}")
            return False, str(e), None
    
    def process_jpeg(self, jpeg_path):
        """
        Process a JPEG image using the following steps:
        1. Convert JPEG to PNG using jpeg_to_png.py
        2. Process the converted image using image_processing.py
        3. Calculate quality metrics using the same metrics as image_processing.py
        """
        try:
            # Step 1: Convert JPEG to PNG using jpeg_to_png.py
            png_path = str(self.converted_dir / f"{Path(jpeg_path).stem}.png")
            convert_jpeg_to_png(jpeg_path, png_path)
            logging.info(f"Converted JPEG to PNG: {png_path}")
            
            # Step 2: Process the converted PNG using image_processing.py
            output_path = str(self.output_dir / f"{Path(jpeg_path).stem}_enhanced.png")
            success, message, processing_info = process_image(png_path, output_path)
            
            # If image processing failed, return the error
            if not success:
                return False, f"Image processing failed: {message}", {}
            
            # The metrics are already calculated in process_image, so we just return the processing info
            return True, "JPEG processing completed successfully", processing_info
            
        except Exception as e:
            logging.error(f"Error processing JPEG {jpeg_path}: {str(e)}")
            return False, f"Failed to process JPEG: {str(e)}", {}
    
    def process_png(self, file_path, additional_info=None):
        """
        Process a PNG file
        """
        try:
            # Create output path for enhanced image
            output_filename = f"{file_path.stem}_enhanced.png"
            output_path = self.output_dir / output_filename
            
            # Process image with image_processing.py
            success, message, processing_info = process_image(
                str(file_path), 
                str(output_path)
            )
            
            # If image processing failed, return the error
            if not success:
                return False, message, None
            
            # Add file information to the processing info
            processing_info.update({
                'filename': file_path.name,
                'original_path': str(file_path),
                'enhanced_path': str(output_path),
                'processing_success': True,
                'processing_message': message
            })
            
            # Add any additional info from DICOM
            if additional_info:
                processing_info.update(additional_info)
                
            return True, message, processing_info
            
        except Exception as e:
            logging.error(f"Error processing PNG {file_path}: {str(e)}")
            return False, str(e), None
    
    def process_file(self, file_path):
        """
        Process a single file based on its extension
        """
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.dcm':
                success, message, metrics = self.process_dicom(file_path)
            elif extension in ['.jpg', '.jpeg']:
                success, message, metrics = self.process_jpeg(str(file_path))
            elif extension == '.png':
                success, message, metrics = self.process_png(file_path)
            else:
                return {
                    'filename': file_path.name,
                    'original_path': str(file_path),
                    'processing_success': False,
                    'processing_message': f"Unsupported file type: {extension}"
                }
            
            if success and metrics:
                return metrics
            else:
                return {
                    'filename': file_path.name,
                    'original_path': str(file_path),
                    'processing_success': False,
                    'processing_message': message
                }
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return {
                'filename': file_path.name,
                'original_path': str(file_path),
                'processing_success': False,
                'processing_message': str(e)
            }
    
    def process_batch(self, batch_files):
        """
        Process a batch of files in parallel
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file_path): file_path for file_path in batch_files}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                              total=len(batch_files), 
                              desc="Processing batch"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    results.append({
                        'filename': file_path.name,
                        'original_path': str(file_path),
                        'processing_success': False,
                        'processing_message': str(e)
                    })
        
        return results
    
    def run(self):
        """
        Run the terminal processing
        """
        # Find all images
        image_files = self.find_images()
        
        if not image_files:
            logging.error(f"No images found in {self.input_dir}")
            return
        
        # Process in batches
        total_batches = (len(image_files) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            logging.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
            
            # Process batch
            batch_results = self.process_batch(batch_files)
            
            # Update metrics
            for result in batch_results:
                if result.get('processing_success', False):
                    self.processed_files.append(result)
                else:
                    self.failed_files.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Log progress
            elapsed_time = time.time() - self.start_time
            processed_count = len(self.processed_files)
            failed_count = len(self.failed_files)
            remaining_count = len(image_files) - processed_count - failed_count
            
            logging.info(f"Progress: {processed_count}/{len(image_files)} processed, {failed_count} failed, {remaining_count} remaining")
            logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            
            # Estimate remaining time
            if processed_count > 0:
                time_per_file = elapsed_time / processed_count
                estimated_remaining = time_per_file * remaining_count
                logging.info(f"Estimated time remaining: {estimated_remaining:.2f} seconds ({estimated_remaining/60:.2f} minutes)")
        
        # Final save and report
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """
        Save current results to JSON
        """
        if self.processed_files:
            with open(self.output_dir / "processed_images.json", "w") as f:
                json.dump(self.processed_files, f, indent=2)
            
        if self.failed_files:
            with open(self.output_dir / "failed_images.json", "w") as f:
                json.dump(self.failed_files, f, indent=2)
    
    def generate_report(self):
        """
        Generate a summary report
        """
        total_time = time.time() - self.start_time
        
        report = {
            "Total images found": len(self.processed_files) + len(self.failed_files),
            "Successfully processed": len(self.processed_files),
            "Failed": len(self.failed_files),
            "Success rate": f"{(len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100):.2f}%",
            "Total processing time (seconds)": f"{total_time:.2f}",
            "Total processing time (minutes)": f"{total_time/60:.2f}",
            "Total processing time (hours)": f"{total_time/3600:.2f}",
            "Average time per image (seconds)": f"{total_time / (len(self.processed_files) + len(self.failed_files)):.2f}"
        }
        
        # Save report
        with open(self.output_dir / "processing_report.txt", "w") as f:
            f.write("TERMINAL PROCESSING REPORT\n")
            f.write("========================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
            
            # Add quality metrics summary if available
            if self.processed_files:
                f.write("\nQUALITY METRICS SUMMARY\n")
                f.write("======================\n\n")
                
                # Calculate average metrics
                metrics_sum = {}
                for file_metrics in self.processed_files:
                    for key, value in file_metrics.items():
                        if key not in ['filename', 'original_path', 'enhanced_path', 'processing_success', 'processing_message']:
                            if key not in metrics_sum:
                                metrics_sum[key] = 0
                            metrics_sum[key] += value
                
                for key, value in metrics_sum.items():
                    avg_value = value / len(self.processed_files)
                    f.write(f"Average {key}: {avg_value:.4f}\n")
        
        logging.info(f"Report saved to {self.output_dir / 'processing_report.txt'}")


def main():
    """
    Main function to run the terminal processor
    """
    parser = argparse.ArgumentParser(description="Terminal processor for medical images")
    parser.add_argument("--input", default="media/images", help="Input directory containing images (default: media/images)")
    parser.add_argument("--output", default="media/processed", help="Output directory for processed images (default: media/processed)")
    parser.add_argument("--temp", default="media/temp", help="Temporary directory for intermediate files (default: media/temp)")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of images to process in each batch")
    
    args = parser.parse_args()
    
    # Get the workspace directory (parent of the script directory)
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    workspace_dir = script_dir.parent
    
    # Create absolute paths
    input_dir = workspace_dir / args.input
    output_dir = workspace_dir / args.output
    temp_dir = workspace_dir / args.temp
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Temp directory: {temp_dir}")
    
    processor = TerminalProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        max_workers=args.workers,
        batch_size=args.batch_size
    )
    
    processor.start_time = time.time()
    processor.run()


if __name__ == "__main__":
    main() 