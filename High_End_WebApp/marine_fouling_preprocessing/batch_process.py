#!/usr/bin/env python3
"""
Batch Processing Script for Marine Fouling Image Preprocessing

This script processes multiple images in parallel using the marine-optimized preprocessing pipeline.

Usage:
    python batch_process.py input_directory output_directory [options]

Examples:
    # Basic batch processing
    python batch_process.py input_images/ processed_images/
    
    # With custom number of workers
    python batch_process.py input_images/ processed_images/ --workers 6
    
    # With progress tracking and verbose output
    python batch_process.py input_images/ processed_images/ --verbose
    
    # Process only specific image types
    python batch_process.py input_images/ processed_images/ --extensions jpg,png,tiff
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import ndimage
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import os
import sys


class MarineBatchPreprocessor:
    """Batch preprocessor for marine fouling images with multiprocessing support."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.config = {
            'clahe_clip_limit': 4.0,  # Marine optimized
            'clahe_tile_grid_size': (6, 6),  # Marine optimized
            'retinex_sigma_list': [20, 100, 300],  # Marine optimized
            'gamma_correction': 1.3,  # Marine optimized
            'sharpening_strength': 0.7,  # Marine optimized
            'gaussian_blur_sigma': 0.8,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            l_channel = clahe.apply(l_channel)
            
            # Merge and convert back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            return clahe.apply(image)
    
    def apply_retinex(self, image: np.ndarray) -> np.ndarray:
        """Apply Multi-Scale Retinex for illumination correction."""
        img_float = image.astype(np.float64) / 255.0
        
        # Avoid log(0) by adding small epsilon
        img_float = np.maximum(img_float, 1e-4)
        
        # Apply MSR
        retinex_output = np.zeros_like(img_float)
        
        for sigma in self.config['retinex_sigma_list']:
            # Gaussian blur
            if len(img_float.shape) == 3:
                blurred = np.zeros_like(img_float)
                for i in range(img_float.shape[2]):
                    blurred[:, :, i] = ndimage.gaussian_filter(img_float[:, :, i], sigma=sigma)
            else:
                blurred = ndimage.gaussian_filter(img_float, sigma=sigma)
            
            # Avoid log(0)
            blurred = np.maximum(blurred, 1e-4)
            
            # Retinex calculation
            retinex_output += np.log(img_float) - np.log(blurred)
        
        retinex_output = retinex_output / len(self.config['retinex_sigma_list'])
        
        # Normalize to 0-255
        retinex_output = np.expm1(retinex_output)  # exp(x) - 1 for better numerical stability
        retinex_output = np.clip(retinex_output, 0, 1) * 255.0
        
        return retinex_output.astype(np.uint8)
    
    def apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for noise reduction."""
        return cv2.bilateralFilter(
            image, 
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
    
    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction for brightness adjustment."""
        gamma = self.config['gamma_correction']
        # Normalize to 0-1, apply gamma, then scale back
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return (corrected * 255.0).astype(np.uint8)
    
    def apply_unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for image sharpening."""
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.0 + self.config['sharpening_strength'], 
                                      blurred, -self.config['sharpening_strength'], 0)
        
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    def process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single image through the marine-optimized pipeline."""
        # Apply processing steps in order
        processed = self.apply_noise_reduction(image)
        processed = self.apply_clahe(processed)
        processed = self.apply_retinex(processed)
        processed = self.apply_gamma_correction(processed)
        processed = self.apply_unsharp_masking(processed)
        
        return processed


def process_single_file(args: Tuple[Path, Path, bool]) -> Dict:
    """Process a single image file. This function is used by multiprocessing."""
    input_path, output_path, verbose = args
    
    try:
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape
        
        # Process image
        preprocessor = MarineBatchPreprocessor()
        processed_image = preprocessor.process_single_image(image)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed image
        save_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), save_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        processing_time = time.time() - start_time
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_shape': original_shape,
            'processed_shape': processed_image.shape,
            'processing_time': processing_time,
            'status': 'success',
            'error': None
        }
        
    except Exception as e:
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_shape': None,
            'processed_shape': None,
            'processing_time': 0,
            'status': 'error',
            'error': str(e)
        }


def find_image_files(input_dir: Path, extensions: List[str]) -> List[Path]:
    """Find all image files in the input directory."""
    image_files = []
    
    for ext in extensions:
        # Handle both with and without dot
        pattern = f"*.{ext.lower().lstrip('.')}"
        image_files.extend(input_dir.rglob(pattern))
        
        # Also check uppercase extensions
        pattern = f"*.{ext.upper().lstrip('.')}"
        image_files.extend(input_dir.rglob(pattern))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort


def create_processing_summary(results: List[Dict], output_dir: Path, start_time: float, end_time: float):
    """Create and save a processing summary report."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    total_processing_time = sum(r['processing_time'] for r in successful)
    avg_processing_time = total_processing_time / len(successful) if successful else 0
    
    summary = {
        'processing_summary': {
            'total_images': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'total_wall_clock_time': end_time - start_time,
            'total_processing_time': total_processing_time,
            'average_processing_time_per_image': avg_processing_time,
            'processing_speed_images_per_second': len(successful) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        },
        'marine_optimization_settings': {
            'clahe_clip_limit': 4.0,
            'clahe_tile_grid_size': [6, 6],
            'retinex_sigma_list': [20, 100, 300],
            'gamma_correction': 1.3,
            'sharpening_strength': 0.7
        },
        'successful_files': successful,
        'failed_files': failed,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save summary to JSON file
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary, summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Batch process marine fouling images with marine-optimized preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic batch processing
    python batch_process.py input_images/ processed_images/
    
    # With custom number of workers
    python batch_process.py input_images/ processed_images/ --workers 6
    
    # With verbose output
    python batch_process.py input_images/ processed_images/ --verbose
    
    # Process only specific file types
    python batch_process.py input_images/ processed_images/ --extensions jpg,png,tiff
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--extensions', default='jpg,jpeg,png,tiff,tif,bmp',
                       help='Comma-separated list of file extensions to process (default: jpg,jpeg,png,tiff,tif,bmp)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Error: Input directory '{input_dir}' does not exist or is not a directory")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    # Find image files
    print(f"🔍 Searching for images in: {input_dir}")
    print(f"📁 Looking for extensions: {', '.join(extensions)}")
    
    image_files = find_image_files(input_dir, extensions)
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return 1
    
    print(f"✅ Found {len(image_files)} image files")
    
    # Setup processing
    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    print(f"🚀 Starting batch processing with {num_workers} workers...")
    print(f"📤 Output directory: {output_dir}")
    
    # Prepare processing arguments
    processing_args = []
    for input_path in image_files:
        # Maintain directory structure in output
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        processing_args.append((input_path, output_path, args.verbose))
    
    # Process images
    start_time = time.time()
    results = []
    
    print("\n🌊 Marine-Optimized Processing Pipeline:")
    print("   - Noise Reduction (Bilateral Filter)")
    print("   - CLAHE Enhancement (Clip: 4.0, Tiles: 6x6)")
    print("   - Multi-Scale Retinex (Scales: [20, 100, 300])")
    print("   - Gamma Correction (γ = 1.3)")
    print("   - Unsharp Masking (Strength: 0.7)")
    print()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_args = {executor.submit(process_single_file, args): args for args in processing_args}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_args):
            result = future.result()
            results.append(result)
            completed += 1
            
            if args.verbose:
                if result['status'] == 'success':
                    print(f"✅ [{completed:3d}/{len(image_files):3d}] {result['input_path']} -> {result['output_path']} ({result['processing_time']:.2f}s)")
                else:
                    print(f"❌ [{completed:3d}/{len(image_files):3d}] {result['input_path']} -> ERROR: {result['error']}")
            else:
                # Show progress without verbose details
                if completed % max(1, len(image_files) // 20) == 0 or completed == len(image_files):
                    progress = completed / len(image_files) * 100
                    print(f"📈 Progress: {completed}/{len(image_files)} ({progress:.1f}%)")
    
    end_time = time.time()
    
    # Create processing summary
    summary, summary_path = create_processing_summary(results, output_dir, start_time, end_time)
    
    # Print final results
    print(f"\n🎉 Batch processing completed!")
    print(f"📊 Results Summary:")
    print(f"   Total images: {summary['processing_summary']['total_images']}")
    print(f"   Successful: {summary['processing_summary']['successful']}")
    print(f"   Failed: {summary['processing_summary']['failed']}")
    print(f"   Success rate: {summary['processing_summary']['success_rate']:.1f}%")
    print(f"   Total time: {summary['processing_summary']['total_wall_clock_time']:.2f}s")
    print(f"   Average per image: {summary['processing_summary']['average_processing_time_per_image']:.2f}s")
    print(f"   Processing speed: {summary['processing_summary']['processing_speed_images_per_second']:.2f} images/second")
    print(f"📄 Detailed report saved to: {summary_path}")
    
    if summary['processing_summary']['failed'] > 0:
        print(f"\n⚠️  {summary['processing_summary']['failed']} images failed to process. Check the summary report for details.")
    
    return 0


if __name__ == "__main__":
    exit(main())