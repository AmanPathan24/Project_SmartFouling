"""
Batch Processing Module for Marine Fouling Image Preprocessing

This module handles batch processing of multiple images through the preprocessing
pipeline with progress tracking, multiprocessing support, and error handling.
"""

import cv2
import numpy as np
from pathlib import Path
import os
from typing import List, Dict, Any, Optional, Callable, Union
from tqdm import tqdm
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime

from .image_preprocessor import ImagePreprocessor
from .data_augmentation import MarineDataAugmenter


class BatchProcessor:
    """
    Handles batch processing of marine fouling images with multiprocessing support.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BatchProcessor.
        
        Args:
            config: Configuration dictionary for batch processing
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize processors
        self.preprocessor = ImagePreprocessor(self.config.get('preprocessing_config'))
        self.augmenter = MarineDataAugmenter(self.config.get('augmentation_config'))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize statistics
        self.processing_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'processing_time': 0.0,
            'errors': []
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for batch processing."""
        return {
            # Processing parameters
            'num_workers': min(mp.cpu_count(), 8),
            'batch_size': 32,
            'enable_multiprocessing': True,
            
            # File handling
            'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
            'output_format': '.jpg',
            'output_quality': 95,
            
            # Processing options
            'apply_preprocessing': True,
            'apply_augmentation': False,
            'num_augmentations': 1,
            'save_original': False,
            
            # Output organization
            'create_subdirs': True,
            'subdir_names': {
                'original': 'original',
                'preprocessed': 'preprocessed',
                'augmented': 'augmented'
            },
            
            # Error handling
            'continue_on_error': True,
            'save_error_log': True,
            
            # Progress tracking
            'show_progress': True,
            'save_processing_report': True,
        }
    
    def discover_images(self, input_path: Union[str, Path]) -> List[Path]:
        """
        Discover all supported image files in the input path.
        
        Args:
            input_path: Path to directory containing images or single image file
            
        Returns:
            List of image file paths
        """
        input_path = Path(input_path)
        image_files = []
        
        if input_path.is_file():
            if input_path.suffix.lower() in self.config['supported_formats']:
                image_files.append(input_path)
        elif input_path.is_dir():
            for ext in self.config['supported_formats']:
                pattern = f"**/*{ext}"
                image_files.extend(input_path.glob(pattern))
                # Also search for uppercase extensions
                pattern = f"**/*{ext.upper()}"
                image_files.extend(input_path.glob(pattern))
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        self.logger.info(f"Discovered {len(image_files)} image files")
        return sorted(image_files)
    
    def create_output_structure(self, output_dir: Path) -> Dict[str, Path]:
        """
        Create output directory structure.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Dictionary mapping subdirectory names to paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        subdirs = {}
        if self.config['create_subdirs']:
            for key, name in self.config['subdir_names'].items():
                subdir_path = output_dir / name
                subdir_path.mkdir(exist_ok=True)
                subdirs[key] = subdir_path
        else:
            subdirs['output'] = output_dir
        
        return subdirs
    
    def process_single_image(self, image_path: Path, output_paths: Dict[str, Path]) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to input image
            output_paths: Dictionary of output directory paths
            
        Returns:
            Processing result dictionary
        """
        result = {
            'image_path': str(image_path),
            'success': False,
            'error': None,
            'output_files': [],
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Load image
            image = self.preprocessor.load_image(str(image_path))
            original_shape = image.shape
            
            # Save original if requested
            if self.config['save_original'] and 'original' in output_paths:
                original_output = output_paths['original'] / f"{image_path.stem}_original{self.config['output_format']}"
                self._save_image(image, original_output)
                result['output_files'].append(str(original_output))
            
            # Apply preprocessing
            if self.config['apply_preprocessing']:
                processed_image = self.preprocessor.process_single_image(image)
                
                if 'preprocessed' in output_paths:
                    processed_output = output_paths['preprocessed'] / f"{image_path.stem}_processed{self.config['output_format']}"
                elif 'output' in output_paths:
                    processed_output = output_paths['output'] / f"{image_path.stem}_processed{self.config['output_format']}"
                else:
                    processed_output = output_paths[list(output_paths.keys())[0]] / f"{image_path.stem}_processed{self.config['output_format']}"
                
                self._save_image(processed_image, processed_output)
                result['output_files'].append(str(processed_output))
            else:
                processed_image = image
            
            # Apply augmentation
            if self.config['apply_augmentation']:
                augmented_images = self.augmenter.augment_batch(
                    [processed_image], 
                    self.config['num_augmentations'],
                    apply_underwater=True
                )
                
                # Skip the first image (original) and save augmented versions
                for i, aug_image in enumerate(augmented_images[1:], 1):
                    if 'augmented' in output_paths:
                        aug_output = output_paths['augmented'] / f"{image_path.stem}_aug_{i}{self.config['output_format']}"
                    elif 'output' in output_paths:
                        aug_output = output_paths['output'] / f"{image_path.stem}_aug_{i}{self.config['output_format']}"
                    else:
                        aug_output = output_paths[list(output_paths.keys())[0]] / f"{image_path.stem}_aug_{i}{self.config['output_format']}"
                    
                    self._save_image(aug_image, aug_output)
                    result['output_files'].append(str(aug_output))
            
            result['success'] = True
            result['original_shape'] = original_shape
            result['num_outputs'] = len(result['output_files'])
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Failed to process {image_path}: {e}")
        
        finally:
            end_time = datetime.now()
            result['processing_time'] = (end_time - start_time).total_seconds()
        
        return result
    
    def _save_image(self, image: np.ndarray, output_path: Path):
        """
        Save image to file with appropriate format conversion.
        
        Args:
            image: Image array to save
            output_path: Output file path
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert image format if necessary
        if len(image.shape) == 3:
            # Convert RGB to BGR for OpenCV
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            save_image = image
        
        # Handle normalization
        if save_image.dtype == np.float32 or save_image.dtype == np.float64:
            save_image = (save_image * 255).astype(np.uint8)
        
        # Save with quality settings
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(
                str(output_path), 
                save_image, 
                [cv2.IMWRITE_JPEG_QUALITY, self.config['output_quality']]
            )
        else:
            cv2.imwrite(str(output_path), save_image)
    
    def process_batch_sequential(self, image_paths: List[Path], output_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
        """
        Process images sequentially (single-threaded).
        
        Args:
            image_paths: List of image file paths
            output_paths: Dictionary of output directory paths
            
        Returns:
            List of processing results
        """
        results = []
        
        progress_bar = tqdm(
            image_paths, 
            desc="Processing images",
            disable=not self.config['show_progress']
        )
        
        for image_path in progress_bar:
            result = self.process_single_image(image_path, output_paths)
            results.append(result)
            
            # Update statistics
            if result['success']:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
                self.processing_stats['errors'].append({
                    'image': str(image_path),
                    'error': result['error']
                })
            
            # Update progress bar
            progress_bar.set_postfix({
                'Success': self.processing_stats['successful'],
                'Failed': self.processing_stats['failed']
            })
            
            if not self.config['continue_on_error'] and not result['success']:
                break
        
        return results
    
    def process_batch_parallel(self, image_paths: List[Path], output_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
        """
        Process images in parallel using multiprocessing.
        
        Args:
            image_paths: List of image file paths
            output_paths: Dictionary of output directory paths
            
        Returns:
            List of processing results
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config['num_workers']) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path, output_paths): path
                for path in image_paths
            }
            
            # Process completed tasks with progress bar
            progress_bar = tqdm(
                as_completed(future_to_path),
                total=len(image_paths),
                desc="Processing images",
                disable=not self.config['show_progress']
            )
            
            for future in progress_bar:
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    if result['success']:
                        self.processing_stats['successful'] += 1
                    else:
                        self.processing_stats['failed'] += 1
                        self.processing_stats['errors'].append({
                            'image': str(image_path),
                            'error': result['error']
                        })
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Success': self.processing_stats['successful'],
                        'Failed': self.processing_stats['failed']
                    })
                    
                except Exception as e:
                    self.logger.error(f"Parallel processing error for {image_path}: {e}")
                    results.append({
                        'image_path': str(image_path),
                        'success': False,
                        'error': str(e),
                        'output_files': [],
                        'processing_time': 0.0
                    })
        
        return results
    
    def process_directory(self, input_path: Union[str, Path], 
                         output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process all images in a directory.
        
        Args:
            input_path: Path to input directory or single image
            output_path: Path to output directory
            
        Returns:
            Processing summary dictionary
        """
        start_time = datetime.now()
        
        # Convert paths
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Discover images
        image_paths = self.discover_images(input_path)
        if not image_paths:
            raise ValueError(f"No supported images found in {input_path}")
        
        # Create output structure
        output_paths = self.create_output_structure(output_path)
        
        # Initialize statistics
        self.processing_stats = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        # Process images
        if self.config['enable_multiprocessing'] and len(image_paths) > 1:
            results = self.process_batch_parallel(image_paths, output_paths)
        else:
            results = self.process_batch_sequential(image_paths, output_paths)
        
        # Calculate final statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        self.processing_stats['processing_time'] = total_time
        
        # Create summary
        summary = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'processing_config': self.config,
            'statistics': self.processing_stats,
            'results': results,
            'timestamp': end_time.isoformat(),
            'avg_processing_time': total_time / len(image_paths) if image_paths else 0
        }
        
        # Save processing report
        if self.config['save_processing_report']:
            report_path = output_path / 'processing_report.json'
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        # Save error log
        if self.config['save_error_log'] and self.processing_stats['errors']:
            error_path = output_path / 'error_log.json'
            with open(error_path, 'w') as f:
                json.dump(self.processing_stats['errors'], f, indent=2)
        
        self.logger.info(f"Batch processing completed: {self.processing_stats['successful']}/{self.processing_stats['total_images']} successful")
        
        return summary
    
    def process_image_list(self, image_paths: List[Union[str, Path]], 
                          output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a specific list of images.
        
        Args:
            image_paths: List of image file paths to process
            output_path: Path to output directory
            
        Returns:
            Processing summary dictionary
        """
        # Convert paths
        image_paths = [Path(p) for p in image_paths]
        output_path = Path(output_path)
        
        # Filter existing images
        existing_paths = [p for p in image_paths if p.exists()]
        if len(existing_paths) != len(image_paths):
            missing = len(image_paths) - len(existing_paths)
            self.logger.warning(f"{missing} image files not found and will be skipped")
        
        if not existing_paths:
            raise ValueError("No valid image files found")
        
        # Create output structure
        output_paths = self.create_output_structure(output_path)
        
        # Initialize statistics
        self.processing_stats = {
            'total_images': len(existing_paths),
            'successful': 0,
            'failed': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        # Process images
        if self.config['enable_multiprocessing'] and len(existing_paths) > 1:
            results = self.process_batch_parallel(existing_paths, output_paths)
        else:
            results = self.process_batch_sequential(existing_paths, output_paths)
        
        # Calculate statistics and create summary
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        summary = {
            'input_files': [str(p) for p in existing_paths],
            'output_path': str(output_path),
            'processing_config': self.config,
            'statistics': self.processing_stats,
            'results': results,
            'timestamp': end_time.isoformat(),
        }
        
        return summary