"""
Marine Fouling Image Preprocessing Pipeline

This module provides a comprehensive image preprocessing pipeline specifically designed
for marine fouling detection and classification. It handles various underwater imaging
challenges including poor lighting, low contrast, and noise.

Author: Marine Fouling Detection System
"""

import cv2
import numpy as np
from skimage import exposure, restoration, filters, morphology
from skimage.color import rgb2gray, gray2rgb
from scipy import ndimage
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import json

from .advanced_preprocessing import AdvancedPreprocessor

class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline for marine fouling detection.
    
    This class provides methods to enhance underwater images through various
    preprocessing techniques including CLAHE, Retinex, noise reduction,
    and data augmentation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ImagePreprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize advanced preprocessor
        advanced_config = config.get('advanced_config', {}) if config else {}
        self.advanced_preprocessor = AdvancedPreprocessor(advanced_config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for preprocessing parameters."""
        return {
            # CLAHE parameters
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid_size': (8, 8),
            
            # Retinex parameters
            'retinex_sigma_list': [15, 80, 250],
            'retinex_low_clip': 0.01,
            'retinex_high_clip': 0.99,
            
            # Noise reduction parameters
            'gaussian_blur_sigma': 0.8,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            
            # Enhancement parameters
            'gamma_correction': 1.2,
            'sharpening_strength': 0.5,
            
            # Output parameters
            'output_size': None,  # (width, height) or None to keep original
            'normalize_output': True,
            
            # Pipeline order with advanced techniques
            'preprocessing_order': [
                'resize',
                'noise_reduction',
                'advanced_white_balance',
                'homomorphic_filtering',
                'dark_channel_prior',
                'color_correction',
                'lighting_enhancement',
                'contrast_enhancement',
                'gabor_enhancement',
                'multiscale_enhancement', 
                'morphological_enhancement',
                'sharpening',
                'normalization'
            ],
            
            # Advanced processing options
            'enable_advanced_processing': True,
            'advanced_steps': {
                'homomorphic_filtering': True,
                'advanced_white_balance': True,
                'dark_channel_prior': False,  # Computationally intensive
                'gabor_enhancement': True,
                'multiscale_enhancement': True,
                'morphological_enhancement': True,
                'histogram_specification': False
            }
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.logger.info(f"Loaded image with shape: {image.shape}")
        return image
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate input image format and properties.
        
        Args:
            image: Input image array
            
        Returns:
            True if image is valid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError("Color image must have 1, 3, or 4 channels")
        
        return True
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        This is particularly useful for underwater images with poor lighting
        and low contrast regions.
        
        Args:
            image: Input image
            
        Returns:
            CLAHE enhanced image
        """
        # Convert to LAB color space for better results
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            enhanced_image = clahe.apply(image)
        
        return enhanced_image
    
    def apply_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Multi-Scale Retinex (MSR) for illumination correction.
        
        This technique is excellent for enhancing underwater images with
        non-uniform illumination and color distortion.
        
        Args:
            image: Input image
            
        Returns:
            Retinex enhanced image
        """
        # Convert to float
        img_float = image.astype(np.float64) / 255.0
        
        if len(img_float.shape) == 2:
            # Grayscale image
            enhanced = self._single_scale_retinex(img_float, self.config['retinex_sigma_list'])
        else:
            # Color image - process each channel
            enhanced = np.zeros_like(img_float)
            for i in range(img_float.shape[2]):
                enhanced[:, :, i] = self._multi_scale_retinex(
                    img_float[:, :, i], 
                    self.config['retinex_sigma_list']
                )
        
        # Normalize and convert back to uint8
        enhanced = self._normalize_retinex_output(enhanced)
        return (enhanced * 255).astype(np.uint8)
    
    def _multi_scale_retinex(self, image: np.ndarray, sigma_list: List[float]) -> np.ndarray:
        """Apply multi-scale retinex to a single channel."""
        retinex = np.zeros_like(image)
        
        for sigma in sigma_list:
            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            # Avoid log(0) by adding small epsilon
            blurred = np.maximum(blurred, 1e-6)
            image_safe = np.maximum(image, 1e-6)
            # Compute log difference
            retinex += np.log(image_safe) - np.log(blurred)
        
        return retinex / len(sigma_list)
    
    def _single_scale_retinex(self, image: np.ndarray, sigma_list: List[float]) -> np.ndarray:
        """Apply single scale retinex (for grayscale)."""
        return self._multi_scale_retinex(image, sigma_list)
    
    def _normalize_retinex_output(self, retinex: np.ndarray) -> np.ndarray:
        """Normalize retinex output to [0, 1] range."""
        # Clip extreme values
        low_clip = np.percentile(retinex, self.config['retinex_low_clip'] * 100)
        high_clip = np.percentile(retinex, self.config['retinex_high_clip'] * 100)
        
        retinex_clipped = np.clip(retinex, low_clip, high_clip)
        
        # Normalize to [0, 1]
        retinex_norm = (retinex_clipped - retinex_clipped.min()) / (
            retinex_clipped.max() - retinex_clipped.min() + 1e-6
        )
        
        return retinex_norm
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques.
        
        Combines bilateral filtering and gentle Gaussian blur to reduce
        noise while preserving edges.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(
                image,
                self.config['bilateral_d'],
                self.config['bilateral_sigma_color'],
                self.config['bilateral_sigma_space']
            )
        else:
            denoised = cv2.bilateralFilter(
                image,
                self.config['bilateral_d'],
                self.config['bilateral_sigma_color'],
                self.config['bilateral_sigma_space']
            )
        
        # Apply gentle Gaussian blur for additional smoothing
        denoised = cv2.GaussianBlur(
            denoised, 
            (3, 3), 
            self.config['gaussian_blur_sigma']
        )
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using histogram stretching.
        
        Args:
            image: Input image
            
        Returns:
            Contrast enhanced image
        """
        if len(image.shape) == 3:
            # For color images, work in YUV space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            # For grayscale images
            enhanced = cv2.equalizeHist(image)
        
        return enhanced
    
    def correct_color(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction for underwater images.
        
        Underwater images often have blue/green tint due to light absorption.
        This method attempts to correct the color balance.
        
        Args:
            image: Input image
            
        Returns:
            Color corrected image
        """
        if len(image.shape) != 3:
            return image
        
        # Apply white balance using gray world assumption
        mean_b = np.mean(image[:, :, 2])  # Blue channel (underwater bias)
        mean_g = np.mean(image[:, :, 1])  # Green channel
        mean_r = np.mean(image[:, :, 0])  # Red channel (often diminished)
        
        # Calculate scaling factors
        gray_mean = (mean_r + mean_g + mean_b) / 3
        
        scale_r = gray_mean / mean_r if mean_r > 0 else 1.0
        scale_g = gray_mean / mean_g if mean_g > 0 else 1.0
        scale_b = gray_mean / mean_b if mean_b > 0 else 1.0
        
        # Apply corrections with limits to avoid oversaturation
        corrected = image.copy().astype(np.float32)
        corrected[:, :, 0] = np.clip(corrected[:, :, 0] * min(scale_r, 2.0), 0, 255)
        corrected[:, :, 1] = np.clip(corrected[:, :, 1] * min(scale_g, 1.5), 0, 255)
        corrected[:, :, 2] = np.clip(corrected[:, :, 2] * min(scale_b, 0.8), 0, 255)
        
        return corrected.astype(np.uint8)
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Apply gamma correction for brightness adjustment.
        
        Args:
            image: Input image
            gamma: Gamma value (if None, uses config value)
            
        Returns:
            Gamma corrected image
        """
        if gamma is None:
            gamma = self.config['gamma_correction']
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking for image sharpening.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Create sharpened image
        sharpened = cv2.addWeighted(
            image, 1.0 + self.config['sharpening_strength'],
            blurred, -self.config['sharpening_strength'],
            0
        )
        
        return sharpened
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            target_size: Target size as (width, height)
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.config['output_size']
        
        if target_size is None:
            return image
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if not self.config['normalize_output']:
            return image
        
        return image.astype(np.float32) / 255.0
    
    def process_single_image(self, image: np.ndarray, 
                           custom_pipeline: Optional[List[str]] = None) -> np.ndarray:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image: Input image
            custom_pipeline: Custom pipeline order (if None, uses default)
            
        Returns:
            Processed image
        """
        self.validate_image(image)
        processed = image.copy()
        
        pipeline_order = custom_pipeline or self.config['preprocessing_order']
        
        for step in pipeline_order:
            if step == 'resize':
                processed = self.resize_image(processed)
            elif step == 'noise_reduction':
                processed = self.reduce_noise(processed)
            elif step == 'advanced_white_balance' and self._is_advanced_step_enabled('advanced_white_balance'):
                processed = self.advanced_preprocessor.advanced_white_balance_correction(processed)
            elif step == 'homomorphic_filtering' and self._is_advanced_step_enabled('homomorphic_filtering'):
                processed = self.advanced_preprocessor.apply_homomorphic_filtering(processed)
            elif step == 'dark_channel_prior' and self._is_advanced_step_enabled('dark_channel_prior'):
                processed = self.advanced_preprocessor.apply_dark_channel_prior(processed)
            elif step == 'color_correction':
                processed = self.correct_color(processed)
            elif step == 'lighting_enhancement':
                # Apply both CLAHE and Retinex for comprehensive lighting enhancement
                processed = self.apply_clahe(processed)
                processed = self.apply_retinex(processed)
            elif step == 'contrast_enhancement':
                processed = self.enhance_contrast(processed)
            elif step == 'gabor_enhancement' and self._is_advanced_step_enabled('gabor_enhancement'):
                processed = self.advanced_preprocessor.apply_gabor_enhancement(processed)
            elif step == 'multiscale_enhancement' and self._is_advanced_step_enabled('multiscale_enhancement'):
                processed = self.advanced_preprocessor.apply_multiscale_enhancement(processed)
            elif step == 'morphological_enhancement' and self._is_advanced_step_enabled('morphological_enhancement'):
                processed = self.advanced_preprocessor.apply_morphological_enhancement(processed)
            elif step == 'gamma_correction':
                processed = self.apply_gamma_correction(processed)
            elif step == 'sharpening':
                processed = self.sharpen_image(processed)
            elif step == 'histogram_specification' and self._is_advanced_step_enabled('histogram_specification'):
                processed = self.advanced_preprocessor.histogram_specification(processed)
            elif step == 'normalization':
                processed = self.normalize_image(processed)
            else:
                self.logger.warning(f"Unknown preprocessing step: {step}")
        
        return processed
    
    def _is_advanced_step_enabled(self, step_name: str) -> bool:
        """Check if an advanced processing step is enabled."""
        if not self.config.get('enable_advanced_processing', True):
            return False
        
        advanced_steps = self.config.get('advanced_steps', {})
        return advanced_steps.get(step_name, False)
