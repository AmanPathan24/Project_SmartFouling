"""
Data Augmentation Module for Marine Fouling Images

This module provides specialized data augmentation techniques for marine fouling
detection, including underwater-specific transformations that simulate various
environmental conditions.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict, Any
import random


class MarineDataAugmenter:
    """
    Specialized data augmentation for marine fouling images.
    
    This class provides augmentations that simulate various underwater
    conditions and imaging artifacts common in marine environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MarineDataAugmenter.
        
        Args:
            config: Configuration dictionary for augmentation parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.transform_pipeline = self._create_augmentation_pipeline()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for data augmentation."""
        return {
            # Geometric transformations
            'rotate_limit': 30,
            'shift_limit': 0.1,
            'scale_limit': 0.2,
            'distortion_limit': 0.1,
            
            # Color and lighting
            'brightness_limit': 0.3,
            'contrast_limit': 0.3,
            'saturation_limit': 0.2,
            'hue_shift_limit': 20,
            
            # Noise and blur
            'gaussian_noise_var': (10.0, 50.0),
            'motion_blur_limit': 7,
            'gaussian_blur_limit': 3,
            
            # Underwater specific
            'water_distortion_alpha': 50,
            'water_distortion_sigma': 5,
            'particles_density': 0.1,
            'color_cast_strength': 0.3,
            
            # Augmentation probabilities
            'geometric_prob': 0.7,
            'color_prob': 0.8,
            'noise_prob': 0.6,
            'underwater_prob': 0.5,
        }
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create the complete augmentation pipeline using albumentations."""
        transforms = []
        
        # Geometric transformations
        if self.config['geometric_prob'] > 0:
            transforms.extend([
                A.Rotate(
                    limit=self.config['rotate_limit'],
                    p=self.config['geometric_prob']
                ),
                A.ShiftScaleRotate(
                    shift_limit=self.config['shift_limit'],
                    scale_limit=self.config['scale_limit'],
                    rotate_limit=0,  # Already handled above
                    border_mode=cv2.BORDER_REFLECT,
                    p=self.config['geometric_prob']
                ),
                A.ElasticTransform(
                    alpha=self.config['water_distortion_alpha'],
                    sigma=self.config['water_distortion_sigma'],
                    alpha_affine=0,
                    border_mode=cv2.BORDER_REFLECT,
                    p=self.config['underwater_prob']
                ),
                A.OpticalDistortion(
                    distort_limit=self.config['distortion_limit'],
                    shift_limit=0.05,
                    border_mode=cv2.BORDER_REFLECT,
                    p=self.config['underwater_prob']
                ),
            ])
        
        # Color and lighting transformations
        if self.config['color_prob'] > 0:
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['brightness_limit'],
                    contrast_limit=self.config['contrast_limit'],
                    p=self.config['color_prob']
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.config['hue_shift_limit'],
                    sat_shift_limit=int(self.config['saturation_limit'] * 100),
                    val_shift_limit=0,
                    p=self.config['color_prob']
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=10,
                    b_shift_limit=-30,  # Simulate underwater blue shift
                    p=self.config['underwater_prob']
                ),
                A.ChannelShuffle(p=0.1),
            ])
        
        # Noise and blur
        if self.config['noise_prob'] > 0:
            transforms.extend([
                A.GaussNoise(
                    var_limit=self.config['gaussian_noise_var'],
                    p=self.config['noise_prob']
                ),
                A.MotionBlur(
                    blur_limit=self.config['motion_blur_limit'],
                    p=self.config['noise_prob'] * 0.3
                ),
                A.GaussianBlur(
                    blur_limit=self.config['gaussian_blur_limit'],
                    p=self.config['noise_prob'] * 0.3
                ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=self.config['noise_prob'] * 0.2
                ),
            ])
        
        # Additional underwater-specific effects
        transforms.extend([
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.1,
                p=self.config['underwater_prob'] * 0.3
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=3,
                src_radius=50,
                p=self.config['underwater_prob'] * 0.2
            ),
        ])
        
        return A.Compose(transforms)
    
    def simulate_underwater_particles(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate suspended particles in water (marine snow, sediment).
        
        Args:
            image: Input image
            
        Returns:
            Image with simulated particles
        """
        height, width = image.shape[:2]
        particle_image = image.copy()
        
        # Calculate number of particles based on density
        num_particles = int(height * width * self.config['particles_density'] / 10000)
        
        for _ in range(num_particles):
            # Random particle properties
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(1, 4)
            brightness = random.randint(150, 255)
            
            # Draw particle (small bright spot)
            cv2.circle(
                particle_image,
                (x, y),
                size,
                (brightness, brightness, brightness),
                -1
            )
            
            # Add slight blur to make it more realistic
            if size > 1:
                roi_x1 = max(0, x - size - 2)
                roi_y1 = max(0, y - size - 2)
                roi_x2 = min(width, x + size + 3)
                roi_y2 = min(height, y + size + 3)
                
                roi = particle_image[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0.5)
                    particle_image[roi_y1:roi_y2, roi_x1:roi_x2] = blurred_roi
        
        return particle_image
    
    def apply_underwater_color_cast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply underwater color cast (blue/green tint).
        
        Args:
            image: Input image
            
        Returns:
            Image with underwater color cast
        """
        if len(image.shape) != 3:
            return image
        
        # Create color cast matrix
        cast_strength = self.config['color_cast_strength']
        
        # Underwater typically reduces red and enhances blue/green
        color_matrix = np.array([
            [1.0 - cast_strength * 0.3, 0.0, cast_strength * 0.1],  # Red
            [0.0, 1.0 + cast_strength * 0.1, cast_strength * 0.1],  # Green  
            [cast_strength * 0.2, cast_strength * 0.1, 1.0 + cast_strength * 0.2]  # Blue
        ])
        
        # Apply color transformation
        image_float = image.astype(np.float32)
        transformed = np.zeros_like(image_float)
        
        for i in range(3):
            for j in range(3):
                transformed[:, :, i] += image_float[:, :, j] * color_matrix[i, j]
        
        # Clip values and convert back to uint8
        transformed = np.clip(transformed, 0, 255)
        return transformed.astype(np.uint8)
    
    def apply_light_attenuation(self, image: np.ndarray, depth_factor: float = 0.3) -> np.ndarray:
        """
        Simulate light attenuation with depth.
        
        Args:
            image: Input image
            depth_factor: Factor representing depth effect (0-1)
            
        Returns:
            Image with simulated depth-based light attenuation
        """
        # Create depth gradient (darker at bottom)
        height, width = image.shape[:2]
        
        # Create gradient mask
        gradient = np.linspace(1.0, 1.0 - depth_factor, height)
        gradient_mask = np.tile(gradient[:, np.newaxis], (1, width))
        
        if len(image.shape) == 3:
            gradient_mask = np.stack([gradient_mask] * 3, axis=2)
        
        # Apply attenuation
        attenuated = image.astype(np.float32) * gradient_mask
        return np.clip(attenuated, 0, 255).astype(np.uint8)
    
    def augment_image(self, image: np.ndarray, apply_underwater: bool = True) -> np.ndarray:
        """
        Apply complete augmentation pipeline to an image.
        
        Args:
            image: Input image
            apply_underwater: Whether to apply underwater-specific effects
            
        Returns:
            Augmented image
        """
        # Apply albumentations pipeline
        augmented = self.transform_pipeline(image=image)['image']
        
        if apply_underwater and random.random() < self.config['underwater_prob']:
            # Apply underwater-specific effects
            if random.random() < 0.4:
                augmented = self.simulate_underwater_particles(augmented)
            
            if random.random() < 0.6:
                augmented = self.apply_underwater_color_cast(augmented)
            
            if random.random() < 0.3:
                depth_factor = random.uniform(0.1, 0.5)
                augmented = self.apply_light_attenuation(augmented, depth_factor)
        
        return augmented
    
    def augment_batch(self, images: List[np.ndarray], 
                     num_augmentations: int = 1,
                     apply_underwater: bool = True) -> List[np.ndarray]:
        """
        Apply augmentation to a batch of images.
        
        Args:
            images: List of input images
            num_augmentations: Number of augmented versions per image
            apply_underwater: Whether to apply underwater-specific effects
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for image in images:
            # Add original image
            augmented_images.append(image)
            
            # Add augmented versions
            for _ in range(num_augmentations):
                aug_image = self.augment_image(image, apply_underwater)
                augmented_images.append(aug_image)
        
        return augmented_images
    
    def create_training_pipeline(self, image_size: Tuple[int, int]) -> A.Compose:
        """
        Create a training-specific augmentation pipeline.
        
        Args:
            image_size: Target image size (height, width)
            
        Returns:
            Albumentations compose object for training
        """
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
                A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=-20),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=5),
            ], p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.7
            ),
            A.ElasticTransform(
                alpha=30,
                sigma=3,
                alpha_affine=0,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3
            ),
            A.Flip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def create_validation_pipeline(self, image_size: Tuple[int, int]) -> A.Compose:
        """
        Create a validation-specific pipeline (minimal augmentation).
        
        Args:
            image_size: Target image size (height, width)
            
        Returns:
            Albumentations compose object for validation
        """
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])