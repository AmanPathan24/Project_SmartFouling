"""
Preprocessing Service for Marine Biofouling Detection
Integrates the marine fouling preprocessing pipeline
"""

import cv2
import numpy as np
from PIL import Image
import asyncio
import logging
from typing import Optional, Dict, Any
import sys
import os

# Add the preprocessing module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marine_fouling_preprocessing'))

try:
    # Try to import from the marine preprocessing directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marine_fouling_preprocessing'))
    
    from src.image_preprocessor import ImagePreprocessor
    from src.config import create_marine_optimized_config
except ImportError as e:
    logging.warning(f"Advanced preprocessing modules not available: {e}")
    ImagePreprocessor = None
    create_marine_optimized_config = None

logger = logging.getLogger(__name__)

class PreprocessingService:
    """Service for preprocessing marine biofouling images"""
    
    def __init__(self):
        self.preprocessor = None
        self.config = None
        self._initialize_preprocessor()
    
    def _initialize_preprocessor(self):
        """Initialize the preprocessing pipeline"""
        try:
            if create_marine_optimized_config is None:
                logger.warning("Preprocessing modules not available, using fallback")
                self._setup_fallback_preprocessor()
                return
            
            # Create marine-optimized configuration
            self.config = create_marine_optimized_config()
            
            # Initialize preprocessor with configuration
            preprocessing_config = self.config.preprocessing.__dict__
            self.preprocessor = ImagePreprocessor(preprocessing_config)
            
            logger.info("Marine preprocessing pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize preprocessing pipeline: {e}")
            self._setup_fallback_preprocessor()
    
    def _setup_fallback_preprocessor(self):
        """Setup fallback preprocessing if main pipeline fails"""
        self.preprocessor = FallbackPreprocessor()
        logger.info("Using fallback preprocessing pipeline")
    
    async def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess a single image for marine biofouling detection
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        try:
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Ensure RGB format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB format
                pass
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA format, convert to RGB
                image_array = image_array[:, :, :3]
            else:
                # Grayscale, convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Run preprocessing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            processed_array = await loop.run_in_executor(
                None, self._process_image_sync, image_array
            )
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(processed_array.astype(np.uint8))
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            # Return original image if preprocessing fails
            return image
    
    def _process_image_sync(self, image_array: np.ndarray) -> np.ndarray:
        """Synchronous image processing (runs in thread pool)"""
        try:
            if self.preprocessor and hasattr(self.preprocessor, 'process_single_image'):
                # Use the advanced preprocessing pipeline
                return self.preprocessor.process_single_image(image_array)
            else:
                # Use fallback preprocessing
                return self.preprocessor.process_image(image_array)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return original image if processing fails
            return image_array
    
    async def batch_preprocess(self, images: list) -> list:
        """
        Preprocess multiple images in batch
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of preprocessed PIL Image objects
        """
        tasks = [self.preprocess_image(image) for image in images]
        return await asyncio.gather(*tasks)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline"""
        if self.config:
            return {
                "pipeline_type": "marine_optimized",
                "config": self.config.preprocessing.__dict__,
                "available_steps": self.config.preprocessing.preprocessing_order
            }
        else:
            return {
                "pipeline_type": "fallback",
                "available_steps": ["resize", "enhance_contrast", "noise_reduction"]
            }

class FallbackPreprocessor:
    """Fallback preprocessing when main pipeline is not available"""
    
    def __init__(self):
        self.target_size = (512, 512)
    
    def process_image(self, image_array: np.ndarray) -> np.ndarray:
        """Basic preprocessing pipeline"""
        try:
            # Resize to target size
            resized = cv2.resize(image_array, self.target_size)
            
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply slight Gaussian blur for noise reduction
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply gamma correction
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            
            return enhanced.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Fallback preprocessing failed: {e}")
            return image_array
