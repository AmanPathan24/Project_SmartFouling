"""
Marine Fouling Image Preprocessing Pipeline

A comprehensive image preprocessing pipeline designed specifically for marine fouling
detection and classification. Handles various underwater imaging challenges including
poor lighting, low contrast, noise, and provides specialized data augmentation.

Main Components:
- ImagePreprocessor: Core preprocessing with CLAHE, Retinex, noise reduction, etc.
- MarineDataAugmenter: Specialized augmentation for underwater images
- BatchProcessor: Efficient batch processing with multiprocessing support
- ConfigManager: Configuration management and validation

Example usage:
    from marine_fouling_preprocessing import ImagePreprocessor, BatchProcessor
    
    # Single image processing
    processor = ImagePreprocessor()
    processed_image = processor.process_single_image(image)
    
    # Batch processing
    batch_processor = BatchProcessor()
    batch_processor.process_directory('input_dir', 'output_dir')
"""

from .image_preprocessor import ImagePreprocessor
from .advanced_preprocessing import AdvancedPreprocessor
from .data_augmentation import MarineDataAugmenter
from .batch_processor import BatchProcessor
from .config import (
    ConfigManager, 
    PipelineConfig, 
    PreprocessingConfig,
    AdvancedPreprocessingConfig,
    AugmentationConfig,
    BatchProcessingConfig,
    ProcessingMode,
    setup_logging,
    create_marine_optimized_config,
    get_default_preprocessing_config,
    get_default_augmentation_config,
    get_default_batch_config
)

__version__ = "1.0.0"
__author__ = "Marine Fouling Detection Team"

__all__ = [
    # Core classes
    'ImagePreprocessor',
    'AdvancedPreprocessor',
    'MarineDataAugmenter', 
    'BatchProcessor',
    
    # Configuration
    'ConfigManager',
    'PipelineConfig',
    'PreprocessingConfig',
    'AdvancedPreprocessingConfig',
    'AugmentationConfig', 
    'BatchProcessingConfig',
    'ProcessingMode',
    
    # Utility functions
    'setup_logging',
    'create_marine_optimized_config',
    'get_default_preprocessing_config',
    'get_default_augmentation_config',
    'get_default_batch_config',
]
