"""
Configuration Management for Marine Fouling Image Preprocessing Pipeline

This module provides configuration management, validation, and utility functions
for the entire preprocessing pipeline.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass, asdict
from enum import Enum


class ProcessingMode(Enum):
    """Enumeration of processing modes."""
    PREPROCESSING_ONLY = "preprocessing_only"
    AUGMENTATION_ONLY = "augmentation_only"
    FULL_PIPELINE = "full_pipeline"
    CUSTOM = "custom"


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = ".jpg"
    PNG = ".png"
    TIFF = ".tiff"
    BMP = ".bmp"


@dataclass
class AdvancedPreprocessingConfig:
    """Configuration for advanced preprocessing parameters."""
    # Homomorphic filtering
    homomorphic_gamma_h: float = 2.0
    homomorphic_gamma_l: float = 0.25
    homomorphic_c: float = 1.0
    homomorphic_d0: float = 10.0
    
    # White balance correction
    wb_method: str = 'max_white'  # 'gray_world', 'max_white', 'perfect_reflector'
    wb_percentile: int = 95
    wb_clip_limit: float = 0.01
    
    # Dark channel prior
    dcp_patch_size: int = 15
    dcp_omega: float = 0.95
    dcp_t0: float = 0.1
    dcp_guided_eps: float = 0.001
    dcp_guided_radius: int = 60
    
    # Gabor filters
    gabor_frequencies: List[float] = None
    gabor_angles: List[int] = None
    gabor_sigma_x: float = 2.0
    gabor_sigma_y: float = 2.0
    gabor_enhancement_strength: float = 0.3
    
    # Multi-scale enhancement
    pyramid_levels: int = 4
    detail_enhancement_factor: float = 1.5
    edge_enhancement_factor: float = 1.2
    
    # Morphological operations
    morphology_kernel_size: int = 3
    morphology_operations: List[str] = None
    boundary_enhancement_strength: float = 0.4
    
    # Advanced processing control
    enable_advanced_processing: bool = True
    advanced_steps: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.gabor_frequencies is None:
            self.gabor_frequencies = [0.1, 0.3, 0.5]
        
        if self.gabor_angles is None:
            self.gabor_angles = [0, 45, 90, 135]
        
        if self.morphology_operations is None:
            self.morphology_operations = ['opening', 'closing']
        
        if self.advanced_steps is None:
            self.advanced_steps = {
                'homomorphic_filtering': True,
                'advanced_white_balance': True,
                'dark_channel_prior': False,  # Computationally intensive
                'gabor_enhancement': True,
                'multiscale_enhancement': True,
                'morphological_enhancement': True,
                'histogram_specification': False
            }


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing parameters."""
    # CLAHE parameters
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # Retinex parameters
    retinex_sigma_list: List[float] = None
    retinex_low_clip: float = 0.01
    retinex_high_clip: float = 0.99
    
    # Noise reduction
    gaussian_blur_sigma: float = 0.8
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    
    # Enhancement parameters
    gamma_correction: float = 1.2
    sharpening_strength: float = 0.5
    
    # Output parameters
    output_size: Optional[tuple] = None
    normalize_output: bool = True
    
    # Pipeline order
    preprocessing_order: List[str] = None
    
    def __post_init__(self):
        if self.retinex_sigma_list is None:
            self.retinex_sigma_list = [15, 80, 250]
        
        if self.preprocessing_order is None:
            self.preprocessing_order = [
                'resize',
                'noise_reduction',
                'color_correction',
                'lighting_enhancement',
                'contrast_enhancement',
                'sharpening',
                'normalization'
            ]


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters."""
    # Geometric transformations
    rotate_limit: int = 30
    shift_limit: float = 0.1
    scale_limit: float = 0.2
    distortion_limit: float = 0.1
    
    # Color and lighting
    brightness_limit: float = 0.3
    contrast_limit: float = 0.3
    saturation_limit: float = 0.2
    hue_shift_limit: int = 20
    
    # Noise and blur
    gaussian_noise_var: tuple = (10.0, 50.0)
    motion_blur_limit: int = 7
    gaussian_blur_limit: int = 3
    
    # Underwater specific
    water_distortion_alpha: int = 50
    water_distortion_sigma: int = 5
    particles_density: float = 0.1
    color_cast_strength: float = 0.3
    
    # Probabilities
    geometric_prob: float = 0.7
    color_prob: float = 0.8
    noise_prob: float = 0.6
    underwater_prob: float = 0.5


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing parameters."""
    # Processing
    num_workers: int = 4
    batch_size: int = 32
    enable_multiprocessing: bool = True
    
    # File handling
    supported_formats: List[str] = None
    output_format: str = ".jpg"
    output_quality: int = 95
    
    # Processing options
    apply_preprocessing: bool = True
    apply_augmentation: bool = False
    num_augmentations: int = 1
    save_original: bool = False
    
    # Output organization
    create_subdirs: bool = True
    subdir_names: Dict[str, str] = None
    
    # Error handling
    continue_on_error: bool = True
    save_error_log: bool = True
    
    # Progress tracking
    show_progress: bool = True
    save_processing_report: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        if self.subdir_names is None:
            self.subdir_names = {
                'original': 'original',
                'preprocessed': 'preprocessed',
                'augmented': 'augmented'
            }


@dataclass
class PipelineConfig:
    """Main configuration class combining all sub-configurations."""
    preprocessing: PreprocessingConfig = None
    advanced_preprocessing: AdvancedPreprocessingConfig = None
    augmentation: AugmentationConfig = None
    batch_processing: BatchProcessingConfig = None
    
    # General settings
    processing_mode: ProcessingMode = ProcessingMode.FULL_PIPELINE
    logging_level: str = "INFO"
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.advanced_preprocessing is None:
            self.advanced_preprocessing = AdvancedPreprocessingConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.batch_processing is None:
            self.batch_processing = BatchProcessingConfig()


class ConfigManager:
    """
    Manages configuration loading, validation, and saving for the preprocessing pipeline.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            PipelineConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration data
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Parse configuration
        return self._parse_config_dict(config_data)
    
    def save_config(self, config: PipelineConfig, config_path: Union[str, Path], 
                   format: str = 'yaml') -> None:
        """
        Save configuration to a file.
        
        Args:
            config: PipelineConfig object to save
            config_path: Output path for configuration file
            format: Output format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(config)
        
        if format.lower() == 'yaml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def _parse_config_dict(self, config_data: Dict[str, Any]) -> PipelineConfig:
        """Parse configuration dictionary into PipelineConfig object."""
        # Parse sub-configurations
        preprocessing_config = None
        if 'preprocessing' in config_data:
            preprocessing_config = PreprocessingConfig(**config_data['preprocessing'])
        
        advanced_preprocessing_config = None
        if 'advanced_preprocessing' in config_data:
            advanced_preprocessing_config = AdvancedPreprocessingConfig(**config_data['advanced_preprocessing'])
        
        augmentation_config = None
        if 'augmentation' in config_data:
            augmentation_config = AugmentationConfig(**config_data['augmentation'])
        
        batch_processing_config = None
        if 'batch_processing' in config_data:
            batch_processing_config = BatchProcessingConfig(**config_data['batch_processing'])
        
        # Parse main config
        main_config_data = {k: v for k, v in config_data.items() 
                           if k not in ['preprocessing', 'advanced_preprocessing', 'augmentation', 'batch_processing']}
        
        # Handle enum conversion
        if 'processing_mode' in main_config_data:
            if isinstance(main_config_data['processing_mode'], str):
                main_config_data['processing_mode'] = ProcessingMode(main_config_data['processing_mode'])
        
        return PipelineConfig(
            preprocessing=preprocessing_config,
            advanced_preprocessing=advanced_preprocessing_config,
            augmentation=augmentation_config,
            batch_processing=batch_processing_config,
            **main_config_data
        )
    
    def _config_to_dict(self, config: PipelineConfig) -> Dict[str, Any]:
        """Convert PipelineConfig object to dictionary."""
        config_dict = {}
        
        # Add main config parameters
        for key, value in asdict(config).items():
            if key not in ['preprocessing', 'advanced_preprocessing', 'augmentation', 'batch_processing']:
                if isinstance(value, Enum):
                    config_dict[key] = value.value
                else:
                    config_dict[key] = value
        
        # Add sub-configurations
        if config.preprocessing:
            config_dict['preprocessing'] = asdict(config.preprocessing)
        if config.advanced_preprocessing:
            config_dict['advanced_preprocessing'] = asdict(config.advanced_preprocessing)
        if config.augmentation:
            config_dict['augmentation'] = asdict(config.augmentation)
        if config.batch_processing:
            config_dict['batch_processing'] = asdict(config.batch_processing)
        
        return config_dict
    
    def create_default_config(self) -> PipelineConfig:
        """Create a default configuration."""
        return PipelineConfig()
    
    def validate_config(self, config: PipelineConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate preprocessing config
        if config.preprocessing:
            self._validate_preprocessing_config(config.preprocessing)
        
        # Validate augmentation config
        if config.augmentation:
            self._validate_augmentation_config(config.augmentation)
        
        # Validate batch processing config
        if config.batch_processing:
            self._validate_batch_processing_config(config.batch_processing)
        
        return True
    
    def _validate_preprocessing_config(self, config: PreprocessingConfig) -> None:
        """Validate preprocessing configuration."""
        if config.clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
        
        if len(config.clahe_tile_grid_size) != 2:
            raise ValueError("CLAHE tile grid size must be a tuple of 2 integers")
        
        if not all(isinstance(x, int) and x > 0 for x in config.clahe_tile_grid_size):
            raise ValueError("CLAHE tile grid size values must be positive integers")
        
        if not (0 < config.retinex_low_clip < config.retinex_high_clip < 1):
            raise ValueError("Retinex clip values must satisfy 0 < low_clip < high_clip < 1")
        
        if config.gamma_correction <= 0:
            raise ValueError("Gamma correction must be positive")
    
    def _validate_augmentation_config(self, config: AugmentationConfig) -> None:
        """Validate augmentation configuration."""
        if not (0 <= config.geometric_prob <= 1):
            raise ValueError("Geometric probability must be between 0 and 1")
        
        if not (0 <= config.color_prob <= 1):
            raise ValueError("Color probability must be between 0 and 1")
        
        if config.rotate_limit < 0:
            raise ValueError("Rotation limit must be non-negative")
        
        if config.particles_density < 0:
            raise ValueError("Particle density must be non-negative")
    
    def _validate_batch_processing_config(self, config: BatchProcessingConfig) -> None:
        """Validate batch processing configuration."""
        if config.num_workers <= 0:
            raise ValueError("Number of workers must be positive")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not (0 <= config.output_quality <= 100):
            raise ValueError("Output quality must be between 0 and 100")
        
        if config.num_augmentations < 0:
            raise ValueError("Number of augmentations must be non-negative")


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for the entire pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_default_preprocessing_config() -> Dict[str, Any]:
    """Get default preprocessing configuration as dictionary."""
    return asdict(PreprocessingConfig())


def get_default_augmentation_config() -> Dict[str, Any]:
    """Get default augmentation configuration as dictionary."""
    return asdict(AugmentationConfig())


def get_default_batch_config() -> Dict[str, Any]:
    """Get default batch processing configuration as dictionary."""
    return asdict(BatchProcessingConfig())


def create_marine_optimized_config() -> PipelineConfig:
    """
    Create a configuration optimized for marine fouling detection.
    
    Returns:
        Optimized PipelineConfig for underwater images
    """
    # Optimized preprocessing for underwater images
    preprocessing = PreprocessingConfig(
        clahe_clip_limit=4.0,  # Higher clip limit for better contrast
        clahe_tile_grid_size=(6, 6),  # Smaller tiles for local adaptation
        retinex_sigma_list=[20, 100, 300],  # Adjusted for underwater lighting
        gamma_correction=1.3,  # Slightly higher gamma for brightness
        sharpening_strength=0.7,  # More sharpening for clarity
        preprocessing_order=[
            'noise_reduction',
            'advanced_white_balance',
            'homomorphic_filtering',
            'color_correction',
            'lighting_enhancement',
            'contrast_enhancement',
            'gabor_enhancement',
            'multiscale_enhancement',
            'morphological_enhancement',
            'sharpening',
            'resize',
            'normalization'
        ],
        enable_advanced_processing=True,
        advanced_steps={
            'homomorphic_filtering': True,
            'advanced_white_balance': True,
            'dark_channel_prior': False,  # Computationally intensive
            'gabor_enhancement': True,
            'multiscale_enhancement': True,
            'morphological_enhancement': True,
            'histogram_specification': False
        }
    )
    
    # Advanced preprocessing optimized for marine conditions
    advanced_preprocessing = AdvancedPreprocessingConfig(
        # Homomorphic filtering tuned for underwater illumination
        homomorphic_gamma_h=2.2,
        homomorphic_gamma_l=0.3,
        homomorphic_d0=15.0,
        
        # Advanced white balance for underwater color correction
        wb_method='max_white',
        wb_percentile=98,
        
        # Gabor filters optimized for fouling texture detection
        gabor_frequencies=[0.05, 0.15, 0.3, 0.5],
        gabor_angles=[0, 30, 60, 90, 120, 150],
        gabor_enhancement_strength=0.4,
        
        # Multi-scale enhancement for various fouling sizes
        pyramid_levels=5,
        detail_enhancement_factor=1.8,
        edge_enhancement_factor=1.4,
        
        # Morphological operations for boundary enhancement
        morphology_kernel_size=5,
        morphology_operations=['opening', 'closing', 'gradient'],
        boundary_enhancement_strength=0.5,
        
        # Enable key advanced processing steps
        enable_advanced_processing=True,
        advanced_steps={
            'homomorphic_filtering': True,
            'advanced_white_balance': True,
            'dark_channel_prior': False,  # Too computationally intensive for real-time
            'gabor_enhancement': True,
            'multiscale_enhancement': True,
            'morphological_enhancement': True,
            'histogram_specification': False
        }
    )
    
    # Enhanced augmentation for marine conditions
    augmentation = AugmentationConfig(
        water_distortion_alpha=60,  # More distortion for realistic water effects
        particles_density=0.15,  # More particles for marine snow simulation
        color_cast_strength=0.4,  # Stronger underwater color cast
        underwater_prob=0.7,  # Higher probability of underwater effects
        geometric_prob=0.8,  # More geometric variations
    )
    
    # Optimized batch processing
    batch_processing = BatchProcessingConfig(
        enable_multiprocessing=True,
        num_workers=min(8, 6),  # Conservative worker count
        output_format=".jpg",
        output_quality=90,  # High quality for analysis
        save_processing_report=True,
    )
    
    return PipelineConfig(
        preprocessing=preprocessing,
        advanced_preprocessing=advanced_preprocessing,
        augmentation=augmentation,
        batch_processing=batch_processing,
        processing_mode=ProcessingMode.FULL_PIPELINE,
        logging_level="INFO"
    )
