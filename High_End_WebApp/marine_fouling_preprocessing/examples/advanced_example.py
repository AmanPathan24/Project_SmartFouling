#!/usr/bin/env python3
"""
Advanced Preprocessing Example: Marine Fouling Image Enhancement

This script demonstrates the advanced preprocessing techniques implemented for
marine fouling detection, including homomorphic filtering, dark channel prior,
Gabor filters, and multi-scale enhancement.

Usage:
    python advanced_example.py input_image.jpg output_directory/ [options]
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from image_preprocessor import ImagePreprocessor
from advanced_preprocessing import AdvancedPreprocessor
from config import create_marine_optimized_config, ConfigManager, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate advanced marine image preprocessing techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Advanced Preprocessing Techniques Demonstrated:
  - Homomorphic Filtering: Separates illumination and reflectance components
  - Advanced White Balance: Multiple methods for underwater color correction  
  - Dark Channel Prior: Removes haze and scattering effects
  - Gabor Enhancement: Multi-orientation texture enhancement for fouling patterns
  - Multi-scale Enhancement: Pyramid-based detail enhancement at multiple scales
  - Morphological Enhancement: Boundary and shape enhancement operations

Examples:
    # Full advanced processing pipeline
    python advanced_example.py input.jpg output/ --full-pipeline

    # Individual technique demonstration
    python advanced_example.py input.jpg output/ --show-individual --create-grid

    # Compare basic vs advanced processing
    python advanced_example.py input.jpg output/ --comparison

    # Use configuration file
    python advanced_example.py input.jpg output/ --config ../config_examples/marine_advanced.yaml
        """
    )
    
    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument('output_dir', help='Output directory for processed images')
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Apply complete advanced preprocessing pipeline')
    parser.add_argument('--show-individual', action='store_true',
                       help='Show effects of individual advanced techniques')
    parser.add_argument('--comparison', action='store_true',
                       help='Compare basic vs advanced processing')
    parser.add_argument('--create-grid', action='store_true',
                       help='Create comparison grids')
    
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--enable-dcp', action='store_true',
                       help='Enable Dark Channel Prior (computationally intensive)')
    
    parser.add_argument('--timing', action='store_true',
                       help='Show processing time for each technique')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load and validate input image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Loaded image: {Path(image_path).name}")
    print(f"   Shape: {image.shape}")
    print(f"   Data type: {image.dtype}")
    print(f"   Value range: [{image.min()}, {image.max()}]")
    
    return image


def save_image(image: np.ndarray, path: Path, quality: int = 95):
    """Save image to file."""
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        save_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        save_image_bgr = image
    
    # Handle normalized images
    if save_image_bgr.dtype in [np.float32, np.float64]:
        save_image_bgr = (save_image_bgr * 255).astype(np.uint8)
    
    cv2.imwrite(str(path), save_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def demonstrate_individual_techniques(original_image: np.ndarray, 
                                    advanced_processor: AdvancedPreprocessor,
                                    output_dir: Path, timing: bool = False):
    """Demonstrate individual advanced preprocessing techniques."""
    print("\n🔬 Demonstrating Individual Advanced Techniques:")
    
    techniques = [
        ('original', 'Original Image', lambda img: img),
        ('homomorphic', 'Homomorphic Filtering', 
         lambda img: advanced_processor.apply_homomorphic_filtering(img)),
        ('white_balance', 'Advanced White Balance', 
         lambda img: advanced_processor.advanced_white_balance_correction(img)),
        ('gabor', 'Gabor Enhancement', 
         lambda img: advanced_processor.apply_gabor_enhancement(img)),
        ('multiscale', 'Multi-scale Enhancement', 
         lambda img: advanced_processor.apply_multiscale_enhancement(img)),
        ('morphological', 'Morphological Enhancement', 
         lambda img: advanced_processor.apply_morphological_enhancement(img)),
    ]
    
    processed_images = []
    timing_results = {}
    
    for technique_name, display_name, technique_func in techniques:
        print(f"   Processing: {display_name}...")
        
        start_time = time.time()
        try:
            processed_img = technique_func(original_image.copy())
            processing_time = time.time() - start_time
            
            if timing:
                timing_results[technique_name] = processing_time
                print(f"     ⏱️  Processing time: {processing_time:.2f}s")
            
            # Save individual result
            save_path = output_dir / f"{technique_name}.jpg"
            save_image(processed_img, save_path)
            print(f"     💾 Saved: {save_path}")
            
            processed_images.append((display_name, processed_img))
            
        except Exception as e:
            print(f"     ❌ Error: {str(e)}")
            continue
    
    if timing:
        print(f"\n⏱️  Timing Summary:")
        for technique, time_taken in timing_results.items():
            print(f"   {technique}: {time_taken:.2f}s")
    
    return processed_images


def demonstrate_dark_channel_prior(original_image: np.ndarray,
                                  advanced_processor: AdvancedPreprocessor,
                                  output_dir: Path, timing: bool = False):
    """Demonstrate Dark Channel Prior (computationally intensive)."""
    print("\n🌫️  Applying Dark Channel Prior for Haze Removal...")
    print("   ⚠️  This is computationally intensive and may take a while...")
    
    start_time = time.time()
    try:
        dehazed_image = advanced_processor.apply_dark_channel_prior(original_image)
        processing_time = time.time() - start_time
        
        if timing:
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        # Save result
        save_path = output_dir / "dark_channel_prior.jpg"
        save_image(dehazed_image, save_path)
        print(f"   💾 Saved: {save_path}")
        
        return dehazed_image
        
    except Exception as e:
        print(f"   ❌ Error applying Dark Channel Prior: {str(e)}")
        return None


def compare_basic_vs_advanced(original_image: np.ndarray, 
                            basic_processor: ImagePreprocessor,
                            advanced_processor: ImagePreprocessor,
                            output_dir: Path):
    """Compare basic vs advanced preprocessing."""
    print("\n⚖️  Comparing Basic vs Advanced Preprocessing:")
    
    # Process with basic pipeline (no advanced features)
    print("   Processing with basic pipeline...")
    basic_config = basic_processor.config.copy()
    basic_config['enable_advanced_processing'] = False
    basic_processor.config = basic_config
    basic_result = basic_processor.process_single_image(original_image)
    
    # Process with advanced pipeline
    print("   Processing with advanced pipeline...")
    advanced_result = advanced_processor.process_single_image(original_image)
    
    # Save comparison results
    save_image(basic_result, output_dir / "basic_processing.jpg")
    save_image(advanced_result, output_dir / "advanced_processing.jpg")
    
    print(f"   💾 Basic result saved: {output_dir / 'basic_processing.jpg'}")
    print(f"   💾 Advanced result saved: {output_dir / 'advanced_processing.jpg'}")
    
    return basic_result, advanced_result


def create_comparison_grid(images_dict: dict, output_path: Path, title: str = "Comparison"):
    """Create a comparison grid showing multiple processing results."""
    print(f"\n📊 Creating {title.lower()} grid...")
    
    n_images = len(images_dict)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, (name, image) in enumerate(images_dict.items()):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(name, fontsize=12)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Grid saved: {output_path}")


def main():
    """Main function for advanced preprocessing demonstration."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    print("=== Advanced Marine Fouling Image Preprocessing Demo ===\n")
    
    try:
        # Load input image
        print("1. Loading input image...")
        original_image = load_image(args.input_image)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create configuration
        if args.config:
            print(f"2. Loading configuration from: {args.config}")
            config_manager = ConfigManager()
            config = config_manager.load_config(args.config)
        else:
            print("2. Creating marine-optimized advanced configuration...")
            config = create_marine_optimized_config()
        
        # Enable Dark Channel Prior if requested
        if args.enable_dcp:
            config.advanced_preprocessing.advanced_steps['dark_channel_prior'] = True
            print("   🌫️  Dark Channel Prior enabled (this will increase processing time)")
        
        # Initialize processors
        print("3. Initializing processors...")
        
        # Advanced processor with full pipeline
        advanced_processor = ImagePreprocessor({
            'advanced_config': config.advanced_preprocessing.__dict__,
            **config.preprocessing.__dict__
        })
        
        # Basic processor for comparison
        basic_config = config.preprocessing.__dict__.copy()
        basic_config['enable_advanced_processing'] = False
        basic_processor = ImagePreprocessor(basic_config)
        
        # Save original image
        save_image(original_image, output_dir / "00_original.jpg")
        
        # Demonstrate individual techniques
        if args.show_individual:
            individual_results = demonstrate_individual_techniques(
                original_image, 
                AdvancedPreprocessor(config.advanced_preprocessing.__dict__),
                output_dir, 
                args.timing
            )
            
            if args.create_grid:
                grid_images = {name: img for name, img in individual_results}
                create_comparison_grid(
                    grid_images, 
                    output_dir / "individual_techniques_grid.png",
                    "Individual Advanced Techniques"
                )
        
        # Demonstrate Dark Channel Prior if enabled
        if args.enable_dcp:
            demonstrate_dark_channel_prior(
                original_image,
                AdvancedPreprocessor(config.advanced_preprocessing.__dict__),
                output_dir,
                args.timing
            )
        
        # Apply full pipeline
        if args.full_pipeline:
            print("\n🚀 Applying Full Advanced Preprocessing Pipeline...")
            
            start_time = time.time()
            final_result = advanced_processor.process_single_image(original_image)
            processing_time = time.time() - start_time
            
            if args.timing:
                print(f"   ⏱️  Total processing time: {processing_time:.2f}s")
            
            save_image(final_result, output_dir / "full_advanced_pipeline.jpg")
            print(f"   💾 Final result saved: {output_dir / 'full_advanced_pipeline.jpg'}")
        
        # Compare basic vs advanced
        if args.comparison:
            basic_result, advanced_result = compare_basic_vs_advanced(
                original_image, basic_processor, advanced_processor, output_dir
            )
            
            if args.create_grid:
                comparison_images = {
                    "Original": original_image,
                    "Basic Processing": basic_result,
                    "Advanced Processing": advanced_result
                }
                create_comparison_grid(
                    comparison_images,
                    output_dir / "basic_vs_advanced_grid.png", 
                    "Basic vs Advanced Processing Comparison"
                )
        
        # Print final summary
        print(f"\n✅ Advanced preprocessing demonstration completed!")
        print(f"📁 All results saved to: {output_dir}")
        
        # Configuration summary
        print(f"\n🔧 Configuration Summary:")
        print(f"   Advanced Processing Enabled: {config.preprocessing.enable_advanced_processing}")
        
        enabled_techniques = [k for k, v in config.advanced_preprocessing.advanced_steps.items() if v]
        print(f"   Enabled Advanced Techniques: {', '.join(enabled_techniques)}")
        
        print(f"\n💡 Advanced Techniques Benefits:")
        print("   🔬 Homomorphic Filtering: Better illumination-reflectance separation")
        print("   🎨 Advanced White Balance: Improved underwater color correction")
        print("   🌫️  Dark Channel Prior: Effective haze and scattering removal")
        print("   🧬 Gabor Enhancement: Superior texture pattern enhancement")
        print("   🔍 Multi-scale Enhancement: Detail enhancement at multiple scales")
        print("   📐 Morphological Enhancement: Better boundary definition")
        
        if args.timing:
            print(f"\n⚡ Performance Note:")
            print("   Advanced techniques provide better quality at the cost of processing time")
            print("   Consider enabling/disabling specific techniques based on your requirements")
        
    except Exception as e:
        print(f"❌ Error during advanced preprocessing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())