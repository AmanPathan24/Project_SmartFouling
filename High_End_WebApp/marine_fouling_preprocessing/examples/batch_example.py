#!/usr/bin/env python3
"""
Batch Processing Example: Process Multiple Images

This script demonstrates how to process multiple images using the Marine Fouling
Image Preprocessing Pipeline with batch processing capabilities.

Usage:
    python batch_example.py input_directory output_directory [--augment] [--workers N]
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from batch_processor import BatchProcessor
from config import create_marine_optimized_config, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process marine fouling images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic batch processing
    python batch_example.py input_images/ output_images/

    # With data augmentation (3 augmented versions per image)
    python batch_example.py input_images/ output_images/ --augment --num-aug 3

    # With custom number of workers
    python batch_example.py input_images/ output_images/ --workers 4

    # Preprocessing only, no subdirectories
    python batch_example.py input_images/ output_images/ --no-subdirs
        """
    )
    
    parser.add_argument('input_dir', 
                       help='Input directory containing images')
    parser.add_argument('output_dir',
                       help='Output directory for processed images')
    
    parser.add_argument('--augment', '-a', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--num-aug', type=int, default=2,
                       help='Number of augmented versions per image (default: 2)')
    
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    
    parser.add_argument('--no-subdirs', action='store_true',
                       help='Do not create subdirectories for outputs')
    parser.add_argument('--save-original', action='store_true',
                       help='Save original images in output')
    
    parser.add_argument('--output-format', choices=['jpg', 'png'], default='jpg',
                       help='Output image format (default: jpg)')
    parser.add_argument('--quality', type=int, default=90,
                       help='Output quality for JPEG (default: 90)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration based on command line arguments."""
    # Start with marine-optimized config
    config = create_marine_optimized_config()
    
    # Update batch processing config based on arguments
    batch_config = config.batch_processing
    
    if args.workers is not None:
        batch_config.num_workers = args.workers
    batch_config.batch_size = args.batch_size
    
    batch_config.apply_augmentation = args.augment
    batch_config.num_augmentations = args.num_aug if args.augment else 0
    
    batch_config.create_subdirs = not args.no_subdirs
    batch_config.save_original = args.save_original
    
    batch_config.output_format = f'.{args.output_format}'
    batch_config.output_quality = args.quality
    
    # Update logging level
    if args.verbose:
        config.logging_level = "DEBUG"
    
    return config


def print_config_summary(config):
    """Print a summary of the configuration."""
    print("📋 Configuration Summary:")
    print(f"   Processing Mode: {config.processing_mode.value}")
    print(f"   Multiprocessing: {config.batch_processing.enable_multiprocessing}")
    print(f"   Workers: {config.batch_processing.num_workers}")
    print(f"   Batch Size: {config.batch_processing.batch_size}")
    print(f"   Apply Preprocessing: {config.batch_processing.apply_preprocessing}")
    print(f"   Apply Augmentation: {config.batch_processing.apply_augmentation}")
    if config.batch_processing.apply_augmentation:
        print(f"   Augmentations per image: {config.batch_processing.num_augmentations}")
    print(f"   Output Format: {config.batch_processing.output_format}")
    print(f"   Create Subdirs: {config.batch_processing.create_subdirs}")
    print(f"   Save Original: {config.batch_processing.save_original}")


def print_processing_summary(summary):
    """Print a summary of the processing results."""
    stats = summary['statistics']
    
    print(f"\n📊 Processing Results Summary:")
    print(f"   Total Images: {stats['total_images']}")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   ⏱️  Total Time: {stats['processing_time']:.2f} seconds")
    
    if stats['total_images'] > 0:
        success_rate = (stats['successful'] / stats['total_images']) * 100
        avg_time = stats['processing_time'] / stats['total_images']
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Avg Time per Image: {avg_time:.2f} seconds")
    
    if stats['failed'] > 0:
        print(f"\n⚠️  Failed Images:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"     {Path(error['image']).name}: {error['error']}")
        if len(stats['errors']) > 5:
            print(f"     ... and {len(stats['errors']) - 5} more")


def main():
    """Main function for batch processing."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    print("=== Marine Fouling Image Preprocessing - Batch Processing ===\n")
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"❌ Input path is not a directory: {input_dir}")
        return 1
    
    output_dir = Path(args.output_dir)
    
    try:
        # Create configuration
        print("1. Creating configuration...")
        config = create_config_from_args(args)
        print_config_summary(config)
        
        # Initialize batch processor
        print(f"\n2. Initializing batch processor...")
        batch_processor = BatchProcessor({
            'preprocessing_config': config.preprocessing.__dict__,
            'augmentation_config': config.augmentation.__dict__,
            **config.batch_processing.__dict__
        })
        
        # Discover images
        print(f"3. Discovering images in: {input_dir}")
        image_paths = batch_processor.discover_images(input_dir)
        if not image_paths:
            print(f"❌ No supported images found in {input_dir}")
            return 1
        
        print(f"   Found {len(image_paths)} images to process")
        
        # Process images
        print(f"\n4. Processing images...")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   {'🚀 Starting batch processing...' if config.batch_processing.enable_multiprocessing else '🔄 Starting sequential processing...'}")
        
        summary = batch_processor.process_directory(input_dir, output_dir)
        
        # Print results
        print_processing_summary(summary)
        
        # Save detailed report
        report_path = output_dir / 'batch_processing_report.json'
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        print(f"\n✅ Batch processing completed!")
        print(f"📁 Output directory: {output_dir}")
        
        # Show output structure
        if config.batch_processing.create_subdirs:
            print(f"\n📂 Output Structure:")
            for subdir_key, subdir_name in config.batch_processing.subdir_names.items():
                subdir_path = output_dir / subdir_name
                if subdir_path.exists():
                    file_count = len(list(subdir_path.glob('*.*')))
                    print(f"   {subdir_name}/: {file_count} files")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during batch processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())