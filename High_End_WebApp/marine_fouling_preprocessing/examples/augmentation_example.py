#!/usr/bin/env python3
"""
Data Augmentation Example: Marine-Specific Image Augmentation

This script demonstrates the specialized data augmentation techniques designed
for marine fouling images, including underwater-specific effects.

Usage:
    python augmentation_example.py input_image.jpg output_directory/ [--num-augmentations N]
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_augmentation import MarineDataAugmenter
from config import create_marine_optimized_config, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate marine-specific data augmentation techniques"
    )
    
    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument('output_dir', help='Output directory for augmented images')
    parser.add_argument('--num-augmentations', '-n', type=int, default=5,
                       help='Number of augmented versions to generate (default: 5)')
    parser.add_argument('--show-effects', '-s', action='store_true',
                       help='Show individual underwater effects')
    parser.add_argument('--create-grid', '-g', action='store_true',
                       help='Create a grid comparison image')
    
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load and validate input image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {Path(image_path).name}")
    print(f"Image shape: {image.shape}")
    
    return image


def save_image(image: np.ndarray, path: Path, quality: int = 95):
    """Save image to file."""
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3:
        save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        save_image = image
    
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), save_image, [cv2.IMWRITE_JPEG_QUALITY, quality])


def demonstrate_individual_effects(original_image: np.ndarray, augmenter: MarineDataAugmenter, 
                                 output_dir: Path):
    """Demonstrate individual underwater effects."""
    print("\n🌊 Demonstrating individual underwater effects:")
    
    effects = [
        ('original', lambda img: img),
        ('particles', lambda img: augmenter.simulate_underwater_particles(img)),
        ('color_cast', lambda img: augmenter.apply_underwater_color_cast(img)),
        ('light_attenuation_shallow', lambda img: augmenter.apply_light_attenuation(img, 0.2)),
        ('light_attenuation_deep', lambda img: augmenter.apply_light_attenuation(img, 0.5)),
    ]
    
    effect_images = []
    
    for effect_name, effect_func in effects:
        print(f"   Applying {effect_name}...")
        effect_image = effect_func(original_image.copy())
        effect_images.append((effect_name, effect_image))
        
        # Save individual effect
        save_path = output_dir / f"effect_{effect_name}.jpg"
        save_image(effect_image, save_path)
        print(f"     Saved: {save_path}")
    
    return effect_images


def create_effects_grid(effect_images, output_dir: Path):
    """Create a grid showing all individual effects."""
    print("\n📊 Creating effects comparison grid...")
    
    # Arrange in a 2x3 grid (we have 5 effects including original)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (effect_name, image) in enumerate(effect_images):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(f"{effect_name.replace('_', ' ').title()}", fontsize=12)
            axes[i].axis('off')
    
    # Hide the last empty subplot
    for i in range(len(effect_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "effects_comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Grid saved: {grid_path}")


def create_augmentation_grid(images, output_dir: Path):
    """Create a grid showing all augmented versions."""
    print("\n📊 Creating augmentation comparison grid...")
    
    # Determine grid size
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    # Handle single row case
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten() if n_images > 1 else [axes]
    
    for i, (name, image) in enumerate(images):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(name, fontsize=10)
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / "augmentations_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Grid saved: {grid_path}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging("INFO")
    
    print("=== Marine Fouling Data Augmentation Demo ===\n")
    
    try:
        # Load input image
        print("1. Loading input image...")
        original_image = load_image(args.input_image)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create marine-optimized configuration
        print("2. Creating marine-optimized augmentation configuration...")
        config = create_marine_optimized_config()
        augmenter = MarineDataAugmenter(config.augmentation.__dict__)
        
        # Save original image
        save_image(original_image, output_dir / "00_original.jpg")
        
        # Demonstrate individual effects if requested
        if args.show_effects:
            effect_images = demonstrate_individual_effects(original_image, augmenter, output_dir)
            if args.create_grid:
                create_effects_grid(effect_images, output_dir)
        
        # Generate augmented versions
        print(f"\n🔄 Generating {args.num_augmentations} augmented versions...")
        augmented_images = [("Original", original_image)]
        
        for i in range(args.num_augmentations):
            print(f"   Generating augmentation {i+1}/{args.num_augmentations}...")
            
            # Apply augmentation
            aug_image = augmenter.augment_image(original_image, apply_underwater=True)
            augmented_images.append((f"Augmentation {i+1}", aug_image))
            
            # Save augmented image
            save_path = output_dir / f"{i+1:02d}_augmented_{i+1}.jpg"
            save_image(aug_image, save_path)
            print(f"     Saved: {save_path}")
        
        # Create comparison grid
        if args.create_grid:
            create_augmentation_grid(augmented_images, output_dir)
        
        # Print summary
        print(f"\n✅ Augmentation completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated {len(augmented_images)} images total")
        
        # Show configuration summary
        print(f"\n🔧 Augmentation Configuration Used:")
        config_dict = config.augmentation.__dict__
        important_params = [
            'particles_density', 'color_cast_strength', 'water_distortion_alpha',
            'underwater_prob', 'geometric_prob', 'color_prob', 'noise_prob'
        ]
        
        for param in important_params:
            if param in config_dict:
                print(f"   {param}: {config_dict[param]}")
        
        print(f"\n💡 Tips for Marine Fouling Detection:")
        print("   - Use augmented data to improve model robustness")
        print("   - Underwater effects simulate realistic marine conditions")
        print("   - Vary augmentation parameters based on your specific dataset")
        print("   - Consider depth-specific augmentations for different environments")
        
    except Exception as e:
        print(f"❌ Error during augmentation: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())