#!/usr/bin/env python3
"""
Basic Example: Single Image Processing

This script demonstrates basic usage of the Marine Fouling Image Preprocessing Pipeline
for processing a single image through the complete pipeline.

Usage:
    python basic_example.py path/to/input/image.jpg path/to/output/image.jpg
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from image_preprocessor import ImagePreprocessor
from config import create_marine_optimized_config, setup_logging


def load_and_display_image(image_path: str) -> np.ndarray:
    """Load an image and display its properties."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Loaded image: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Value range: [{image.min()}, {image.max()}]")
    
    return image


def save_comparison_plot(original: np.ndarray, processed: np.ndarray, output_path: str):
    """Save a side-by-side comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Handle normalized images
    if processed.dtype == np.float32 or processed.dtype == np.float64:
        if processed.max() <= 1.0:
            display_processed = processed
        else:
            display_processed = processed / 255.0
    else:
        display_processed = processed
    
    ax2.imshow(display_processed)
    ax2.set_title('Processed Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def main():
    # Setup logging
    setup_logging("INFO")
    
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python basic_example.py <input_image> <output_image>")
        print("Example: python basic_example.py input.jpg output.jpg")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=== Marine Fouling Image Preprocessing - Basic Example ===\n")
    
    try:
        # Load the input image
        print("1. Loading input image...")
        original_image = load_and_display_image(input_path)
        
        # Create optimized configuration for marine images
        print("\n2. Creating marine-optimized configuration...")
        config = create_marine_optimized_config()
        
        # Initialize the preprocessor
        print("3. Initializing image preprocessor...")
        preprocessor = ImagePreprocessor(config.preprocessing.__dict__)
        
        # Process the image through the complete pipeline
        print("4. Processing image through preprocessing pipeline...")
        print("   - Noise reduction")
        print("   - Color correction")
        print("   - Lighting enhancement (CLAHE + Retinex)")
        print("   - Contrast enhancement")
        print("   - Image sharpening")
        print("   - Normalization")
        
        processed_image = preprocessor.process_single_image(original_image)
        
        # Save the processed image
        print(f"\n5. Saving processed image to: {output_path}")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert back to BGR for saving with OpenCV
        if len(processed_image.shape) == 3:
            if processed_image.dtype in [np.float32, np.float64]:
                save_image = (processed_image * 255).astype(np.uint8)
            else:
                save_image = processed_image
            save_image_bgr = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        else:
            if processed_image.dtype in [np.float32, np.float64]:
                save_image_bgr = (processed_image * 255).astype(np.uint8)
            else:
                save_image_bgr = processed_image
        
        cv2.imwrite(output_path, save_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create and save comparison plot
        print("6. Creating comparison plot...")
        comparison_path = str(Path(output_path).with_suffix('_comparison.png'))
        save_comparison_plot(original_image, processed_image, comparison_path)
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Processed image saved to: {output_path}")
        print(f"📊 Comparison plot saved to: {comparison_path}")
        
        # Display processing statistics
        print(f"\n📈 Processing Statistics:")
        print(f"   Original image shape: {original_image.shape}")
        print(f"   Processed image shape: {processed_image.shape}")
        print(f"   Original dtype: {original_image.dtype}")
        print(f"   Processed dtype: {processed_image.dtype}")
        
        if processed_image.dtype in [np.float32, np.float64]:
            print(f"   Processed value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        else:
            print(f"   Processed value range: [{processed_image.min()}, {processed_image.max()}]")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())