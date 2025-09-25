#!/usr/bin/env python3
"""
Simple Test Script: Single Image Processing - Basic Features Only

This script demonstrates basic image preprocessing without advanced features
to avoid import issues while still showing the core functionality.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import ndimage


class SimpleImagePreprocessor:
    """Simplified image preprocessor with basic marine-optimized features."""
    
    def __init__(self):
        self.config = {
            'clahe_clip_limit': 4.0,  # Marine optimized
            'clahe_tile_grid_size': (6, 6),  # Marine optimized
            'retinex_sigma_list': [20, 100, 300],  # Marine optimized
            'gamma_correction': 1.3,  # Marine optimized
            'sharpening_strength': 0.7,  # Marine optimized
            'gaussian_blur_sigma': 0.8,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
    
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            l_channel = clahe.apply(l_channel)
            
            # Merge and convert back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            return clahe.apply(image)
    
    def apply_retinex(self, image):
        """Apply Multi-Scale Retinex for illumination correction."""
        img_float = image.astype(np.float64) / 255.0
        
        # Avoid log(0) by adding small epsilon
        img_float = np.maximum(img_float, 1e-4)
        
        # Apply MSR
        retinex_output = np.zeros_like(img_float)
        
        for sigma in self.config['retinex_sigma_list']:
            # Gaussian blur
            if len(img_float.shape) == 3:
                blurred = np.zeros_like(img_float)
                for i in range(img_float.shape[2]):
                    blurred[:, :, i] = ndimage.gaussian_filter(img_float[:, :, i], sigma=sigma)
            else:
                blurred = ndimage.gaussian_filter(img_float, sigma=sigma)
            
            # Avoid log(0)
            blurred = np.maximum(blurred, 1e-4)
            
            # Retinex calculation
            retinex_output += np.log(img_float) - np.log(blurred)
        
        retinex_output = retinex_output / len(self.config['retinex_sigma_list'])
        
        # Normalize to 0-255
        retinex_output = np.expm1(retinex_output)  # exp(x) - 1 for better numerical stability
        retinex_output = np.clip(retinex_output, 0, 1) * 255.0
        
        return retinex_output.astype(np.uint8)
    
    def apply_noise_reduction(self, image):
        """Apply bilateral filtering for noise reduction."""
        return cv2.bilateralFilter(
            image, 
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
    
    def apply_gamma_correction(self, image):
        """Apply gamma correction for brightness adjustment."""
        gamma = self.config['gamma_correction']
        # Normalize to 0-1, apply gamma, then scale back
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return (corrected * 255.0).astype(np.uint8)
    
    def apply_unsharp_masking(self, image):
        """Apply unsharp masking for image sharpening."""
        # Gaussian blur
        if len(image.shape) == 3:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.0 + self.config['sharpening_strength'], 
                                      blurred, -self.config['sharpening_strength'], 0)
        
        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    def process_single_image(self, image):
        """Process a single image through the marine-optimized pipeline."""
        print("   - Applying noise reduction...")
        processed = self.apply_noise_reduction(image)
        
        print("   - Applying CLAHE for contrast enhancement...")
        processed = self.apply_clahe(processed)
        
        print("   - Applying Multi-Scale Retinex for illumination correction...")
        processed = self.apply_retinex(processed)
        
        print("   - Applying gamma correction...")
        processed = self.apply_gamma_correction(processed)
        
        print("   - Applying unsharp masking for sharpening...")
        processed = self.apply_unsharp_masking(processed)
        
        return processed


def load_and_display_image(image_path: str):
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


def save_comparison_plot(original, processed, output_path):
    """Save a side-by-side comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(processed)
    ax2.set_title('Processed Image (Marine Optimized)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def main():
    import sys
    
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python simple_test.py <input_image> <output_image>")
        print("Example: python simple_test.py sample.jpg processed_sample.jpg")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("=== Marine Fouling Image Preprocessing - Simple Test ===\n")
    
    try:
        # Load the input image
        print("1. Loading input image...")
        original_image = load_and_display_image(input_path)
        
        # Initialize the simplified preprocessor
        print("\n2. Initializing marine-optimized preprocessor...")
        preprocessor = SimpleImagePreprocessor()
        
        # Process the image through the pipeline
        print("3. Processing image through preprocessing pipeline...")
        processed_image = preprocessor.process_single_image(original_image)
        
        # Save the processed image
        print(f"\n4. Saving processed image to: {output_path}")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert back to BGR for saving with OpenCV
        save_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, save_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create and save comparison plot
        print("5. Creating comparison plot...")
        comparison_path = str(Path(output_path).with_stem(Path(output_path).stem + '_comparison').with_suffix('.png'))
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
        print(f"   Original value range: [{original_image.min()}, {original_image.max()}]")
        print(f"   Processed value range: [{processed_image.min()}, {processed_image.max()}]")
        
        print(f"\n🌊 Applied Marine-Optimized Settings:")
        print(f"   - CLAHE clip limit: 4.0 (enhanced for underwater conditions)")
        print(f"   - CLAHE tile size: 6x6 (optimized for marine features)")
        print(f"   - Retinex scales: [20, 100, 300] (underwater illumination)")
        print(f"   - Gamma correction: 1.3 (marine brightness adjustment)")
        print(f"   - Sharpening strength: 0.7 (enhanced detail visibility)")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())