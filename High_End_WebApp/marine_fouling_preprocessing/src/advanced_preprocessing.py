"""
Advanced Preprocessing Techniques for Marine Fouling Images

This module implements sophisticated preprocessing techniques specifically designed
for underwater marine fouling detection, including homomorphic filtering, 
dark channel prior, Gabor filters, and multi-scale enhancement.
"""

import cv2
import numpy as np
from scipy import ndimage, fft
from skimage import morphology, filters, feature, restoration
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import logging
from typing import Tuple, List, Optional, Dict, Any, Union


class AdvancedPreprocessor:
    """
    Advanced preprocessing techniques for marine fouling images.
    
    Provides sophisticated algorithms for underwater image enhancement including
    homomorphic filtering, dark channel prior, Gabor texture enhancement,
    and multi-scale feature enhancement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdvancedPreprocessor.
        
        Args:
            config: Configuration dictionary for advanced preprocessing
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced preprocessing."""
        return {
            # Homomorphic filtering
            'homomorphic_gamma_h': 2.0,
            'homomorphic_gamma_l': 0.25,
            'homomorphic_c': 1.0,
            'homomorphic_d0': 10.0,
            
            # White balance correction
            'wb_method': 'max_white',  # 'gray_world', 'max_white', 'perfect_reflector'
            'wb_percentile': 95,
            'wb_clip_limit': 0.01,
            
            # Dark channel prior
            'dcp_patch_size': 15,
            'dcp_omega': 0.95,
            'dcp_t0': 0.1,
            'dcp_guided_eps': 0.001,
            'dcp_guided_radius': 60,
            
            # Gabor filters
            'gabor_frequencies': [0.1, 0.3, 0.5],
            'gabor_angles': [0, 45, 90, 135],
            'gabor_sigma_x': 2.0,
            'gabor_sigma_y': 2.0,
            'gabor_enhancement_strength': 0.3,
            
            # Multi-scale enhancement
            'pyramid_levels': 4,
            'detail_enhancement_factor': 1.5,
            'edge_enhancement_factor': 1.2,
            
            # Morphological operations
            'morphology_kernel_size': 3,
            'morphology_operations': ['opening', 'closing'],
            'boundary_enhancement_strength': 0.4,
        }
    
    def apply_homomorphic_filtering(self, image: np.ndarray) -> np.ndarray:
        """
        Apply homomorphic filtering for illumination-reflectance separation.
        
        Homomorphic filtering is excellent for separating illumination and 
        reflectance components in underwater images with uneven lighting.
        
        Args:
            image: Input image
            
        Returns:
            Homomorphic filtered image
        """
        if len(image.shape) == 3:
            # Process each channel separately for color images
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                result[:, :, i] = self._homomorphic_filter_channel(image[:, :, i])
            return result
        else:
            return self._homomorphic_filter_channel(image)
    
    def _homomorphic_filter_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply homomorphic filtering to a single channel."""
        # Convert to float and add small constant to avoid log(0)
        img_float = channel.astype(np.float32) + 1.0
        
        # Take natural logarithm
        img_log = np.log(img_float)
        
        # Apply FFT
        img_fft = fft.fft2(img_log)
        img_fft_shift = fft.fftshift(img_fft)
        
        # Create high-frequency emphasis filter
        rows, cols = img_log.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create frequency domain filter
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols) - ccol
        D = np.sqrt(u**2 + v**2)
        
        # High-frequency emphasis filter
        gamma_h = self.config['homomorphic_gamma_h']
        gamma_l = self.config['homomorphic_gamma_l']
        c = self.config['homomorphic_c']
        d0 = self.config['homomorphic_d0']
        
        H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2) / (d0**2))) + gamma_l
        
        # Apply filter
        img_filtered = img_fft_shift * H
        
        # Inverse FFT
        img_ifft_shift = fft.ifftshift(img_filtered)
        img_ifft = fft.ifft2(img_ifft_shift)
        img_result = np.real(img_ifft)
        
        # Exponential to reverse logarithm
        img_final = np.exp(img_result) - 1.0
        
        # Normalize to [0, 255]
        img_final = np.clip(img_final, 0, 255)
        
        return img_final.astype(np.uint8)
    
    def advanced_white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced white balance correction algorithms.
        
        Implements multiple white balance methods beyond gray world assumption,
        specifically optimized for underwater conditions.
        
        Args:
            image: Input color image
            
        Returns:
            White balance corrected image
        """
        if len(image.shape) != 3:
            return image
        
        method = self.config['wb_method']
        
        if method == 'gray_world':
            return self._gray_world_wb(image)
        elif method == 'max_white':
            return self._max_white_wb(image)
        elif method == 'perfect_reflector':
            return self._perfect_reflector_wb(image)
        else:
            self.logger.warning(f"Unknown white balance method: {method}")
            return image
    
    def _gray_world_wb(self, image: np.ndarray) -> np.ndarray:
        """Gray world white balance algorithm."""
        img_float = image.astype(np.float32)
        
        # Calculate channel means
        mean_r = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_b = np.mean(img_float[:, :, 2])
        
        # Calculate gray value
        gray_mean = (mean_r + mean_g + mean_b) / 3.0
        
        # Calculate scaling factors
        scale_r = gray_mean / mean_r if mean_r > 0 else 1.0
        scale_g = gray_mean / mean_g if mean_g > 0 else 1.0
        scale_b = gray_mean / mean_b if mean_b > 0 else 1.0
        
        # Apply corrections
        corrected = img_float.copy()
        corrected[:, :, 0] *= scale_r
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_b
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _max_white_wb(self, image: np.ndarray) -> np.ndarray:
        """Max white (white patch) white balance algorithm."""
        img_float = image.astype(np.float32)
        percentile = self.config['wb_percentile']
        
        # Find max values for each channel (using percentile to avoid outliers)
        max_r = np.percentile(img_float[:, :, 0], percentile)
        max_g = np.percentile(img_float[:, :, 1], percentile)
        max_b = np.percentile(img_float[:, :, 2], percentile)
        
        # Calculate scaling factors
        max_val = max(max_r, max_g, max_b)
        scale_r = max_val / max_r if max_r > 0 else 1.0
        scale_g = max_val / max_g if max_g > 0 else 1.0
        scale_b = max_val / max_b if max_b > 0 else 1.0
        
        # Apply corrections
        corrected = img_float.copy()
        corrected[:, :, 0] *= scale_r
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_b
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _perfect_reflector_wb(self, image: np.ndarray) -> np.ndarray:
        """Perfect reflector white balance algorithm."""
        img_float = image.astype(np.float32)
        
        # Find brightest pixels (potential white references)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        threshold = np.percentile(gray, 99)
        
        # Mask for bright pixels
        bright_mask = gray > threshold
        
        if np.sum(bright_mask) > 0:
            # Calculate mean of bright pixels for each channel
            mean_r = np.mean(img_float[:, :, 0][bright_mask])
            mean_g = np.mean(img_float[:, :, 1][bright_mask])
            mean_b = np.mean(img_float[:, :, 2][bright_mask])
            
            # Calculate scaling factors
            max_val = max(mean_r, mean_g, mean_b)
            scale_r = max_val / mean_r if mean_r > 0 else 1.0
            scale_g = max_val / mean_g if mean_g > 0 else 1.0
            scale_b = max_val / mean_b if mean_b > 0 else 1.0
            
            # Apply corrections
            corrected = img_float.copy()
            corrected[:, :, 0] *= scale_r
            corrected[:, :, 1] *= scale_g
            corrected[:, :, 2] *= scale_b
            
            return np.clip(corrected, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_dark_channel_prior(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Dark Channel Prior for underwater haze removal.
        
        Dark Channel Prior is effective for removing haze and scattering
        effects common in underwater imaging environments.
        
        Args:
            image: Input color image
            
        Returns:
            Dehazed image
        """
        if len(image.shape) != 3:
            return image
        
        img_float = image.astype(np.float32) / 255.0
        
        # Step 1: Compute dark channel
        dark_channel = self._compute_dark_channel(img_float)
        
        # Step 2: Estimate atmospheric light
        atmospheric_light = self._estimate_atmospheric_light(img_float, dark_channel)
        
        # Step 3: Estimate transmission map
        transmission = self._estimate_transmission(img_float, atmospheric_light)
        
        # Step 4: Refine transmission map using guided filter
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        transmission_refined = self._guided_filter(gray, transmission)
        
        # Step 5: Recover scene radiance
        recovered = self._recover_scene_radiance(img_float, atmospheric_light, transmission_refined)
        
        return (np.clip(recovered, 0, 1) * 255).astype(np.uint8)
    
    def _compute_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Compute dark channel of the image."""
        patch_size = self.config['dcp_patch_size']
        
        # Find minimum across color channels
        min_channel = np.min(image, axis=2)
        
        # Apply minimum filter (erosion)
        kernel = np.ones((patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def _estimate_atmospheric_light(self, image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        """Estimate atmospheric light from brightest pixels in dark channel."""
        h, w = dark_channel.shape
        num_pixels = int(0.001 * h * w)  # Top 0.1% of pixels
        
        # Find brightest pixels in dark channel
        dark_vec = dark_channel.reshape(-1)
        indices = np.argsort(dark_vec)[-num_pixels:]
        
        # Find corresponding pixels in original image
        atmospheric_light = np.zeros(3)
        max_intensity = 0
        
        for idx in indices:
            y, x = divmod(idx, w)
            intensity = np.sum(image[y, x, :])
            if intensity > max_intensity:
                max_intensity = intensity
                atmospheric_light = image[y, x, :].copy()
        
        return atmospheric_light
    
    def _estimate_transmission(self, image: np.ndarray, atmospheric_light: np.ndarray) -> np.ndarray:
        """Estimate transmission map."""
        omega = self.config['dcp_omega']
        
        # Normalize by atmospheric light
        normalized = image / atmospheric_light
        
        # Compute dark channel of normalized image
        transmission = 1 - omega * self._compute_dark_channel(normalized)
        
        return transmission
    
    def _guided_filter(self, guide: np.ndarray, src: np.ndarray) -> np.ndarray:
        """Apply guided filter for transmission map refinement."""
        eps = self.config['dcp_guided_eps']
        radius = self.config['dcp_guided_radius']
        
        # Box filter implementation
        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (r, r))
        
        # Calculate statistics
        mean_guide = box_filter(guide, radius)
        mean_src = box_filter(src, radius)
        mean_guide_src = box_filter(guide * src, radius)
        
        # Covariance
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        
        # Variance of guide
        mean_guide_sq = box_filter(guide * guide, radius)
        var_guide = mean_guide_sq - mean_guide * mean_guide
        
        # Linear coefficients
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        # Filter coefficients
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        
        # Output
        filtered = mean_a * guide + mean_b
        
        return filtered
    
    def _recover_scene_radiance(self, image: np.ndarray, atmospheric_light: np.ndarray, 
                               transmission: np.ndarray) -> np.ndarray:
        """Recover scene radiance from hazy image."""
        t0 = self.config['dcp_t0']
        
        # Ensure transmission is not too small
        transmission = np.maximum(transmission, t0)
        
        # Recover radiance
        recovered = np.zeros_like(image)
        for i in range(3):
            recovered[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
        return recovered
    
    def apply_gabor_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filters for texture enhancement.
        
        Gabor filters are excellent for enhancing texture patterns
        characteristic of marine fouling organisms.
        
        Args:
            image: Input image
            
        Returns:
            Gabor enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            is_color = True
        else:
            gray = image
            is_color = False
        
        # Apply Gabor filters
        gabor_responses = self._apply_gabor_filters(gray)
        
        # Combine responses
        enhanced_gray = self._combine_gabor_responses(gray, gabor_responses)
        
        if is_color:
            # Apply enhancement to original color image
            enhancement_factor = enhanced_gray / (gray + 1e-6)
            enhanced = image.astype(np.float32) * enhancement_factor[:, :, np.newaxis]
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            return enhanced_gray
    
    def _apply_gabor_filters(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply multiple Gabor filters with different orientations and frequencies."""
        responses = []
        
        frequencies = self.config['gabor_frequencies']
        angles = self.config['gabor_angles']
        sigma_x = self.config['gabor_sigma_x']
        sigma_y = self.config['gabor_sigma_y']
        
        for freq in frequencies:
            for angle in angles:
                # Convert angle to radians
                theta = np.radians(angle)
                
                # Apply Gabor filter
                real, _ = filters.gabor(image, frequency=freq, theta=theta, 
                                      sigma_x=sigma_x, sigma_y=sigma_y)
                responses.append(real)
        
        return responses
    
    def _combine_gabor_responses(self, original: np.ndarray, 
                                responses: List[np.ndarray]) -> np.ndarray:
        """Combine multiple Gabor filter responses."""
        # Calculate response magnitude
        magnitude = np.zeros_like(original, dtype=np.float32)
        
        for response in responses:
            magnitude += np.abs(response)
        
        # Normalize magnitude
        magnitude = magnitude / len(responses)
        
        # Enhance original image
        enhancement_strength = self.config['gabor_enhancement_strength']
        enhanced = original.astype(np.float32) + enhancement_strength * magnitude
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def apply_multiscale_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale feature enhancement using image pyramids.
        
        Enhances details at multiple scales, which is crucial for detecting
        fouling patterns of different sizes.
        
        Args:
            image: Input image
            
        Returns:
            Multi-scale enhanced image
        """
        if len(image.shape) == 3:
            # Process each channel separately
            enhanced = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = self._multiscale_enhance_channel(image[:, :, i])
            return enhanced.astype(np.uint8)
        else:
            return self._multiscale_enhance_channel(image)
    
    def _multiscale_enhance_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply multi-scale enhancement to a single channel."""
        levels = self.config['pyramid_levels']
        detail_factor = self.config['detail_enhancement_factor']
        edge_factor = self.config['edge_enhancement_factor']
        
        # Create Gaussian pyramid
        pyramid = [channel.astype(np.float32)]
        for i in range(levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Create Laplacian pyramid (detail layers)
        laplacian_pyramid = []
        for i in range(levels - 1):
            # Expand and subtract
            expanded = cv2.pyrUp(pyramid[i + 1], dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
            laplacian_pyramid.append(pyramid[i] - expanded)
        
        # Enhance detail layers
        enhanced_laplacian = []
        for i, detail in enumerate(laplacian_pyramid):
            # Apply different enhancement based on scale
            scale_factor = detail_factor * (0.8 ** i)  # Less enhancement at coarser scales
            enhanced_detail = detail * scale_factor
            enhanced_laplacian.append(enhanced_detail)
        
        # Reconstruct image
        reconstructed = pyramid[-1]  # Start with coarsest level
        for i in range(levels - 2, -1, -1):
            # Expand and add detail
            expanded = cv2.pyrUp(reconstructed, dstsize=(enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0]))
            reconstructed = expanded + enhanced_laplacian[i]
        
        # Additional edge enhancement
        edges = cv2.Laplacian(channel.astype(np.float32), cv2.CV_32F)
        reconstructed += edge_factor * edges
        
        return np.clip(reconstructed, 0, 255)
    
    def apply_morphological_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations for boundary enhancement.
        
        Morphological operations help enhance boundaries and shapes
        of fouling organisms against the background.
        
        Args:
            image: Input image
            
        Returns:
            Morphologically enhanced image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for morphological operations
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            enhanced_gray = self._apply_morphological_ops(gray)
            
            # Apply enhancement back to color image
            enhancement_mask = enhanced_gray.astype(np.float32) / (gray.astype(np.float32) + 1e-6)
            enhanced = image.astype(np.float32) * enhancement_mask[:, :, np.newaxis]
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            return self._apply_morphological_ops(image)
    
    def _apply_morphological_ops(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance boundaries."""
        kernel_size = self.config['morphology_kernel_size']
        operations = self.config['morphology_operations']
        enhancement_strength = self.config['boundary_enhancement_strength']
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        processed = image.copy()
        
        for op in operations:
            if op == 'opening':
                # Remove small objects and smooth boundaries
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            elif op == 'closing':
                # Fill small holes and connect nearby objects
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            elif op == 'gradient':
                # Enhance boundaries
                processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
            elif op == 'tophat':
                # Enhance bright features
                processed = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, kernel)
            elif op == 'blackhat':
                # Enhance dark features
                processed = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine with original for boundary enhancement
        enhanced = image.astype(np.float32) + enhancement_strength * (processed.astype(np.float32) - image.astype(np.float32))
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def histogram_specification(self, image: np.ndarray, 
                               reference_hist: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply histogram specification to match target histogram.
        
        Useful for normalizing images to a standard appearance.
        
        Args:
            image: Input image
            reference_hist: Reference histogram (if None, uses ideal underwater histogram)
            
        Returns:
            Histogram specified image
        """
        if len(image.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = self._histogram_specification_channel(
                    image[:, :, i], reference_hist
                )
            return result
        else:
            return self._histogram_specification_channel(image, reference_hist)
    
    def _histogram_specification_channel(self, channel: np.ndarray, 
                                       reference_hist: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply histogram specification to a single channel."""
        # If no reference histogram provided, create ideal underwater histogram
        if reference_hist is None:
            reference_hist = self._create_ideal_underwater_histogram()
        
        # Calculate CDFs
        source_hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        source_cdf = np.cumsum(source_hist).astype(np.float32)
        source_cdf = source_cdf / source_cdf[-1]
        
        reference_cdf = np.cumsum(reference_hist).astype(np.float32)
        reference_cdf = reference_cdf / reference_cdf[-1]
        
        # Create mapping
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value in reference
            diff = np.abs(reference_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        return mapping[channel]
    
    def _create_ideal_underwater_histogram(self) -> np.ndarray:
        """Create an ideal histogram for underwater images."""
        # Create a histogram that emphasizes mid-tones and reduces extremes
        hist = np.zeros(256)
        
        # Gaussian-like distribution centered around mid-tones
        x = np.arange(256)
        mu1, sigma1 = 80, 40    # Dark-mid tones
        mu2, sigma2 = 160, 30   # Mid-bright tones
        
        hist += 0.6 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        hist += 0.4 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        
        # Normalize
        hist = hist / np.sum(hist)
        
        return hist