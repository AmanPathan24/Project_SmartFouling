"""
Utility functions for Marine Biofouling Detection Backend
"""

import uuid
import os
from datetime import datetime
from typing import Union
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate a unique session ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"session_{timestamp}_{unique_id}"

async def save_image(image: Union[Image.Image, np.ndarray], file_path: str) -> str:
    """
    Save image to file system
    
    Args:
        image: PIL Image or numpy array
        file_path: Path to save the image
        
    Returns:
        Saved file path
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image, mode='L')
        
        # Save the image
        image.save(file_path)
        
        logger.info(f"Image saved to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save image to {file_path}: {e}")
        raise

async def load_image(file_path: str) -> Image.Image:
    """
    Load image from file system
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Image loaded from {file_path}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image from {file_path}: {e}")
        raise

def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        "size": image.size,
        "mode": image.mode,
        "format": image.format,
        "width": image.width,
        "height": image.height
    }

def resize_image_if_needed(image: Image.Image, max_size: tuple = (2048, 2048)) -> Image.Image:
    """
    Resize image if it's too large
    
    Args:
        image: PIL Image object
        max_size: Maximum size (width, height)
        
    Returns:
        Resized PIL Image object
    """
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Image resized to {image.size}")
    
    return image

def calculate_image_hash(image: Image.Image) -> str:
    """
    Calculate a simple hash for image comparison
    
    Args:
        image: PIL Image object
        
    Returns:
        Hash string
    """
    # Convert to grayscale and resize to small size for hashing
    small_image = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
    
    # Calculate average pixel value
    pixels = list(small_image.getdata())
    avg_pixel = sum(pixels) / len(pixels)
    
    # Create hash based on pixel comparison with average
    hash_bits = []
    for pixel in pixels:
        hash_bits.append('1' if pixel > avg_pixel else '0')
    
    # Convert binary to hexadecimal
    hash_string = hex(int(''.join(hash_bits), 2))[2:]
    
    return hash_string

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_filename(filename: str) -> str:
    """
    Clean filename for safe file system usage
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return filename

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename
    
    Args:
        filename: Filename with extension
        
    Returns:
        File extension (without dot)
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')

def is_supported_image_format(filename: str) -> bool:
    """
    Check if file format is supported
    
    Args:
        filename: Filename to check
        
    Returns:
        True if supported, False otherwise
    """
    supported_formats = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
    extension = get_file_extension(filename)
    return extension in supported_formats

def create_directory_structure(base_path: str) -> dict:
    """
    Create necessary directory structure for the application
    
    Args:
        base_path: Base directory path
        
    Returns:
        Dictionary with created directory paths
    """
    directories = {
        'uploads': os.path.join(base_path, 'uploads'),
        'processed': os.path.join(base_path, 'processed'),
        'outputs': os.path.join(base_path, 'outputs'),
        'static': os.path.join(base_path, 'static'),
        'data': os.path.join(base_path, 'data'),
        'models': os.path.join(base_path, 'models')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories
