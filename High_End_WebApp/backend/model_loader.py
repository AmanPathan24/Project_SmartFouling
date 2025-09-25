"""
Model Loader for your specific PyTorch model format
Handles loading the best_model_dice_0.5029 model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class YourBiofoulingModel(nn.Module):
    """
    Model architecture for your biofouling detection model
    This is a generic U-Net style architecture - you may need to adjust this
    based on your actual model architecture
    """
    
    def __init__(self, num_classes=8):
        super(YourBiofoulingModel, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final segmentation map
        output = self.final(dec1)
        
        return output

def load_your_model(model_path: str = "best_model_dice_0.5029") -> Optional[torch.nn.Module]:
    """
    Load your specific model from the directory format
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Loaded PyTorch model or None if failed
    """
    try:
        # Check if the model directory exists
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found: {model_path}")
            return None
        
        # Method 1: Try to load using torch.load with map_location
        try:
            # Load the entire model directory
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info(f"Model loaded successfully using torch.load from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"torch.load failed: {e}")
        
        # Method 2: Try to load from data.pkl
        data_pkl_path = os.path.join(model_path, "data.pkl")
        if os.path.exists(data_pkl_path):
            try:
                with open(data_pkl_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create model architecture
                model = YourBiofoulingModel(num_classes=8)
                
                # If model_data contains state_dict, load it
                if isinstance(model_data, dict) and 'state_dict' in model_data:
                    model.load_state_dict(model_data['state_dict'])
                    logger.info("Model loaded from data.pkl with state_dict")
                elif isinstance(model_data, dict):
                    # Try to load the weights directly
                    model.load_state_dict(model_data)
                    logger.info("Model loaded from data.pkl with direct state_dict")
                else:
                    logger.warning("Unknown model_data format in data.pkl")
                    return None
                
                model.eval()
                return model
                
            except Exception as e:
                logger.error(f"Failed to load from data.pkl: {e}")
        
        # Method 3: Try to reconstruct from the data directory
        data_dir = os.path.join(model_path, "data")
        if os.path.exists(data_dir):
            try:
                # This is more complex - you'd need to know the exact format
                logger.info("Found data directory, but reconstruction not implemented yet")
                return None
            except Exception as e:
                logger.error(f"Failed to reconstruct from data directory: {e}")
        
        logger.error("All loading methods failed")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def test_model_loading():
    """Test function to verify model loading"""
    print("🔍 Testing model loading...")
    
    # Try different possible paths
    possible_paths = [
        "best_model_dice_0.5029",
        "../best_model_dice_0.5029",
        "backend/best_model_dice_0.5029"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found model at: {path}")
            model = load_your_model(path)
            if model is not None:
                print(f"✅ Model loaded successfully!")
                print(f"Model type: {type(model)}")
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"Total parameters: {total_params:,}")
                return model
            else:
                print(f"❌ Failed to load model from {path}")
        else:
            print(f"❌ Model not found at: {path}")
    
    print("❌ No model could be loaded")
    return None

if __name__ == "__main__":
    # Test the model loading
    model = test_model_loading()
    
    if model is not None:
        print("\n🎉 Model loading test successful!")
        
        # Test a dummy forward pass
        try:
            dummy_input = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    else:
        print("\n❌ Model loading test failed!")
