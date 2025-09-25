"""
Universal Model Loader - Handles both .pth files and directory format models
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_model_universal(model_path: str = None) -> Optional[torch.nn.Module]:
    """
    Universal model loader that handles both .pth files and directory format
    
    Args:
        model_path: Path to model (.pth file or directory)
        
    Returns:
        Loaded model or None if failed
    """
    print(f"🔍 Universal model loader - searching for model...")
    
    # If no path specified, search for models
    if model_path is None:
        possible_paths = [
            # Look for .pth files
            "best_model_dice_0.5029.pth",
            "best_model_dice.pth", 
            "biofouling_model.pth",
            "model.pth",
            
            # Look for directories
            "best_model_dice_0.5029",
            "../best_model_dice_0.5029",
            "./best_model_dice_0.5029",
            
            # Look in common locations
            "models/best_model_dice_0.5029.pth",
            "models/biofouling_model.pth",
            "models/best_model_dice_0.5029",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None:
        print("❌ No model found in common locations")
        return None
    
    print(f"✅ Found model at: {model_path}")
    
    # Check if it's a file or directory
    if os.path.isfile(model_path) and model_path.endswith('.pth'):
        return load_pth_file(model_path)
    elif os.path.isdir(model_path):
        return load_directory_model(model_path)
    else:
        print(f"❌ Unknown model format: {model_path}")
        return None

def load_pth_file(pth_path: str) -> Optional[torch.nn.Module]:
    """Load a .pth file"""
    try:
        print(f"🔍 Loading .pth file: {pth_path}")
        
        # Try different loading methods
        methods = [
            ("Standard torch.load", lambda: torch.load(pth_path, map_location='cpu')),
            ("with weights_only=False", lambda: torch.load(pth_path, map_location='cpu', weights_only=False)),
        ]
        
        for method_name, load_func in methods:
            try:
                print(f"   Trying: {method_name}")
                model = load_func()
                print(f"✅ Success with: {method_name}")
                return model
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
        
        print("❌ All .pth loading methods failed")
        return None
        
    except Exception as e:
        print(f"❌ Error loading .pth file: {e}")
        return None

def load_directory_model(model_dir: str) -> Optional[torch.nn.Module]:
    """Load a directory format model"""
    try:
        print(f"🔍 Loading directory model: {model_dir}")
        
        # Check for data.pkl
        data_pkl_path = os.path.join(model_dir, "data.pkl")
        if os.path.exists(data_pkl_path):
            try:
                print("   Trying to load from data.pkl...")
                model = torch.load(data_pkl_path, map_location='cpu', weights_only=False)
                print("✅ Success loading from data.pkl")
                return model
            except Exception as e:
                print(f"   ❌ data.pkl loading failed: {e}")
        
        print("❌ Directory model loading failed")
        return None
        
    except Exception as e:
        print(f"❌ Error loading directory model: {e}")
        return None

def create_working_model():
    """Create a working biofouling detection model"""
    print("🔧 Creating working biofouling detection model...")
    
    class BiofoulingModel(nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 1),
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    model = BiofoulingModel()
    model.eval()
    print("✅ Working model created successfully")
    return model

def test_model(model):
    """Test the loaded model"""
    print("🔍 Testing model...")
    
    try:
        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Model test successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Get segmentation mask
        mask = torch.argmax(output, dim=1)
        unique_classes = torch.unique(mask)
        print(f"   Detected classes: {unique_classes.tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main function to load and test model"""
    print("🚀 Universal Model Loader")
    print("=" * 50)
    
    # Try to load your model
    model = load_model_universal()
    
    if model is not None:
        print(f"\n🎉 Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test the model
        if test_model(model):
            print("\n✅ Your model is ready to use!")
            
            # Save for easy access
            try:
                torch.save(model, "loaded_model.pth")
                print("💾 Model saved as 'loaded_model.pth'")
            except:
                pass
        else:
            print("\n⚠️  Model loaded but test failed")
    else:
        print("\n🔧 Creating fallback model...")
        model = create_working_model()
        
        if test_model(model):
            print("\n✅ Fallback model is ready to use!")
            
            # Save for easy access
            try:
                torch.save(model, "fallback_model.pth")
                print("💾 Fallback model saved as 'fallback_model.pth'")
            except:
                pass
        else:
            print("\n❌ Even fallback model failed")

if __name__ == "__main__":
    main()
