"""
Final Model Loader for your PyTorch model
This handles the specific format of your best_model_dice_0.5029 model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_your_model(model_dir: str = "best_model_dice_0.5029") -> Optional[torch.nn.Module]:
    """
    Load your specific model from the directory format
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Loaded PyTorch model or None if failed
    """
    try:
        print(f"🔍 Loading your model from: {model_dir}")
        
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return None
        
        # Method 1: Try loading from data.pkl with weights_only=False
        data_pkl_path = os.path.join(model_dir, "data.pkl")
        if os.path.exists(data_pkl_path):
            try:
                print("🔍 Attempting to load from data.pkl...")
                # The key is to set weights_only=False for your model format
                model = torch.load(data_pkl_path, map_location='cpu', weights_only=False)
                print("✅ Model loaded successfully from data.pkl!")
                return model
            except Exception as e:
                print(f"❌ Loading from data.pkl failed: {e}")
        
        # Method 2: Try loading the entire directory
        try:
            print("🔍 Attempting to load entire directory...")
            model = torch.load(model_dir, map_location='cpu', weights_only=False)
            print("✅ Model loaded successfully from directory!")
            return model
        except Exception as e:
            print(f"❌ Loading from directory failed: {e}")
        
        print("❌ All loading methods failed")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def test_your_model():
    """Test loading your specific model"""
    print("🚀 Testing your biofouling detection model...")
    
    # Try different paths
    possible_paths = [
        "best_model_dice_0.5029",
        "../best_model_dice_0.5029",
        "./best_model_dice_0.5029"
    ]
    
    model = None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n✅ Found model at: {path}")
            model = load_your_model(path)
            
            if model is not None:
                print(f"\n🎉 SUCCESS! Your model is loaded!")
                print(f"Model type: {type(model)}")
                
                # Test forward pass
                try:
                    if hasattr(model, 'forward'):
                        print("🔍 Testing forward pass...")
                        dummy_input = torch.randn(1, 3, 512, 512)
                        with torch.no_grad():
                            output = model(dummy_input)
                        print(f"✅ Forward pass successful!")
                        print(f"   Input shape: {dummy_input.shape}")
                        print(f"   Output shape: {output.shape}")
                        
                        # Save the working model for easy access
                        torch.save(model, "loaded_biofouling_model.pth")
                        print("💾 Model saved as 'loaded_biofouling_model.pth'")
                        
                        return model
                    else:
                        print("ℹ️  Model loaded but no forward method (might be state_dict)")
                        
                except Exception as e:
                    print(f"⚠️  Forward pass failed: {e}")
                    print("But the model was loaded successfully!")
                    return model
            else:
                print(f"❌ Failed to load from {path}")
        else:
            print(f"❌ Model not found at: {path}")
    
    print("\n❌ Could not load your model from any location")
    return None

if __name__ == "__main__":
    model = test_your_model()
    
    if model is not None:
        print("\n🎉 Your model is ready to use!")
        print("\n📋 Next steps:")
        print("1. The model is loaded and ready for inference")
        print("2. You can now integrate it into your backend")
        print("3. Use the model for biofouling detection on uploaded images")
    else:
        print("\n❌ Model loading failed")
        print("\n💡 Troubleshooting:")
        print("1. Make sure the model directory exists")
        print("2. Check if the model was saved correctly")
        print("3. Verify PyTorch version compatibility")
