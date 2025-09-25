"""
Proper Model Loader for PyTorch directory format models
Handles loading models saved with torch.save() in directory format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def load_model_from_directory(model_dir: str) -> Optional[torch.nn.Module]:
    """
    Load a PyTorch model saved in directory format
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Loaded PyTorch model or None if failed
    """
    try:
        print(f"🔍 Attempting to load model from: {model_dir}")
        
        # Method 1: Try loading the entire directory as a model
        try:
            # Use torch.load with proper error handling
            model = torch.load(model_dir, map_location='cpu', weights_only=False)
            print(f"✅ Model loaded successfully using torch.load")
            return model
        except Exception as e:
            print(f"❌ torch.load failed: {e}")
        
        # Method 2: Try to load from the data directory structure
        try:
            # Load the model from the PyTorch directory format
            # This is the newer PyTorch format for saving models
            model = torch.jit.load(model_dir, map_location='cpu')
            print(f"✅ Model loaded successfully using torch.jit.load")
            return model
        except Exception as e:
            print(f"❌ torch.jit.load failed: {e}")
        
        # Method 3: Try to load using the newer PyTorch format
        try:
            # This handles the case where the model was saved with torch.save()
            # in the newer directory format
            import pickle
            import io
            
            # Try to read the data.pkl file with proper handling
            data_pkl_path = os.path.join(model_dir, "data.pkl")
            if os.path.exists(data_pkl_path):
                print(f"🔍 Found data.pkl at: {data_pkl_path}")
                
                # Use torch.load with proper pickle handling
                with open(data_pkl_path, 'rb') as f:
                    model = torch.load(f, map_location='cpu', weights_only=False)
                
                print(f"✅ Model loaded successfully from data.pkl")
                return model
                
        except Exception as e:
            print(f"❌ Loading from data.pkl failed: {e}")
        
        # Method 4: Try to reconstruct from the data directory
        try:
            data_dir = os.path.join(model_dir, "data")
            if os.path.exists(data_dir):
                print(f"🔍 Found data directory at: {data_dir}")
                
                # Try to load individual tensor files
                # This is for models saved in the new format
                model_files = [f for f in os.listdir(data_dir) if f.isdigit()]
                if model_files:
                    print(f"🔍 Found {len(model_files)} model files")
                    
                    # Try to load as a complete model
                    # This is a more advanced approach for the new format
                    try:
                        # Load the model using the newer PyTorch loading mechanism
                        model = torch.load(data_dir, map_location='cpu', weights_only=False)
                        print(f"✅ Model loaded successfully from data directory")
                        return model
                    except Exception as e:
                        print(f"❌ Loading from data directory failed: {e}")
                        
        except Exception as e:
            print(f"❌ Data directory processing failed: {e}")
        
        print("❌ All loading methods failed")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}")
        return None

def inspect_model_directory(model_dir: str):
    """Inspect the model directory structure"""
    print(f"\n🔍 Inspecting model directory: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory does not exist: {model_dir}")
        return
    
    print(f"📁 Directory contents:")
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
            # List contents of subdirectory
            try:
                sub_items = os.listdir(item_path)
                for sub_item in sub_items[:10]:  # Show first 10 items
                    print(f"      📄 {sub_item}")
                if len(sub_items) > 10:
                    print(f"      ... and {len(sub_items) - 10} more files")
            except PermissionError:
                print(f"      (permission denied)")
        else:
            size = os.path.getsize(item_path)
            print(f"   📄 {item} ({size} bytes)")
    
    # Check if it's a PyTorch model directory
    data_pkl = os.path.join(model_dir, "data.pkl")
    data_dir = os.path.join(model_dir, "data")
    
    if os.path.exists(data_pkl):
        print(f"\n✅ Found data.pkl - this is a PyTorch model directory")
        try:
            size = os.path.getsize(data_pkl)
            print(f"   Size: {size} bytes")
        except:
            pass
    
    if os.path.exists(data_dir):
        print(f"✅ Found data/ directory - this is a PyTorch model directory")
        try:
            file_count = len(os.listdir(data_dir))
            print(f"   Contains {file_count} files")
        except:
            pass

def test_model_loading():
    """Test function to verify model loading"""
    print("🚀 Testing PyTorch model loading...")
    
    # Check different possible paths
    possible_paths = [
        "best_model_dice_0.5029",
        "../best_model_dice_0.5029",
        "./best_model_dice_0.5029"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n✅ Found model at: {path}")
            inspect_model_directory(path)
            
            model = load_model_from_directory(path)
            if model is not None:
                print(f"\n🎉 Model loaded successfully!")
                print(f"Model type: {type(model)}")
                
                # Test a dummy forward pass if possible
                try:
                    if hasattr(model, 'forward'):
                        dummy_input = torch.randn(1, 3, 512, 512)
                        with torch.no_grad():
                            output = model(dummy_input)
                        print(f"✅ Forward pass successful! Output shape: {output.shape}")
                    else:
                        print("ℹ️  Model doesn't have forward method (might be state_dict)")
                except Exception as e:
                    print(f"⚠️  Forward pass failed: {e}")
                
                return model
            else:
                print(f"❌ Failed to load model from {path}")
        else:
            print(f"❌ Model not found at: {path}")
    
    print("\n❌ No model could be loaded")
    return None

if __name__ == "__main__":
    model = test_model_loading()
    
    if model is not None:
        print("\n🎉 Model loading test completed successfully!")
    else:
        print("\n❌ Model loading test failed!")
        print("\n💡 Suggestions:")
        print("1. Make sure the model was saved with torch.save()")
        print("2. Check if the model format is compatible")
        print("3. Try saving the model in a different format")
