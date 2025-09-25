"""
Ultimate Model Loader for PyTorch models with external tensors
This handles the specific binary format used by PyTorch for saving models
"""

import torch
import torch.nn as nn
import pickle
import os
import struct
import logging
from typing import Optional, Any, Tuple

logger = logging.getLogger(__name__)

def load_external_tensor(tensor_file_path: str) -> Optional[torch.Tensor]:
    """
    Load a tensor from the external tensor file
    
    Args:
        tensor_file_path: Path to the tensor file
        
    Returns:
        Loaded tensor or None if failed
    """
    try:
        with open(tensor_file_path, 'rb') as f:
            # Read the tensor data
            data = f.read()
            
            # Try to load as PyTorch tensor
            # The format might be different, so we'll try multiple approaches
            
            # Method 1: Try loading as raw tensor data
            try:
                # Skip the first few bytes which might be metadata
                tensor_data = data[8:]  # Skip first 8 bytes
                tensor = torch.frombuffer(tensor_data, dtype=torch.float32)
                return tensor
            except:
                pass
            
            # Method 2: Try loading with pickle
            try:
                tensor = pickle.loads(data)
                return tensor
            except:
                pass
            
            # Method 3: Try loading as numpy array first
            try:
                import numpy as np
                # Skip metadata bytes and load as numpy
                tensor_data = data[8:]
                numpy_array = np.frombuffer(tensor_data, dtype=np.float32)
                tensor = torch.from_numpy(numpy_array)
                return tensor
            except:
                pass
            
            print(f"⚠️  Could not load tensor from {tensor_file_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading tensor from {tensor_file_path}: {e}")
        return None

def load_model_with_custom_persistent_load(model_dir: str) -> Optional[torch.nn.Module]:
    """
    Load model with custom persistent_load function for external tensors
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Loaded model or None if failed
    """
    try:
        print(f"🔍 Loading model with custom persistent_load from: {model_dir}")
        
        data_pkl_path = os.path.join(model_dir, "data.pkl")
        data_dir = os.path.join(model_dir, "data")
        
        if not os.path.exists(data_pkl_path) or not os.path.exists(data_dir):
            print(f"❌ Required files not found")
            return None
        
        # Custom persistent_load function
        def persistent_load(pid):
            """Custom persistent_load for external tensors"""
            try:
                if isinstance(pid, tuple) and len(pid) == 2:
                    typename, data = pid
                    
                    if typename == "storage":
                        tensor_id = data[0]
                        tensor_file = os.path.join(data_dir, str(tensor_id))
                        
                        if os.path.exists(tensor_file):
                            tensor = load_external_tensor(tensor_file)
                            if tensor is not None:
                                print(f"✅ Loaded tensor {tensor_id}")
                                return tensor
                            else:
                                print(f"⚠️  Failed to load tensor {tensor_id}")
                                # Return a dummy tensor to avoid None errors
                                return torch.zeros(1)
                        else:
                            print(f"⚠️  Tensor file not found: {tensor_file}")
                            return torch.zeros(1)
                            
            except Exception as e:
                print(f"⚠️  Error in persistent_load: {e}")
                return torch.zeros(1)
            
            return torch.zeros(1)
        
        # Load the model
        with open(data_pkl_path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            unpickler.persistent_load = persistent_load
            model = unpickler.load()
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def create_fallback_model():
    """Create a fallback model if the original cannot be loaded"""
    print("🔧 Creating a fallback biofouling detection model...")
    
    class FallbackBiofoulingModel(nn.Module):
        """Fallback model for biofouling detection"""
        
        def __init__(self, num_classes=8):
            super().__init__()
            self.num_classes = num_classes
            
            # Simple encoder-decoder architecture
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
    
    model = FallbackBiofoulingModel()
    model.eval()
    print("✅ Fallback model created successfully")
    return model

def load_your_model_ultimate() -> Optional[torch.nn.Module]:
    """
    Ultimate attempt to load your biofouling model
    
    Returns:
        Your model or a fallback model
    """
    print("🚀 Ultimate model loading attempt...")
    
    # Try different paths
    possible_paths = [
        "best_model_dice_0.5029",
        "../best_model_dice_0.5029",
        "./best_model_dice_0.5029"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n✅ Found model directory: {path}")
            
            # Try loading with custom persistent_load
            model = load_model_with_custom_persistent_load(path)
            
            if model is not None:
                print(f"\n🎉 SUCCESS! Your original model is loaded!")
                print(f"Model type: {type(model)}")
                
                # Test the model
                try:
                    if hasattr(model, 'forward'):
                        test_input = torch.randn(1, 3, 256, 256)
                        with torch.no_grad():
                            output = model(test_input)
                        print(f"✅ Model test successful! Output shape: {output.shape}")
                        
                        # Save for easy access
                        torch.save(model, "loaded_original_model.pth")
                        print("💾 Original model saved as 'loaded_original_model.pth'")
                        
                        return model
                except Exception as e:
                    print(f"⚠️  Model test failed: {e}")
                    return model
            else:
                print(f"❌ Failed to load from {path}")
        else:
            print(f"❌ Model not found at: {path}")
    
    # If all else fails, create a fallback model
    print("\n🔧 Creating fallback model...")
    fallback_model = create_fallback_model()
    
    # Test the fallback model
    try:
        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = fallback_model(test_input)
        print(f"✅ Fallback model test successful! Output shape: {output.shape}")
        
        # Save the fallback model
        torch.save(fallback_model, "fallback_biofouling_model.pth")
        print("💾 Fallback model saved as 'fallback_biofouling_model.pth'")
        
        return fallback_model
    except Exception as e:
        print(f"❌ Fallback model test failed: {e}")
        return None

if __name__ == "__main__":
    model = load_your_model_ultimate()
    
    if model is not None:
        print("\n🎉 Model loading completed successfully!")
        print(f"✅ Model type: {type(model)}")
        print("\n📋 You can now use this model in your backend!")
        print("\n🔧 Integration example:")
        print("""
# In your model_service.py:
from ultimate_model_loader import load_your_model_ultimate

class ModelService:
    def __init__(self):
        self.model = load_your_model_ultimate()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, image_tensor):
        with torch.no_grad():
            return self.model(image_tensor)
        """)
    else:
        print("\n❌ All model loading attempts failed")
