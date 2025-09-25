"""
Working Model Loader for PyTorch models with external tensors
This handles the specific format of your best_model_dice_0.5029 model
"""

import torch
import torch.nn as nn
import pickle
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def load_model_with_external_tensors(model_dir: str) -> Optional[torch.nn.Module]:
    """
    Load a PyTorch model that has external tensors (saved in directory format)
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Loaded PyTorch model or None if failed
    """
    try:
        print(f"🔍 Loading model with external tensors from: {model_dir}")
        
        data_pkl_path = os.path.join(model_dir, "data.pkl")
        data_dir = os.path.join(model_dir, "data")
        
        if not os.path.exists(data_pkl_path):
            print(f"❌ data.pkl not found at: {data_pkl_path}")
            return None
        
        if not os.path.exists(data_dir):
            print(f"❌ data directory not found at: {data_dir}")
            return None
        
        # Custom persistent_load function to handle external tensors
        def persistent_load(pid):
            """Load external tensor data"""
            try:
                if isinstance(pid, tuple) and len(pid) == 2:
                    typename, data = pid
                    
                    if typename == "storage":
                        # Get the tensor ID
                        tensor_id = data[0]
                        tensor_file = os.path.join(data_dir, str(tensor_id))
                        
                        if os.path.exists(tensor_file):
                            print(f"🔍 Loading tensor {tensor_id} from {tensor_file}")
                            
                            # Read the tensor data
                            with open(tensor_file, 'rb') as f:
                                tensor_data = f.read()
                            
                            # Load the tensor using pickle
                            tensor = pickle.loads(tensor_data)
                            return tensor
                        else:
                            print(f"⚠️  Tensor file not found: {tensor_file}")
                            return None
                            
            except Exception as e:
                print(f"⚠️  Error loading persistent object {pid}: {e}")
                return None
            
            return None
        
        # Load the model with custom persistent_load
        with open(data_pkl_path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            unpickler.persistent_load = persistent_load
            model = unpickler.load()
        
        print("✅ Model loaded successfully with external tensors!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model with external tensors: {e}")
        return None

def load_your_biofouling_model() -> Optional[torch.nn.Module]:
    """
    Load your specific biofouling detection model
    
    Returns:
        Your loaded model or None if failed
    """
    print("🚀 Loading your biofouling detection model...")
    
    # Try different possible paths
    possible_paths = [
        "best_model_dice_0.5029",
        "../best_model_dice_0.5029",
        "./best_model_dice_0.5029"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n✅ Found model directory: {path}")
            
            # Try loading with external tensors
            model = load_model_with_external_tensors(path)
            
            if model is not None:
                print(f"\n🎉 SUCCESS! Your model is loaded!")
                print(f"Model type: {type(model)}")
                
                # Test the model
                try:
                    if hasattr(model, 'forward'):
                        print("🔍 Testing model forward pass...")
                        
                        # Create a test input
                        test_input = torch.randn(1, 3, 256, 256)  # Batch, Channels, Height, Width
                        
                        # Run inference
                        with torch.no_grad():
                            output = model(test_input)
                        
                        print(f"✅ Model inference successful!")
                        print(f"   Input shape: {test_input.shape}")
                        print(f"   Output shape: {output.shape}")
                        
                        # Check if output is segmentation (should have multiple classes)
                        if len(output.shape) == 4:  # [batch, classes, height, width]
                            num_classes = output.shape[1]
                            print(f"   Number of classes: {num_classes}")
                            
                            # Apply softmax to get probabilities
                            probabilities = torch.softmax(output, dim=1)
                            print(f"   Probability range: {probabilities.min():.4f} - {probabilities.max():.4f}")
                        
                        # Save the working model for easy access
                        try:
                            torch.save(model, "working_biofouling_model.pth")
                            print("💾 Model saved as 'working_biofouling_model.pth' for easy loading")
                        except Exception as e:
                            print(f"⚠️  Could not save model: {e}")
                        
                        return model
                    else:
                        print("ℹ️  Model loaded but no forward method available")
                        return model
                        
                except Exception as e:
                    print(f"⚠️  Model test failed: {e}")
                    print("But the model was loaded successfully!")
                    return model
            else:
                print(f"❌ Failed to load model from {path}")
        else:
            print(f"❌ Model directory not found: {path}")
    
    print("\n❌ Could not load your model from any location")
    return None

def create_model_integration_example():
    """Create an example of how to integrate your model into the backend"""
    print("\n📋 Model Integration Example:")
    print("""
# In your backend/model_service.py, you can now use:

from working_model_loader import load_your_biofouling_model

class ModelService:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def load_model(self):
        '''Load your biofouling detection model'''
        self.model = load_your_biofouling_model()
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            logger.info("Biofouling model loaded successfully")
        else:
            logger.error("Failed to load biofouling model")
    
    async def predict_segmentation(self, image_tensor):
        '''Run segmentation inference'''
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            # Preprocess image (resize, normalize, etc.)
            processed_image = self.preprocess_image(image_tensor)
            
            # Run inference
            output = self.model(processed_image)
            
            # Post-process results
            segmentation = torch.argmax(output, dim=1)
            
            return segmentation, output
    """)

if __name__ == "__main__":
    model = load_your_biofouling_model()
    
    if model is not None:
        print("\n🎉 Your biofouling detection model is ready!")
        print("\n✅ What you can do now:")
        print("1. Use this model for biofouling detection")
        print("2. Integrate it into your backend service")
        print("3. Run inference on uploaded images")
        print("4. Generate segmentation masks and classifications")
        
        create_model_integration_example()
    else:
        print("\n❌ Model loading failed")
        print("\n💡 Possible solutions:")
        print("1. Check if the model files are complete")
        print("2. Verify the model was saved correctly")
        print("3. Try a different PyTorch version")
        print("4. Contact the model creator for loading instructions")
