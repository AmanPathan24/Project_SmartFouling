"""
Final loader for your exact biofouling detection model
Based on the complete state_dict analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class YourFinalBiofoulingModel(nn.Module):
    """
    Your exact model architecture based on complete state_dict analysis
    """
    
    def __init__(self, n_channels=3, n_classes=4):
        super(YourFinalBiofoulingModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial convolution (32 channels)
        self.inc = DoubleConv(n_channels, 32)
        
        # Downsampling layers with maxpool + double conv
        self.down1 = Down(32, 64)    # 64 channels
        self.down2 = Down(64, 128)   # 128 channels
        self.down3 = Down(128, 128)  # 128 channels (stays same)
        self.down4 = Down(128, 256)  # 256 channels
        
        # Upsampling layers (ConvTranspose2d)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # [256, 128, 2, 2]
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # [128, 128, 2, 2]
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # [128, 64, 2, 2]
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)    # [64, 32, 2, 2]
        
        # Conv layers (these process the upsampled features)
        self.conv1 = DoubleConv(256, 128)  # [128, 256, 3, 3] - processes up1 output
        self.conv2 = DoubleConv(256, 128)  # [128, 256, 3, 3] - processes up2 output
        self.conv3 = DoubleConv(128, 64)   # [64, 128, 3, 3] - processes up3 output
        self.conv4 = DoubleConv(64, 32)    # [32, 64, 3, 3] - processes up4 output
        
        # Output layer (4 classes)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # 32 channels
        x2 = self.down1(x1)     # 64 channels
        x3 = self.down2(x2)     # 128 channels
        x4 = self.down3(x3)     # 128 channels
        x5 = self.down4(x4)     # 256 channels
        
        # Decoder with skip connections
        # Up1: 256 -> 128
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)  # 128 + 128 = 256
        x = self.conv1(x)              # 256 -> 128
        
        # Up2: 128 -> 128
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)  # 128 + 128 = 256
        x = self.conv2(x)              # 256 -> 128
        
        # Up3: 128 -> 64
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)  # 64 + 64 = 128
        x = self.conv3(x)              # 128 -> 64
        
        # Up4: 64 -> 32
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)  # 32 + 32 = 64
        x = self.conv4(x)              # 64 -> 32
        
        # Output
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

def load_your_final_model(pth_path="/Users/yash/Desktop/mlapp/best_model_dice_0.5029.pth"):
    """
    Load your final biofouling detection model
    
    Args:
        pth_path: Path to your .pth file
        
    Returns:
        Loaded model ready for inference
    """
    print(f"🔍 Loading your final model from: {pth_path}")
    
    if not os.path.exists(pth_path):
        print(f"❌ Model file not found: {pth_path}")
        return None
    
    try:
        # Load the state_dict
        print("📥 Loading state_dict...")
        state_dict = torch.load(pth_path, map_location='cpu')
        
        print(f"✅ State_dict loaded successfully!")
        print(f"   Number of parameters: {len(state_dict)}")
        
        # Create the exact model architecture
        print("🏗️  Creating final model architecture...")
        model = YourFinalBiofoulingModel(n_channels=3, n_classes=4)
        
        # Load the weights into the model
        print("⚖️  Loading weights into model...")
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print("✅ Your final biofouling detection model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_model(model):
    """Test the loaded model"""
    print("🔍 Testing your final model...")
    
    try:
        # Create a test image
        test_input = torch.randn(1, 3, 256, 256)
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Model test successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Get segmentation mask
        mask = torch.argmax(output, dim=1)
        unique_classes = torch.unique(mask)
        print(f"   Detected classes: {unique_classes.tolist()}")
        
        # Get class probabilities
        probabilities = torch.softmax(output, dim=1)
        max_probs = torch.max(probabilities, dim=1)[0]
        print(f"   Confidence range: {max_probs.min():.4f} - {max_probs.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def save_model_for_use(model, filename="your_final_model.pth"):
    """Save the complete model for easy loading"""
    try:
        torch.save(model, filename)
        print(f"💾 Complete model saved as '{filename}'")
        print(f"   You can now load it with: torch.load('{filename}')")
    except Exception as e:
        print(f"⚠️  Could not save complete model: {e}")

def main():
    """Main function to load and test your final model"""
    print("🚀 Loading Your Final Biofouling Detection Model")
    print("=" * 60)
    
    # Load your model
    model = load_your_final_model()
    
    if model is not None:
        # Test the model
        if test_model(model):
            print("\n🎉 Your final model is working perfectly!")
            
            # Save for easy access
            save_model_for_use(model)
            
            print("\n📋 Model Details:")
            print(f"   - Input channels: 3 (RGB)")
            print(f"   - Output classes: 4")
            print(f"   - Architecture: Custom U-Net variant")
            print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            print("\n📋 How to use your model in your backend:")
            print("""
# In your backend/model_service.py:

from load_your_final_model import load_your_final_model

class ModelService:
    def __init__(self):
        self.model = load_your_final_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def predict_biofouling(self, image_tensor):
        '''Run biofouling detection on image'''
        with torch.no_grad():
            # Run inference
            output = self.model(image_tensor)
            
            # Get segmentation mask
            mask = torch.argmax(output, dim=1)
            
            # Get class probabilities
            probabilities = torch.softmax(output, dim=1)
            
            return output, mask, probabilities
            """)
        else:
            print("\n⚠️  Model loaded but test failed")
    else:
        print("\n❌ Failed to load your final model")

if __name__ == "__main__":
    main()
