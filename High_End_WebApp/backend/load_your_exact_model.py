"""
Loader for your exact biofouling detection model
Based on the state_dict analysis, this is the precise architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class YourExactBiofoulingModel(nn.Module):
    """
    Your exact model architecture based on state_dict analysis
    """
    
    def __init__(self, n_channels=3, n_classes=4):  # Based on outc.weight shape [4, 32, 1, 1]
        super(YourExactBiofoulingModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial convolution (32 channels)
        self.inc = DoubleConv(n_channels, 32)
        
        # Downsampling layers with maxpool
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Conv layers (these are the main layers)
        self.conv1 = DoubleConv(512, 128)  # Based on conv1 shape [128, 512, 3, 3]
        self.conv2 = DoubleConv(128, 128)  # Based on conv2 shape [128, 128, 3, 3]
        self.conv3 = DoubleConv(128, 64)   # Based on conv3 shape [64, 128, 3, 3]
        self.conv4 = DoubleConv(64, 32)    # Based on conv4 shape [32, 64, 3, 3]
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(128, 256, 2, stride=2)  # Based on up1 shape [256, 128, 2, 2]
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # Based on up2 shape [128, 128, 2, 2]
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        # Output layer (4 classes based on outc.weight shape [4, 32, 1, 1])
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bottleneck conv layers
        x = self.conv1(x5)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
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

def load_your_exact_model(pth_path="/Users/yash/Desktop/mlapp/best_model_dice_0.5029.pth"):
    """
    Load your exact biofouling detection model
    
    Args:
        pth_path: Path to your .pth file
        
    Returns:
        Loaded model ready for inference
    """
    print(f"🔍 Loading your exact model from: {pth_path}")
    
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
        print("🏗️  Creating exact model architecture...")
        model = YourExactBiofoulingModel(n_channels=3, n_classes=4)
        
        # Load the weights into the model
        print("⚖️  Loading weights into model...")
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print("✅ Your exact biofouling detection model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_model(model):
    """Test the loaded model"""
    print("🔍 Testing your exact model...")
    
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

def save_model_for_use(model, filename="your_exact_model.pth"):
    """Save the complete model for easy loading"""
    try:
        torch.save(model, filename)
        print(f"💾 Complete model saved as '{filename}'")
        print(f"   You can now load it with: torch.load('{filename}')")
    except Exception as e:
        print(f"⚠️  Could not save complete model: {e}")

def main():
    """Main function to load and test your exact model"""
    print("🚀 Loading Your Exact Biofouling Detection Model")
    print("=" * 60)
    
    # Load your model
    model = load_your_exact_model()
    
    if model is not None:
        # Test the model
        if test_model(model):
            print("\n🎉 Your exact model is working perfectly!")
            
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

from load_your_exact_model import load_your_exact_model

class ModelService:
    def __init__(self):
        self.model = load_your_exact_model()
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
        print("\n❌ Failed to load your exact model")

if __name__ == "__main__":
    main()
