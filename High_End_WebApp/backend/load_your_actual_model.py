"""
Loader for your actual biofouling detection model
Based on the state_dict keys, this is a custom U-Net architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class YourBiofoulingModel(nn.Module):
    """
    Your actual model architecture based on the state_dict keys
    This is a custom U-Net variant
    """
    
    def __init__(self, n_channels=3, n_classes=8):
        super(YourBiofoulingModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial convolution (32 channels instead of 64)
        self.inc = DoubleConv(n_channels, 32)
        
        # Downsampling layers
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Upsampling layers
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        
        # Output convolution
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
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

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def load_your_model(pth_path="/Users/yash/Desktop/mlapp/best_model_dice_0.5029.pth"):
    """
    Load your biofouling detection model from the .pth file
    
    Args:
        pth_path: Path to your .pth file
        
    Returns:
        Loaded model ready for inference
    """
    print(f"🔍 Loading your model from: {pth_path}")
    
    if not os.path.exists(pth_path):
        print(f"❌ Model file not found: {pth_path}")
        return None
    
    try:
        # Load the state_dict
        print("📥 Loading state_dict...")
        state_dict = torch.load(pth_path, map_location='cpu')
        
        print(f"✅ State_dict loaded successfully!")
        print(f"   Number of parameters: {len(state_dict)}")
        
        # Print some keys to understand the structure
        print("🔍 Analyzing model structure...")
        keys = list(state_dict.keys())
        print(f"   First few keys: {keys[:5]}")
        print(f"   Last few keys: {keys[-5:]}")
        
        # Create the model architecture
        print("🏗️  Creating model architecture...")
        model = YourBiofoulingModel(n_channels=3, n_classes=8)
        
        # Load the weights into the model
        print("⚖️  Loading weights into model...")
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print("✅ Your biofouling detection model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n🔍 Let's try a different approach...")
        
        # Try to create a model that matches the exact structure
        return create_matching_model(state_dict)

def create_matching_model(state_dict):
    """Create a model that exactly matches the state_dict structure"""
    print("🔧 Creating model to match your exact architecture...")
    
    class ExactModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Based on the state_dict keys, create the exact structure
            self.inc = DoubleConv(3, 32)
            
            # Create conv layers that match the keys
            self.conv1 = DoubleConv(32, 64)
            self.conv2 = DoubleConv(64, 128)
            self.conv3 = DoubleConv(128, 256)
            self.conv4 = DoubleConv(256, 512)
            
            # Upsampling layers
            self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            
            # Output layer
            self.outc = nn.Conv2d(32, 8, 1)
            
        def forward(self, x):
            # This is a simplified forward pass - you may need to adjust based on your model
            x1 = self.inc(x)
            x2 = self.conv1(x1)
            x3 = self.conv2(x2)
            x4 = self.conv3(x3)
            x5 = self.conv4(x4)
            
            # Upsample and combine
            x = self.up1(x5)
            x = torch.cat([x, x4], dim=1)
            x = self.up2(x)
            x = torch.cat([x, x3], dim=1)
            x = self.up3(x)
            x = torch.cat([x, x2], dim=1)
            x = self.up4(x)
            x = torch.cat([x, x1], dim=1)
            
            return self.outc(x)
    
    try:
        model = ExactModel()
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model created to match your architecture!")
        return model
    except Exception as e:
        print(f"❌ Still failed to match architecture: {e}")
        return None

def test_model(model):
    """Test the loaded model"""
    print("🔍 Testing your model...")
    
    try:
        # Create a test image
        test_input = torch.randn(1, 3, 256, 256)  # Smaller size for testing
        
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

def main():
    """Main function to load and test your model"""
    print("🚀 Loading Your Actual Biofouling Detection Model")
    print("=" * 60)
    
    # Load your model
    model = load_your_model()
    
    if model is not None:
        # Test the model
        if test_model(model):
            print("\n🎉 Your model is working perfectly!")
            
            # Save for easy access
            try:
                torch.save(model, "your_working_model.pth")
                print("💾 Complete model saved as 'your_working_model.pth'")
            except:
                pass
            
            print("\n📋 How to use your model in your backend:")
            print("""
# In your backend/model_service.py:

from load_your_actual_model import load_your_model

class ModelService:
    def __init__(self):
        self.model = load_your_model()
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
        print("\n❌ Failed to load your model")

if __name__ == "__main__":
    main()
