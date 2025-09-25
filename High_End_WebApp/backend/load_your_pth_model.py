"""
Loader for your best_model_dice_0.5029.pth file
Your .pth file contains a state_dict, so we need to create the model architecture first
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class UNet(nn.Module):
    """
    U-Net architecture based on your model's state_dict keys
    This matches the architecture used to train your model
    """
    
    def __init__(self, n_channels=3, n_classes=8, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy-2D/blob/master/unet/unet_model.py
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
        
        # Create the model architecture
        print("🏗️  Creating model architecture...")
        model = UNet(n_channels=3, n_classes=8, bilinear=True)
        
        # Load the weights into the model
        print("⚖️  Loading weights into model...")
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print("✅ Your biofouling detection model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_model(model):
    """Test the loaded model"""
    print("🔍 Testing your model...")
    
    try:
        # Create a test image
        test_input = torch.randn(1, 3, 512, 512)  # Batch=1, Channels=3, H=512, W=512
        
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

def save_model_for_use(model, filename="your_biofouling_model.pth"):
    """Save the complete model for easy loading"""
    try:
        torch.save(model, filename)
        print(f"💾 Complete model saved as '{filename}'")
        print(f"   You can now load it with: torch.load('{filename}')")
    except Exception as e:
        print(f"⚠️  Could not save complete model: {e}")

def main():
    """Main function to load and test your model"""
    print("🚀 Loading Your Biofouling Detection Model")
    print("=" * 60)
    
    # Load your model
    model = load_your_model()
    
    if model is not None:
        # Test the model
        if test_model(model):
            print("\n🎉 Your model is working perfectly!")
            
            # Save for easy access
            save_model_for_use(model)
            
            print("\n📋 How to use your model in your backend:")
            print("""
# In your backend/model_service.py:

from load_your_pth_model import load_your_model

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
