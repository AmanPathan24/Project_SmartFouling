# 🐚 Advanced Biofouling Analysis System

A comprehensive marine biofouling analysis platform combining state-of-the-art deep learning models with interactive visualizations for segmentation, classification, and 3D heatmap generation.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This application provides dual-mode analysis for marine biofouling research and monitoring:

- **🔬 Segmentation Mode**: Pixel-level detection and quantification of biofouling regions
- **🧬 Classification Mode**: Species identification across 11 different biofouling categories
- **🚢 3D Visualization**: Interactive submarine heatmap with real-time gradient mapping

## ✨ Key Features

### 🎯 **Dual Analysis Modes**
- **Segmentation**: U-Net-based pixel-level biofouling detection
- **Classification**: CNN-based species identification
- **Flexible workflow**: Analyze different images for different purposes

### 📊 **Comprehensive Data Analytics**
- **15+ Interactive Charts**: Pie charts, bar graphs, heatmaps, and spatial distributions
- **Advanced Metrics**: Connected components, fragmentation analysis, coverage statistics
- **Multi-dimensional Analysis**: Spatial, temporal, and intensity-based insights

### 🚢 **3D Submarine Visualization**
- **Three.js WebGL Renderer**: Professional 3D visualization with natural proportions
- **Heat Gradient Mapping**: Real-time biofouling intensity visualization
- **Interactive Controls**: Full 3D navigation with orbit controls
- **Embedded Model Support**: Direct GLB model loading with base64 encoding

### 🎨 **Professional Interface**
- **Responsive Design**: Adapts to different screen sizes
- **Custom Styling**: Marine-themed color scheme
- **Interactive Elements**: Collapsible sections, hover effects, and real-time feedback

## 🏗️ Architecture

### **Core Components**

```
├── 🎯 Analysis Modes
│   ├── Segmentation (U-Net Architecture)
│   └── Classification (Multi-label CNN)
├── 📊 Data Visualization
│   ├── Statistical Analysis (15+ charts)
│   ├── Spatial Distribution
│   └── Advanced Metrics
├── 🚢 3D Visualization
│   ├── Three.js Renderer
│   ├── GLB Model Loading
│   └── Heat Gradient Mapping
└── 🎨 User Interface
    ├── Streamlit Framework
    ├── Custom CSS Styling
    └── Responsive Components
```

### **Deep Learning Models**

#### **Segmentation Model (U-Net)**
- **Architecture**: U-Net with skip connections
- **Input**: 256×256 RGB images (normalized)
- **Output**: 4-class segmentation mask
- **Framework**: PyTorch
- **Model File**: `Models/best_model_dice_0.5029.pth`

#### **Classification Model (CNN)**
- **Architecture**: Multi-label Convolutional Neural Network
- **Input**: 256×256 RGB images (normalized)
- **Classes**: 11 species categories
- **Framework**: PyTorch
- **Model File**: `classimodel1/best_biofouling_model.pth`

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- CUDA-compatible GPU (optional, falls back to CPU)
- Modern web browser with WebGL support

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Biofouling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   ```
   ├── Models/
   │   └── best_model_dice_0.5029.pth
   ├── classimodel1/
   │   └── best_biofouling_model.pth
   └── submarine.glb (for 3D visualization)
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### **1. Select Analysis Mode**
Choose between **Segmentation** or **Classification** in the sidebar based on your research needs.

### **2. Upload Image**
- **Supported formats**: PNG, JPG, JPEG
- **Recommended**: Clear underwater/marine images with visible biofouling
- **Processing**: Images are automatically preprocessed and normalized

### **3. View Results**

#### **Segmentation Mode Output**
- **a) Mask Overlay**: Biofouling regions overlaid on original image
- **b) Mask Outline**: Boundary detection of biofouling areas
- **c) Raw Mask**: Direct segmentation output with class colors
- **d) Area Ratio**: Quantitative coverage measurement (0-1)

#### **Classification Mode Output**
- **Species Identification**: Predicted biofouling species with confidence
- **Species Description**: Detailed information about detected organism
- **Complete Category Grid**: All 11 recognized species with descriptions

### **4. Explore Analytics**
Segmentation mode provides comprehensive data analysis:
- **Pixel Distribution**: Pie charts and bar graphs
- **Spatial Analysis**: Horizontal and vertical distribution patterns
- **Statistical Metrics**: Connectivity, fragmentation, and coverage analysis
- **Advanced Visualizations**: Heatmaps, quadrant analysis, and radial distributions

### **5. 3D Submarine Visualization**
- **Interactive Model**: Rotate, zoom, and explore the submarine
- **Heat Gradients**: Real-time biofouling intensity mapping
- **Natural Proportions**: Authentic submarine shape preservation

## 🔬 Species Categories

The classification system recognizes 11 distinct biofouling categories:

| Species | Description | Icon |
|---------|-------------|------|
| **Algae** | Marine plant organisms | 🌿 |
| **Barnacles** | Small marine crustaceans | 🦪 |
| **Clean** | No significant biofouling | ✨ |
| **Hydrozoan** | Colonial marine organisms | 🪼 |
| **Jellyfish** | Gelatinous marine animals | 🎐 |
| **Mussels** | Bivalve mollusks | 🦪 |
| **Rust** | Corrosion on metal surfaces | 🟤 |
| **Starfish** | Echinoderms | ⭐ |
| **Worms** | Marine worm species | 🪱 |
| **Zebra Mussels** | Invasive freshwater mussels | 🦓 |
| **Tunicates** | Marine filter feeders | 🫧 |

## 🎨 Technical Details

### **Data Processing Pipeline**
```python
Input Image → Preprocessing → Model Inference → Post-processing → Visualization
```

### **Segmentation Pipeline**
1. **Image Preprocessing**: Resize to 256×256, RGB conversion, normalization
2. **U-Net Inference**: 4-class segmentation prediction
3. **Post-processing**: Mask overlay generation, outline detection, area calculation
4. **Visualization**: Multiple output formats and comprehensive analytics

### **Classification Pipeline**
1. **Image Preprocessing**: Resize to 256×256, RGB conversion, normalization
2. **CNN Inference**: Multi-class species prediction
3. **Post-processing**: Confidence scoring and species mapping
4. **Visualization**: Species identification with detailed descriptions

### **3D Visualization Pipeline**
1. **Model Loading**: GLB file embedded as base64 data URL
2. **Segment Mapping**: Area ratio mapped to specific submarine segments
3. **Heat Generation**: Gradient calculation with Green→Yellow→Red color scale
4. **WebGL Rendering**: Real-time 3D visualization with orbit controls

## 📊 Performance Metrics

### **Model Performance**
- **Segmentation Model**: Dice coefficient of 0.5029
- **Classification Model**: Multi-class accuracy optimized for biofouling species
- **Processing Time**: ~2-5 seconds per image (GPU), ~10-15 seconds (CPU)

### **System Requirements**
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **GPU**: CUDA-compatible (optional)
- **Browser**: Modern browser with WebGL 2.0 support

## 🛠️ Development

### **Project Structure**
```
Biofouling/
├── app.py                     # Main Streamlit application
├── models.py                  # Model definitions and inference
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Models/                    # Segmentation model directory
│   └── best_model_dice_0.5029.pth
├── classimodel1/              # Classification model directory
│   └── best_biofouling_model.pth
└── submarine.glb              # 3D submarine model
```

### **Key Dependencies**
```python
streamlit>=1.28.0       # Web application framework
torch>=1.13.0           # Deep learning framework
torchvision>=0.14.0     # Computer vision utilities
plotly>=5.15.0          # Interactive visualizations
opencv-python>=4.7.0    # Image processing
numpy>=1.21.0           # Numerical computing
matplotlib>=3.5.0       # Static plotting
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Model Loading Errors**
```bash
# Verify model files exist
ls -la Models/
ls -la classimodel1/
```

#### **CUDA/GPU Issues**
- Application automatically falls back to CPU if GPU unavailable
- Check PyTorch CUDA installation: `torch.cuda.is_available()`

#### **3D Visualization Problems**
- Ensure `submarine.glb` is in the root directory
- Check browser WebGL support: Visit `chrome://gpu/`
- Clear browser cache and reload

#### **Memory Issues**
- Use smaller images or reduce batch processing
- Close other applications to free RAM
- Consider using CPU-only mode for lower memory usage

### **Debug Information**
Enable debug logging by checking browser console (F12 → Console) for:
- Model loading status
- 3D visualization errors
- Processing time metrics

## 📈 Future Enhancements

### **Planned Features**
- [ ] **Batch Processing**: Multiple image analysis
- [ ] **Export Functionality**: PDF reports and data export
- [ ] **Model Fine-tuning**: Custom model training interface
- [ ] **Real-time Processing**: Webcam/video stream analysis
- [ ] **Database Integration**: Results storage and retrieval
- [ ] **API Endpoints**: RESTful API for external integrations

### **Research Applications**
- **Marine Biology**: Species diversity and distribution studies
- **Naval Engineering**: Hull maintenance and fouling prevention
- **Environmental Monitoring**: Ecosystem health assessment
- **Aquaculture**: Biofouling impact on marine farming

## 🤝 Contributing

We welcome contributions from the marine research community! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Streamlit Team** for the web application framework
- **Three.js Community** for 3D visualization capabilities
- **Marine Research Community** for dataset contributions and validation

## 📞 Contact

For questions, suggestions, or collaborations:

- **Project Repository**: [GitHub Link]
- **Issues**: [GitHub Issues]
- **Documentation**: [Documentation Link]

---

<div align="center">
  <p><strong>🐚 Advanced Biofouling Analysis System</strong></p>
  <p><em>Empowering marine research through AI-driven analysis</em></p>
</div>
