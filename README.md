# 🚢 Marine Biofouling AI Platform

A comprehensive AI-powered platform for **detecting, analyzing, and managing marine biofouling** on ships and submarines.  
This repository combines two complementary systems:

1. **Marine Biofouling Detection System** – End-to-end application with preprocessing, ML analysis, dashboards, and 3D ship visualization.  
2. **Advanced Biofouling Analysis System** – Research-oriented platform with segmentation, classification, analytics, and 3D submarine heatmaps.

---

## 🌟 Key Features

### 🔹 Core Functionality
- **Image Upload & Processing** with advanced preprocessing pipeline (CLAHE, Retinex, noise reduction)
- **AI-Powered Analysis**:
  - **Segmentation (U-Net)** for fouling detection and coverage mapping
  - **Classification (CNN)** for multi-species identification (11 categories)
- **3D Visualization**:
  - Ship hull overlay (Three.js + React)
  - Submarine heatmap mapping (Streamlit + Three.js)
- **Analytics Dashboards** with 15+ interactive charts (species distribution, cost impact, coverage trends)
- **Maintenance Management** with scheduling, task tracking, and cost projections
- **Reporting**: Comprehensive reports with visualizations and metrics

### 🔹 Technical Highlights
- Detection of **7+ ship fouling species** (Barnacles, Mussels, Seaweed, etc.)  
- Classification of **11 categories** (Algae, Barnacles, Mussels, Rust, Worms, etc.)  
- **Model Performance**:
  - Segmentation Dice ≈ 0.50
  - Classification Accuracy ~94%  
- GPU/CPU compatible, with processing time ~2–5s (GPU) or ~10–15s (CPU)

---

## 🏗️ Architecture

### Backend (FastAPI + PyTorch)
backend/
├── main.py
├── preprocessing_service.py
├── model_service.py
├── database.py
└── utils.py


### Frontend (React + TypeScript + Vite)
frontend/
├── components/
│ ├── UploadPanel.tsx
│ ├── ResultsPanel.tsx
│ ├── AnalyticsPanel.tsx
│ ├── Ship3DView.tsx
│ └── MaintenancePanel.tsx
├── App.tsx
└── main.tsx

---

### Research App (Streamlit)
Biofouling/
├── app.py
├── models.py
├── Models/ (Segmentation)
├── classimodel1/ (Classification)
└── submarine.glb (3D model)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA GPU (optional)
- Git

### Installation
```bash
git clone <repository-url>
cd Biofouling

cd backend
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py


cd frontend
npm install
npm run dev

pip install -r requirements.txt
streamlit run app.py

# 📖 Usage Workflow

1. **Upload Images** → Preprocessing → AI Analysis *(Segmentation/Classification)*
2. **View Results** → Original, Preprocessed, and Masked Images
3. **Analytics** → Interactive charts on fouling density, costs, species distribution
4. **3D Visualization** → Ship hull overlay or Submarine heatmap exploration
5. **Maintenance** → Schedule tasks, project costs, and track maintenance

---

# 📊 Supported Species

### **Detection (7+)**
- Barnacles  
- Mussels  
- Seaweed  
- Sponges  
- Anemones  
- Tunicates  
- Other fouling  

### **Classification (11)**
- Algae 🌿  
- Barnacles 🦪  
- Clean ✨  
- Hydrozoan 🪼  
- Jellyfish 🎐  
- Mussels 🦪  
- Rust 🟤  
- Starfish ⭐  
- Worms 🪱  
- Zebra Mussels 🦓  
- Tunicates 🫧  

---

# 🔧 Troubleshooting

- **Backend not starting** → Check Python 3.8+, dependencies, port 8000 availability  
- **Frontend not loading** → Ensure Node.js 16+, run `npm install`, free port 3000  
- **Model errors** → Verify `.pth` files in `Models/` and `classimodel1/`  
- **3D issues** → Check WebGL support (`chrome://gpu`) and presence of `submarine.glb`  

Enable debug logs:  
```bash
export DEBUG=1
export LOG_LEVEL=DEBUG

