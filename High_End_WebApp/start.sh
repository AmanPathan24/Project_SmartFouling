#!/bin/bash

# Marine Biofouling Detection System Startup Script

echo "🚢 Starting Marine Biofouling Detection System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/data
mkdir -p backend/uploads
mkdir -p backend/processed
mkdir -p backend/outputs
mkdir -p backend/static

# Copy the model file to the backend directory
echo "🤖 Setting up PyTorch model..."
if [ -f "best_model_dice_0.5029.zip" ]; then
    echo "   Found model archive, extracting..."
    unzip -q -o best_model_dice_0.5029.zip -d backend/
    echo "   ✅ Model extracted successfully"
elif [ -d "best_model_dice_0.5029" ]; then
    echo "   Found model directory, copying..."
    cp -r best_model_dice_0.5029 backend/
    echo "   ✅ Model copied successfully"
else
    echo "   ⚠️  No model file found, will use mock model"
fi

# Copy demo images to backend static directory
echo "🖼️ Setting up demo images..."
if [ -d "webapp/public/demo" ]; then
    cp -r webapp/public/demo backend/static/ 2>/dev/null || echo "   Demo images not found, using fallback"
    echo "   ✅ Demo images set up"
else
    echo "   ⚠️  No demo images found"
fi

# Install backend dependencies
echo "🐍 Installing Python dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# Set up demo data
echo "🎭 Setting up demo data..."
python3 setup_demo.py

# Start backend server
echo "🚀 Starting backend server..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend development server
echo "🎨 Starting frontend development server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ System started successfully!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🎯 Features Available:"
echo "   • Image upload with automatic preprocessing"
echo "   • AI-powered fouling detection using PyTorch model"
echo "   • Interactive 3D visualization"
echo "   • LLM-powered PDF report generation"
echo "   • Maintenance scheduling recommendations"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for processes
wait