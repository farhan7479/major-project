#!/bin/bash

# Energy Forecasting App Startup Script

echo "🚀 Starting Energy Forecasting Application..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists npm; then
    echo "❌ Node.js and npm are required but not installed."
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Start backend
echo "🔧 Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "📥 Installing backend dependencies..."
pip install -r requirements.txt

# Start backend server in background
echo "🎯 Starting FastAPI server..."
python main.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Setup frontend
echo "🎨 Setting up frontend..."
cd ../frontend

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    npm install
fi

# Start frontend server
echo "🌐 Starting React development server..."
npm run dev &
FRONTEND_PID=$!

# Wait a bit for servers to start
sleep 5

echo ""
echo "🎉 Application started successfully!"
echo ""
echo "📍 Services running at:"
echo "   Frontend: http://localhost:5173"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🛑 To stop the application, press Ctrl+C"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🧹 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Application stopped!"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
