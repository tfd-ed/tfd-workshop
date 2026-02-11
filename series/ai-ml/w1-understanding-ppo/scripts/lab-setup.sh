#!/bin/bash

# Lab Setup Script for PPO Workshop
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "PPO Workshop - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if Python 3.9 or 3.10
if [[ ! $PYTHON_VERSION =~ ^3\.(9|10) ]]; then
    echo "⚠️  Warning: Python 3.9 or 3.10 is recommended for best compatibility with Gymnasium"
    echo "   Your version: $PYTHON_VERSION"
    echo "   You may encounter compatibility issues. Continue? (y/n)"
    read -r response
    if [[ ! $response =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Choose Environment Manager"
echo "=========================================="
echo "1) Conda (recommended)"
echo "2) venv (Python's built-in)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    # Conda setup
    echo ""
    echo "Setting up Conda environment..."
    
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "❌ Error: Conda not found!"
        echo "   Please install Miniconda or Anaconda first:"
        echo "   https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    ENV_NAME="ppo-workshop"
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "Environment '$ENV_NAME' already exists."
        read -p "Remove and recreate? (y/n): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n $ENV_NAME -y
        else
            echo "Using existing environment."
        fi
    fi
    
    # Create new environment
    if ! conda env list | grep -q "^$ENV_NAME "; then
        echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
        conda create -n $ENV_NAME python=3.10 -y
    fi
    
    echo "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    # Install dependencies
    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install gymnasium[box2d]==0.29.1
    pip install numpy==1.24.3
    pip install matplotlib==3.7.2
    pip install tqdm==4.66.1
    pip install tensorboard==2.14.0
    
    echo ""
    echo "✅ Conda environment setup complete!"
    echo ""
    echo "To activate this environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To deactivate:"
    echo "  conda deactivate"
    
elif [ "$choice" == "2" ]; then
    # venv setup
    echo ""
    echo "Setting up Python virtual environment..."
    
    ENV_NAME="ppo-workshop"
    
    # Remove existing environment if it exists
    if [ -d "$ENV_NAME" ]; then
        echo "Virtual environment '$ENV_NAME' already exists."
        read -p "Remove and recreate? (y/n): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            rm -rf $ENV_NAME
        else
            echo "Using existing environment."
        fi
    fi
    
    # Create new environment
    if [ ! -d "$ENV_NAME" ]; then
        echo "Creating virtual environment '$ENV_NAME'..."
        python3 -m venv $ENV_NAME
    fi
    
    echo "Activating environment..."
    source $ENV_NAME/bin/activate
    
    # Install dependencies
    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install gymnasium[box2d]==0.29.1
    pip install numpy==1.24.3
    pip install matplotlib==3.7.2
    pip install tqdm==4.66.1
    pip install tensorboard==2.14.0
    
    echo ""
    echo "✅ Virtual environment setup complete!"
    echo ""
    echo "To activate this environment, run:"
    echo "  source $ENV_NAME/bin/activate"
    echo ""
    echo "To deactivate:"
    echo "  deactivate"
    
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python3 << END
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not found")
    sys.exit(1)

try:
    import gymnasium
    print(f"✅ Gymnasium {gymnasium.__version__}")
except ImportError:
    print("❌ Gymnasium not found")
    sys.exit(1)

try:
    import numpy
    print(f"✅ NumPy {numpy.__version__}")
except ImportError:
    print("❌ NumPy not found")
    sys.exit(1)

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("❌ Matplotlib not found")
    sys.exit(1)

try:
    import tqdm
    print(f"✅ tqdm {tqdm.__version__}")
except ImportError:
    print("❌ tqdm not found")
    sys.exit(1)

# Test Gymnasium environment
try:
    env = gymnasium.make('LunarLander-v2')
    state, _ = env.reset()
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    env.close()
    print("✅ LunarLander-v2 environment working")
except Exception as e:
    print(f"❌ LunarLander-v2 environment failed: {e}")
    sys.exit(1)

print("\n✅ All dependencies installed and working correctly!")
END

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You're ready to start the workshop exercises!"
echo ""
echo "Next steps:"
echo "  1. Navigate to the exercises directory:"
echo "     cd exercises/"
echo ""
echo "  2. Start with Exercise 1:"
echo "     python exercise1_explore_env.py"
echo ""
echo "Happy learning! 🚀"
echo ""
