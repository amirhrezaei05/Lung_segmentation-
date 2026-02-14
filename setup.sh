#!/bin/bash
# Quick start script for lung segmentation project

echo "========================================="
echo "Lung Segmentation Project - Quick Start"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Install package in editable mode
echo "Installing package..."
pip install -e . > /dev/null 2>&1
echo "✓ Package installed"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download the dataset and place it in data/raw/"
echo "2. Start MLflow UI: mlflow ui"
echo "3. Train a model:"
echo "   python -m src.train --config configs/efficient_unet_config.yaml --data_dir data/raw/Lung\\ Segmentation"
echo ""
echo "For more information, see README.md and IMPLEMENTATION_GUIDE.md"
echo ""
