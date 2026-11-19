#!/bin/bash
# Run once on the VM to install dependencies
set -e

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install google-genai openai fal-client soundfile numpy pandas scipy

echo "Creating directories..."
mkdir -p study2_stimuli
mkdir -p study2_results
mkdir -p study2_checkpoints

echo "Done! Next steps:"
echo "  1. Edit config.env with your API keys"
echo "  2. Upload study2_stimuli/ files"
echo "  3. Run: python run_experiment.py"
