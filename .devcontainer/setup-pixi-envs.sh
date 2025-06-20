#!/bin/bash
set -e

echo "Setting up Pixi environments..."

cd /workspace

# Ensure pixi is in PATH
export PATH="/home/node/.pixi/bin:$PATH"

# Install the project and its environments
echo "Installing project with pixi..."
echo "Installing all environments..."
if pixi install --all; then
    echo "✓ Successfully installed project and all environments"
else
    echo "✗ Failed to install project"
    exit 1
fi

echo "Pixi environment setup complete!"

# Show what was created
echo ""
echo "Available environments:"
pixi info

echo ""
echo "To activate an environment, use: pixi shell -e <env-name>"
echo "Or run commands directly: pixi run -e <env-name> <command>"