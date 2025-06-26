#!/bin/bash
set -e

echo "Setting up Pixi environments for container..."

cd /workspace

# Ensure pixi is in PATH
export PATH="/home/node/.pixi/bin:$PATH"

# Create the container environments directory
mkdir -p /workspace/.pixi-container

# Install the project and its environments
echo "Installing project with pixi (using container-specific directory)..."
if pixi install --all; then
    echo "✓ Successfully installed project and all environments"
else
    echo "✗ Failed to install project"
    exit 1
fi

# Clean up .pixi directory if it only contains symlinks (container-created)
echo ""
echo "Checking .pixi directory for symlinks..."
if [ -L "/workspace/.pixi/envs" ]; then
    echo "Found .pixi/envs as symlink - removing .pixi directory (container-created)"
    rm -rf /workspace/.pixi
    echo "✓ Removed container-created .pixi directory with symlinks"
elif [ -d "/workspace/.pixi/envs" ]; then
    echo "✓ Found .pixi/envs as real directory (host environments) - keeping it"
elif [ -d "/workspace/.pixi" ]; then
    # .pixi exists but no envs subdirectory - likely safe to remove
    echo "Found .pixi directory without envs - removing"
    rm -rf /workspace/.pixi
    echo "✓ Removed empty .pixi directory"
else
    echo "✓ No .pixi directory found - nothing to clean up"
fi

echo ""
echo "Pixi environment setup complete!"

echo ""
echo "Available environments:"
pixi info

echo ""
echo "Container environments are stored in: /workspace/.pixi-container"
echo "Host environments use the default: /workspace/.pixi"
echo ""
echo "To activate an environment, use: pixi shell -e <env-name>"
echo "Or run commands directly: pixi run -e <env-name> <command>"
echo ""
echo "Note: Container uses global pixi config for detached environments"

# Final cleanup - remove .pixi directory if it only contains symlinks
echo ""
echo "Final cleanup check..."
if [ -L "/workspace/.pixi/envs" ]; then
    echo "Found .pixi/envs as symlink - removing .pixi directory (container-created)"
    rm -rf /workspace/.pixi
    echo "✓ Removed container-created .pixi directory with symlinks"
elif [ -d "/workspace/.pixi/envs" ]; then
    echo "✓ Found .pixi/envs as real directory (host environments) - keeping it"
fi