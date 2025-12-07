#!/bin/bash
# Setup script for Git LFS
# This script sets up Git LFS for tracking large files in the wine recommender project

set -e

echo "🚀 Setting up Git LFS for wine recommender project..."

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed."
    echo ""
    echo "Please install Git LFS first:"
    echo ""
    echo "  macOS (using Homebrew):"
    echo "    brew install git-lfs"
    echo ""
    echo "  Or download from: https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo "✅ Git LFS is installed"

# Initialize Git LFS
echo "📦 Initializing Git LFS..."
git lfs install

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "📁 Initializing Git repository..."
    git init
fi

# Track large files
echo "🔍 Configuring Git LFS to track large files..."
git lfs track "*.json"
git lfs track "*.csv"
git lfs track "wine_embeddings.json"
git lfs track "df.csv"

# Add .gitattributes
echo "📝 Adding .gitattributes to repository..."
git add .gitattributes

echo ""
echo "✅ Git LFS setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your files: git add wine_embeddings.json df.csv"
echo "  2. Commit: git commit -m 'Add wine data and embeddings with Git LFS'"
echo "  3. Push to GitHub: git push origin main"
echo ""
echo "Note: Make sure your GitHub repository has Git LFS enabled."

