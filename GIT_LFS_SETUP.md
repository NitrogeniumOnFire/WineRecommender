# Git LFS Setup Guide

This guide will help you set up Git LFS to track large files (wine_embeddings.json ~934MB and df.csv ~45MB) in your GitHub repository.

## Step 1: Install Xcode Command Line Tools (if needed)

If you see errors about "No developer tools were found", install Xcode Command Line Tools:

```bash
xcode-select --install
```

Follow the installation prompts. This may take 10-15 minutes.

## Step 2: Install Git LFS

### Option A: Using Homebrew (Recommended for macOS)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git LFS
brew install git-lfs
```

### Option B: Direct Download

1. Visit: https://git-lfs.github.com/
2. Download the macOS installer
3. Run the installer
4. Verify installation: `git lfs version`

## Step 3: Initialize Git Repository

```bash
cd /Users/nick/Downloads/Cursor

# Initialize Git (if not already done)
git init

# Initialize Git LFS
git lfs install
```

## Step 4: Configure Git LFS Tracking

The `.gitattributes` file is already created. Verify it exists:

```bash
cat .gitattributes
```

It should show:
```
*.json filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
wine_embeddings.json filter=lfs diff=lfs merge=lfs -text
df.csv filter=lfs diff=lfs merge=lfs -text
```

## Step 5: Add Files to Git

```bash
# Add .gitattributes first (important!)
git add .gitattributes

# Add other files
git add "index (1).html"
git add rescsys.py
git add README.md
git add setup_git_lfs.sh

# Add large files (Git LFS will handle these)
git add wine_embeddings.json
git add df.csv
```

## Step 6: Verify Git LFS is Tracking Large Files

```bash
# Check which files are tracked by LFS
git lfs ls-files

# You should see:
# wine_embeddings.json
# df.csv
```

## Step 7: Commit and Push

```bash
# Create initial commit
git commit -m "Initial commit: Wine recommender with Git LFS"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git push -u origin main
# or if your default branch is 'master':
git push -u origin master
```

## Step 8: Verify on GitHub

1. Go to your GitHub repository
2. Check that `wine_embeddings.json` and `df.csv` show as "Stored with Git LFS"
3. The file size should show correctly

## Troubleshooting

### Git LFS files not showing as LFS on GitHub

If files appear as regular files instead of LFS:

```bash
# Remove files from Git cache
git rm --cached wine_embeddings.json df.csv

# Re-add them (they should be tracked by LFS now)
git add wine_embeddings.json df.csv

# Commit the change
git commit -m "Fix: Track large files with Git LFS"

# Force push (if needed)
git push --force
```

### Check Git LFS status

```bash
# See what Git LFS is tracking
git lfs track

# See which files are stored in LFS
git lfs ls-files
```

### GitHub LFS Quota

- Free accounts: 1 GB storage, 1 GB bandwidth/month
- Your files: ~979 MB total (within free tier)
- If you exceed limits, consider:
  - Using a smaller sample of embeddings
  - Compressing the JSON file
  - Using GitHub's package registry

## Quick Setup Script

Alternatively, you can run the automated setup script:

```bash
./setup_git_lfs.sh
```

This will handle steps 3-4 automatically.

## Important Notes

1. **Always commit `.gitattributes` first** before adding large files
2. **Git LFS must be installed** on any machine that clones the repo
3. **GitHub automatically enables LFS** when it detects `.gitattributes`
4. **Large files count toward your LFS quota** (1 GB free on GitHub)

## File Sizes

- `wine_embeddings.json`: ~934 MB (must use LFS)
- `df.csv`: ~45 MB (should use LFS)
- Other files: Small, regular Git tracking is fine

