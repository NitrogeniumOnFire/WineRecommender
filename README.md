# Wine Recommender System

A semantic wine recommendation system using LLM embeddings and vector similarity search.

## Features

- **Semantic Search**: Find wines based on natural language descriptions
- **Vector Similarity**: Uses sentence transformers for embedding-based retrieval
- **Filtering**: Filter by variety, country, and price range
- **Web Interface**: Beautiful, responsive UI built with Tailwind CSS

## Setup

### Prerequisites

1. **Install Git LFS** (required for large files):
   ```bash
   # macOS (using Homebrew)
   brew install git-lfs
   
   # Or download from: https://git-lfs.github.com/
   ```

2. **Run the setup script**:
   ```bash
   ./setup_git_lfs.sh
   ```

   This will:
   - Initialize Git LFS
   - Initialize Git repository (if not already done)
   - Configure Git LFS to track large files
   - Add `.gitattributes` to the repository

### Manual Setup

If you prefer to set up manually:

```bash
# Install Git LFS (if not already installed)
brew install git-lfs  # macOS
# or download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Initialize Git repository (if not already done)
git init

# Track large files
git lfs track "*.json"
git lfs track "*.csv"
git lfs track "wine_embeddings.json"
git lfs track "df.csv"

# Add .gitattributes
git add .gitattributes
```

## Files

- `index (1).html` - Main web interface
- `rescsys.py` - Python implementation (reference)
- `df.csv` - Wine dataset (tracked with Git LFS)
- `wine_embeddings.json` - Pre-computed embeddings (tracked with Git LFS)
- `.gitattributes` - Git LFS configuration

## Usage

1. Open `index (1).html` in a web browser
2. The page will automatically load:
   - Wine data from `df.csv`
   - Embeddings from `wine_embeddings.json`
   - Sentence transformer model (loaded from CDN)

3. Search for wines using natural language:
   - "rich red wine with notes of cherry and chocolate"
   - "crisp white wine"
   - "full-bodied Cabernet"

4. Use filters to narrow down results:
   - Select wine varieties
   - Select countries
   - Adjust price range

## Technical Details

- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (same as Python version)
- **Similarity Metric**: Cosine similarity
- **Framework**: Vanilla JavaScript with ES modules
- **UI**: Tailwind CSS

## GitHub Pages Deployment

1. Push to GitHub (Git LFS will handle large files automatically)
2. Enable GitHub Pages in repository settings
3. Select the branch containing `index (1).html`
4. The site will be available at: `https://yourusername.github.io/repo-name/`

**Important**: Make sure your GitHub repository has Git LFS enabled. GitHub automatically enables it for repositories with `.gitattributes` files.

## Notes

- First search may take a few seconds while the model downloads (~50MB)
- The embeddings file is ~934MB and must be tracked with Git LFS
- The CSV file is ~45MB and is also tracked with Git LFS

