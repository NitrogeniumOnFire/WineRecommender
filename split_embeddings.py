#!/usr/bin/env python3
"""
Split wine_embeddings.json into smaller chunks that can be loaded via GitHub Pages.
Each chunk will be under 50MB to avoid GitHub Pages size limits.
"""

import json
import math
import os

CHUNK_SIZE_MB = 50  # Target chunk size in MB
BYTES_PER_MB = 1024 * 1024
TARGET_CHUNK_SIZE = CHUNK_SIZE_MB * BYTES_PER_MB

def split_embeddings(input_file='wine_embeddings.json', output_dir='embeddings_chunks'):
    """Split embeddings JSON into smaller chunks."""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    embeddings = data['embeddings']
    count = data.get('count', len(embeddings))
    dimension = data.get('dimension', len(embeddings[0]) if embeddings else 0)
    
    print(f"Total embeddings: {count}")
    print(f"Dimension: {dimension}")
    print(f"Total size: ~{os.path.getsize(input_file) / BYTES_PER_MB:.2f} MB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate how many embeddings per chunk
    # Estimate: each embedding is ~dimension * 4 bytes (float32) + JSON overhead
    # JSON overhead is roughly 2x the data size
    estimated_bytes_per_embedding = dimension * 4 * 2  # rough estimate
    embeddings_per_chunk = max(1, int(TARGET_CHUNK_SIZE / estimated_bytes_per_embedding))
    
    print(f"\nSplitting into chunks of ~{embeddings_per_chunk} embeddings each...")
    
    num_chunks = math.ceil(len(embeddings) / embeddings_per_chunk)
    print(f"Will create {num_chunks} chunks")
    
    chunk_files = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * embeddings_per_chunk
        end_idx = min(start_idx + embeddings_per_chunk, len(embeddings))
        
        chunk_embeddings = embeddings[start_idx:end_idx]
        
        chunk_data = {
            'chunk_index': chunk_idx,
            'total_chunks': num_chunks,
            'start_index': start_idx,
            'end_index': end_idx,
            'count': len(chunk_embeddings),
            'dimension': dimension,
            'embeddings': chunk_embeddings
        }
        
        chunk_filename = f'embeddings_chunk_{chunk_idx:04d}.json'
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f)
        
        file_size_mb = os.path.getsize(chunk_path) / BYTES_PER_MB
        chunk_files.append(chunk_filename)
        
        print(f"  Created {chunk_filename}: {len(chunk_embeddings)} embeddings, {file_size_mb:.2f} MB")
    
    # Create manifest file
    manifest = {
        'total_embeddings': count,
        'dimension': dimension,
        'total_chunks': num_chunks,
        'chunk_files': chunk_files,
        'embeddings_per_chunk': embeddings_per_chunk
    }
    
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Created {num_chunks} chunks + manifest.json")
    print(f"📁 Output directory: {output_dir}/")
    print(f"\nNext steps:")
    print(f"1. Add {output_dir}/ to .gitattributes (track with Git LFS if chunks are large)")
    print(f"2. Commit and push the chunks")
    print(f"3. Update index.html to load chunks instead of single file")
    
    return manifest_path

if __name__ == '__main__':
    split_embeddings()

