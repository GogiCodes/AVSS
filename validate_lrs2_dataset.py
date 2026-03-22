#!/usr/bin/env python3
"""
Script to validate LRS2-2Mix dataset structure and create proper file lists
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def validate_lrs2_2mix_structure(ds_root):
    """
    Validate the LRS2-2Mix dataset structure and report file availability
    """
    print("=" * 80)
    print("LRS2-2Mix Dataset Validation")
    print("=" * 80)
    
    # Check main directories
    required_dirs = [
        'audio/wav16k/min',
        'mouths',
        'faces',
    ]
    
    print("\n1. Checking directory structure...")
    for dir_name in required_dirs:
        full_path = os.path.join(ds_root, dir_name)
        if os.path.exists(full_path):
            print(f"   ✓ {dir_name}")
        else:
            print(f"   ✗ {dir_name} NOT FOUND")
    
    # Check split files
    print("\n2. Checking split files...")
    split_files = ['train.txt', 'test.txt', 'dev.txt']
    for split_file in split_files:
        full_path = os.path.join(ds_root, split_file)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            print(f"   ✓ {split_file}: {len(lines)} samples")
        else:
            print(f"   ✗ {split_file} NOT FOUND")
    
    # Check audio subdirectories
    print("\n3. Checking audio splits (train/val/test)...")
    audio_root = os.path.join(ds_root, 'audio/wav16k/min')
    if os.path.exists(audio_root):
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(audio_root, split, 's2')
            if os.path.exists(split_path):
                audio_files = len([f for f in os.listdir(split_path) if f.endswith('.wav')])
                print(f"   ✓ {split}/s2: {audio_files} audio files")
            else:
                print(f"   ✗ {split}/s2 NOT FOUND")
    
    # Sample analysis
    print("\n4. Sample analysis...")
    with open(os.path.join(ds_root, 'test.txt'), 'r') as f:
        test_samples = [l.strip() for l in f.readlines() if l.strip()][:5]
    
    print(f"   Sample speaker IDs from test.txt:")
    for sample in test_samples:
        mouth_path = os.path.join(ds_root, 'mouths', f'{sample}.npz')
        mouth_exists = "✓" if os.path.exists(mouth_path) else "✗"
        print(f"   {mouth_exists} {sample}")
    
    # Estimate dataset statistics
    print("\n5. Dataset statistics...")
    stats = {
        'train': 0,
        'test': 0,
        'val': 0,
    }
    
    for split, count in stats.items():
        split_file = os.path.join(ds_root, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                count = len([l.strip() for l in f.readlines() if l.strip()])
            stats[split] = count
    
    total = sum(stats.values())
    print(f"   Total samples: {total}")
    print(f"   - Train: {stats['train']}")
    print(f"   - Test: {stats['test']}")
    print(f"   - Val: {stats['val']}")
    
    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


def create_file_list_csv(ds_root, output_file='lrs2_2mix_filelist.csv'):
    """
    Create a CSV file with audio file mappings
    Format: mixture_file,speaker1_id,speaker2_id,mouth1_path,mouth2_path
    """
    print(f"\nGenerating file list: {output_file}")
    
    audio_root = os.path.join(ds_root, 'audio/wav16k/min/test/s2')
    
    if not os.path.exists(audio_root):
        print(f"Error: Audio directory not found: {audio_root}")
        return
    
    lines = []
    audio_files = sorted([f for f in os.listdir(audio_root) if f.endswith('.wav')])
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in audio_files[:10]:  # Sample first 10
        mixture_path = os.path.join(audio_root, audio_file)
        # Parse speaker IDs from filename (format: id1_track1_scale1_id2_track2_scale2.wav)
        name_parts = audio_file.replace('.wav', '').split('_')
        
        lines.append(f"{mixture_path}")
    
    # Write to file
    output_path = os.path.join(ds_root, output_file)
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"✓ Created {output_file} with {len(lines)} entries")


if __name__ == '__main__':
    ds_root = '/Users/sumanth/Desktop/DDP/RAVSS/ravss_code/lrs2_rebuild'
    
    print(f"Dataset root: {ds_root}\n")
    
    # Validate structure
    validate_lrs2_2mix_structure(ds_root)
    
    # Create file list
    create_file_list_csv(ds_root)
