import math
from typing import Dict, List, Union
import numpy as np
import torch
import torch.utils.data as tdata
import soundfile
import os
from pathlib import Path
import re

class LRS2_2Mix_Dataset(tdata.Dataset):
    """
    LRS2-2Mix Dataset Loader
    Structure:
    - Mixed audio: audio/wav16k/min/{tr|cv|tt}/mix/*.wav
    - Clean sources: audio/wav16k/min/{tr|cv|tt}/s1/*.wav, s2/*.wav
    - Visual features: mouths/*.npz, faces/*.npz
    - Split files: train.txt, test.txt, dev.txt
    
    Filename format: speaker1_id_track1_scale1_speaker2_id_track2_scale2.wav
    """
    
    def __init__(self, 
                 mix_num=2, 
                 scp_file=None, 
                 ds_root=None, 
                 dstype='train',
                 batch_size=4, 
                 max_duration=6, 
                 sr=16000):
        
        self._dataframe = []
        self.batch_size = batch_size
        self.mix_num = mix_num
        self._sr = sr
        self.max_duration_in_samples = int(max_duration * sr)
        self.ds_root = ds_root
        self.dstype = dstype
        
        # Map dstype to folder names (train->tr, test->tt, val->cv)
        type_map = {'train': 'tr', 'test': 'tt', 'val': 'cv', 'dev': 'cv'}
        folder_type = type_map.get(dstype, 'tr')
        
        # If scp_file not provided, use default path based on dstype
        if scp_file is None:
            txt_name_map = {'train': 'train.txt', 'test': 'test.txt', 'val': 'dev.txt', 'dev': 'dev.txt'}
            scp_file = os.path.join(ds_root, txt_name_map.get(dstype, 'train.txt'))
        
        # Load the data list
        print(f"Loading LRS2-2Mix {dstype} set from {scp_file}")
        
        if not os.path.exists(scp_file):
            print(f"Warning: {scp_file} not found")
            self._dataframe = []
        else:
            with open(scp_file, 'r') as f:
                speaker_tracks = [l.strip() for l in f.readlines() if l.strip()]
            
            # Get mixture files
            mix_dir = os.path.join(ds_root, f'audio/wav16k/min/{folder_type}/mix')
            s1_dir = os.path.join(ds_root, f'audio/wav16k/min/{folder_type}/s1')
            s2_dir = os.path.join(ds_root, f'audio/wav16k/min/{folder_type}/s2')
            
            if os.path.exists(mix_dir):
                for mix_file in os.listdir(mix_dir):
                    if not mix_file.endswith('.wav'):
                        continue
                    
                    # Check if corresponding s1 and s2 files exist
                    s1_file = os.path.join(s1_dir, mix_file)
                    s2_file = os.path.join(s2_dir, mix_file)
                    
                    if not (os.path.exists(s1_file) and os.path.exists(s2_file)):
                        continue
                    
                    # Parse mixture filename to extract speaker IDs:
                    # Format: speaker1_id_track1_scale1_speaker2_id_track2_scale2.wav
                    match = re.match(r'(\d+)_(\d+)_([-\d.]+)_(\d+)_(\d+)_([-\d.]+)\.wav', mix_file)
                    if not match:
                        continue
                    
                    spk1_id, spk1_track, scale1, spk2_id, spk2_track, scale2 = match.groups()
                    spk1_key = f"{spk1_id}_{spk1_track}"
                    spk2_key = f"{spk2_id}_{spk2_track}"
                    
                    # Get visual features
                    mouth1_file = os.path.join(ds_root, 'mouths', f'{spk1_key}.npz')
                    mouth2_file = os.path.join(ds_root, 'mouths', f'{spk2_key}.npz')
                    
                    self._dataframe.append({
                        'mixture': os.path.join(mix_dir, mix_file),
                        'sources': [s1_file, s2_file],
                        'mouths': [mouth1_file if os.path.exists(mouth1_file) else None,
                                   mouth2_file if os.path.exists(mouth2_file) else None],
                        'spk_ids': [spk1_key, spk2_key],
                    })
        
        print(f"Found {len(self._dataframe)} samples in {dstype} set")
        
        # Create minibatches
        self._minibatch = []
        start = 0
        while True:
            end = min(len(self._dataframe), start + self.batch_size)
            self._minibatch.append(self._dataframe[start:end])
            if end == len(self._dataframe):
                break
            start = end
        
        self.len = len(self._minibatch)
        
        if self.len == 0:
            print(f"Warning: No batches created! Check dataset paths.")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        batch_list = self._minibatch[index]
        mixtures = []
        sources = []
        conditions = []
        
        for meta_info in batch_list:
            # Load mixture audio
            try:
                mixture, _ = soundfile.read(meta_info['mixture'], dtype='float32')
            except Exception as e:
                print(f"Error reading mixture {meta_info['mixture']}: {e}")
                continue
            
            # Load clean source audios
            clean_sources = []
            min_length = mixture.shape[0]
            
            for source_path in meta_info['sources']:
                try:
                    source, _ = soundfile.read(source_path, dtype='float32')
                    clean_sources.append(source)
                    min_length = min(min_length, source.shape[0])
                except Exception as e:
                    print(f"Error reading source {source_path}: {e}")
                    clean_sources.append(None)
            
            # Check if all sources were loaded successfully
            if any(s is None for s in clean_sources):
                continue
            
            # Trim all to same length
            mixture = mixture[:min_length]
            clean_sources = [s[:min_length] for s in clean_sources]
            
            # Normalize
            if np.max(np.abs(mixture)) > 0:
                mixture = np.divide(mixture, np.max(np.abs(mixture)) + 1e-8)
            
            for i in range(len(clean_sources)):
                if np.max(np.abs(clean_sources[i])) > 0:
                    clean_sources[i] = np.divide(clean_sources[i], np.max(np.abs(clean_sources[i])) + 1e-8)
            
            # Concatenate clean sources (as done in original dataset format)
            sources_concat = np.concatenate(clean_sources)
            
            # Load visual features (mouths)
            visual_features = []
            for mouth_path in meta_info['mouths']:
                if mouth_path and os.path.exists(mouth_path):
                    try:
                        mouth_data = np.load(mouth_path)
                        mouth_array = mouth_data[list(mouth_data.keys())[0]]
                        visual_features.append(mouth_array)
                    except Exception as e:
                        print(f"Warning: Could not load mouth {mouth_path}: {e}")
                        visual_features.append(None)
                else:
                    visual_features.append(None)
            
            # If no visual features, create dummy ones
            for i in range(len(visual_features)):
                if visual_features[i] is None:
                    # Create dummy visual feature - shape (frames, features)
                    num_frames = int(min_length / (self._sr / 25))
                    visual_features[i] = np.zeros((num_frames, 512), dtype='float32')
            
            # Concatenate visual features from both speakers
            visual_concat = np.concatenate(visual_features, axis=1) if len(visual_features) > 1 else visual_features[0]
            
            # Trim to max duration
            mixture = mixture[:self.max_duration_in_samples]
            sources_concat = sources_concat[:self.max_duration_in_samples * 2]  # 2 speakers worth
            
            # Trim visual features to match
            max_frames = int(self.max_duration_in_samples / (self._sr / 25))
            visual_concat = visual_concat[:max_frames]
            
            mixtures.append(mixture)
            sources.append(sources_concat)
            conditions.append(visual_concat)
        
        # Convert to tensors
        mixtures = torch.tensor(np.array(mixtures), dtype=torch.float32)
        sources = torch.tensor(np.array(sources), dtype=torch.float32)
        
        # Stack visual features with padding if necessary
        try:
            conditions = torch.tensor(np.array(conditions), dtype=torch.float32)
        except:
            # If shapes don't match, pad them
            max_frames = max([c.shape[0] for c in conditions])
            max_features = max([c.shape[1] if len(c.shape) > 1 else 1 for c in conditions])
            
            padded_conditions = []
            for c in conditions:
                if len(c.shape) == 1:
                    c = c[:, np.newaxis]
                padded = np.zeros((max_frames, max_features), dtype='float32')
                padded[:c.shape[0], :c.shape[1]] = c
                padded_conditions.append(padded)
            
            conditions = torch.tensor(np.array(padded_conditions), dtype=torch.float32)
        
        return mixtures, sources, conditions, None


def dummy_collate_fn(x):
    """Collate function for batch processing"""
    if len(x) == 1:
        return x[0]
    else:
        return x


if __name__ == '__main__':
    # Test the dataset
    dataset = LRS2_2Mix_Dataset(
        mix_num=2, 
        ds_root='/Users/sumanth/Desktop/DDP/RAVSS/ravss_code/lrs2_rebuild',
        dstype='train',
        batch_size=2,
        max_duration=6
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shapes: mixtures {sample[0].shape}, sources {sample[1].shape}, conditions {sample[2].shape}")
