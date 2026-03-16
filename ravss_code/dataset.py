import math
from typing import Dict, List, Union
import json
import os
import subprocess

import numpy as np
import torch
import torch.utils.data as tdata
import soundfile
import pdb
import random

class Vox2_Dataset(tdata.Dataset):
    def __init__(self, mix_num, scp_file, ds_root, dstype, visual_embed_type: str = 'resnet', batch_size: int = 4, max_duration: int = 6, sr: int = 16000):
        self._dataframe=[]
        self.batch_size = batch_size
        self.mix_num = mix_num

        id2dur = {}
        with open(scp_file, 'r') as f:
            for line in f.readlines():
                mix_name = line.rstrip('\n').replace(',','_').replace('/','_')+'.wav'
                line=line.strip().split(',')

                ###########################################################################################################
                uid=line[1]
                for mix_id in range(mix_num):
                    uid += '#'+line[4*mix_id+2]+'/'+line[4*mix_id+3]
                id2dur[uid]=int(float(line[-1])*16000)
                mixture_path= ds_root + 'audio_mixture_5mix/'+dstype+'/'+mix_name

                s_path = []
                c_path = []
                for mix_id in range(mix_num):
                    s_path.append(ds_root + 'audio_clean/'+dstype+'/'+line[4*mix_id+2]+'/'+line[4*mix_id+3]+'.wav')
                    c_path.append(ds_root + 'lip/'+dstype+'/'+line[4*mix_id+2]+'/'+line[4*mix_id+3]+'.npy')

                self._dataframe.append({'uid': uid, 'mixture': mixture_path, 's': s_path, 'c': c_path, 'dur': id2dur[uid]})
                ############################################################################################################

        self.visual_embed_type = visual_embed_type
        self._dataframe = sorted(self._dataframe, key=lambda d: d['dur'], reverse=True)
        self._minibatch = []
        start = 0
        while True:
            end = min(len(self._dataframe), start + self.batch_size)
            self._minibatch.append(self._dataframe[start: end])
            if end == len(self._dataframe):
                break
            start = end
        

        self.len = len(self._minibatch)
        self._sr=sr
        self.max_duration = max_duration
        self.max_duration_in_samples = int(max_duration * sr)
        self.max_duration_in_frames = int(max_duration * 25)
        
        # self.spkmap = {}
        # index = 0
        # for item in self._dataframe:
        #     dstype,spkid1, spkid2 = item['uid'].split('#')
        #     spkid1 = spkid1.split('/')[0]
        #     spkid2 = spkid2.split('/')[0]
        #     if spkid1 not in self.spkmap:
        #         self.spkmap[spkid1] = index
        #         index += 1
        #     if spkid2 not in self.spkmap:
        #         self.spkmap[spkid2] = index
        #         index += 1
           
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        batch_list = self._minibatch[index]
        # min_length = batch_list[-1]['dur']
        # min_length_in_second = min_length / 16000.0
        # min_length_in_frame = math.floor(min_length_in_second * 25)
        mixtures = []
        sources = []
        conditions = []
        spkids = None
        for meta_info in batch_list:
            mixture, _ = soundfile.read(meta_info['mixture'], dtype='float32')
            # soundfile.write('mixture.wav',mixture,16000)
            min_length = mixture.shape[0]
            s_id = {}
            c_id = {}
            ####################################################################
            for mix_id in range(self.mix_num):
                s,_ = soundfile.read(meta_info['s'][mix_id],dtype='float32')
                # soundfile.write(f'clean_{mix_id}.wav',s,16000)
                s_id[mix_id] = s
                c = np.load(meta_info['c'][mix_id]) #(149,1024,1)
                c_id[mix_id] = c
                min_length = min(min_length,s.shape[0])
            ####################################################################
            min_length_in_second = min_length / 16000.0
            min_length_in_frame = math.floor(min_length_in_second * 25)

            mixture = mixture[:min_length]    
            mixture = np.divide(mixture, np.max(np.abs(mixture)))

            for mix_id in range(self.mix_num):
                s_id[mix_id] = s_id[mix_id][:min_length]
                s_id[mix_id] = np.divide(s_id[mix_id], np.max(np.abs(s_id[mix_id])))
                c_id[mix_id] = c_id[mix_id][:min_length_in_frame]

                if self.visual_embed_type == 'resnet':
                    if c_id[mix_id].shape[0] < min_length_in_frame:
                        c_id[mix_id] = np.pad(c_id[mix_id], ((0, min_length_in_frame - c_id[mix_id].shape[0]), (0, 0)), mode = 'edge')
                else:
                    if c_id[mix_id].shape[0] < min_length_in_frame:
                        c_id[mix_id] = np.pad(c_id[mix_id], ((0, min_length_in_frame - c_id[mix_id].shape[0]), (0, 0), (0, 0)), mode = 'edge')
            
                sources.append(s_id[mix_id][:self.max_duration_in_samples])
                conditions.append(c_id[mix_id][:self.max_duration_in_frames])
                mixtures.append(mixture[:self.max_duration_in_samples])

        mixtures = torch.tensor(np.array(mixtures))
        sources = torch.tensor(np.array(sources))
        conditions = torch.tensor(np.array(conditions))

        return mixtures, sources, conditions, spkids

def dummy_collate_fn(x):
    if len(x) == 1:
        return x[0]
    else:
        return x

class New_Dataset(tdata.Dataset):
    def __init__(self, mix_num, ds_root, dstype, visual_embed_type: str = 'resnet', batch_size: int = 4, max_duration: int = 6, sr: int = 16000):
        self._dataframe = []
        self.batch_size = batch_size
        self.mix_num = mix_num
        self.ds_root = ds_root
        self.dstype = dstype

        # Scan all sessions
        sessions = [d for d in os.listdir(ds_root) if os.path.isdir(os.path.join(ds_root, d)) and d.startswith('session')]

        for session in sessions:
            session_path = os.path.join(ds_root, session)
            metadata_path = os.path.join(session_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            speakers_dir = os.path.join(session_path, 'speakers')
            if not os.path.exists(speakers_dir):
                continue

            speakers = [d for d in os.listdir(speakers_dir) if os.path.isdir(os.path.join(speakers_dir, d)) and d.startswith('spk_')]

            # For simplicity, create mixtures of mix_num speakers
            if len(speakers) < mix_num:
                continue

            # Randomly select mix_num speakers
            selected_speakers = random.sample(speakers, mix_num)

            uid = f"{session}_{'_'.join(selected_speakers)}"

            mixture_path = None  # We'll generate on the fly
            s_paths = []
            c_paths = []

            for spk in selected_speakers:
                spk_path = os.path.join(speakers_dir, spk)
                # Use central_crops for now
                crops_dir = os.path.join(spk_path, 'central_crops')
                if os.path.exists(crops_dir):
                    track_files = [f for f in os.listdir(crops_dir) if f.startswith('track_00') and f.endswith('.mp4')]
                    if track_files:
                        video_path = os.path.join(crops_dir, track_files[0])  # Assume one track
                        audio_path = video_path.replace('.mp4', '.wav')
                        # Extract audio if not exists
                        if not os.path.exists(audio_path):
                            self.extract_audio(video_path, audio_path)
                        s_paths.append(audio_path)

                        # For lip features, use the .json file
                        json_path = os.path.join(crops_dir, 'track_00.json')
                        c_paths.append(json_path)

            if len(s_paths) == mix_num and len(c_paths) == mix_num:
                # Calculate duration from one audio
                if os.path.exists(s_paths[0]):
                    audio, _ = soundfile.read(s_paths[0])
                    duration = len(audio) / sr
                    self._dataframe.append({
                        'uid': uid,
                        'mixture': None,  # Generate on fly
                        's': s_paths,
                        'c': c_paths,
                        'dur': len(audio)
                    })

        self.visual_embed_type = visual_embed_type
        self._dataframe = sorted(self._dataframe, key=lambda d: d['dur'], reverse=True)
        self._minibatch = []
        start = 0
        while True:
            end = min(len(self._dataframe), start + self.batch_size)
            self._minibatch.append(self._dataframe[start: end])
            if end == len(self._dataframe):
                break
            start = end

        self.len = len(self._minibatch)
        self._sr = sr
        self.max_duration = max_duration
        self.max_duration_in_samples = int(max_duration * sr)
        self.max_duration_in_frames = int(max_duration * 25)

    def extract_audio(self, video_path, audio_path):
        """Extract audio from video using ffmpeg"""
        command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(command, shell=True, check=True)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        batch_list = self._minibatch[index]
        mixtures = []
        sources = []
        conditions = []
        spkids = None

        for meta_info in batch_list:
            # Load sources
            s_list = []
            c_list = []
            min_length = float('inf')

            for s_path in meta_info['s']:
                s, _ = soundfile.read(s_path, dtype='float32')
                s_list.append(s)
                min_length = min(min_length, len(s))

            # Truncate all to min_length
            for i in range(len(s_list)):
                s_list[i] = s_list[i][:int(min_length)]

            # Create mixture
            mixture = np.sum(s_list, axis=0)
            mixture = np.divide(mixture, np.max(np.abs(mixture)))

            # Normalize sources
            for i in range(len(s_list)):
                s_list[i] = np.divide(s_list[i], np.max(np.abs(s_list[i])))

            # Load conditions (lip features from json)
            for c_path in meta_info['c']:
                with open(c_path, 'r') as f:
                    c_data = json.load(f)
                # TODO: Adapt this based on the actual structure of track_00.json
                # Assuming it has 'features' key with numpy array or list
                if 'features' in c_data:
                    c = np.array(c_data['features'])
                elif isinstance(c_data, list):
                    c = np.array(c_data)
                elif isinstance(c_data, dict) and 'landmarks' in c_data:
                    # If landmarks, perhaps flatten or process
                    c = np.array(c_data['landmarks']).flatten()
                else:
                    # Fallback to random
                    num_frames = int(min_length / self._sr * 25)
                    c = np.random.randn(num_frames, 512)  # Placeholder
                c_list.append(c)

            # Truncate to max duration
            mixture = mixture[:self.max_duration_in_samples]
            for i in range(len(s_list)):
                s_list[i] = s_list[i][:self.max_duration_in_samples]
                c_list[i] = c_list[i][:self.max_duration_in_frames]

            mixtures.append(mixture)
            sources.append(np.array(s_list))
            conditions.append(np.array(c_list))

        mixtures = torch.tensor(np.array(mixtures))
        sources = torch.tensor(np.array(sources))
        conditions = torch.tensor(np.array(conditions))

        return mixtures, sources, conditions, spkids

if __name__ == '__main__':
    model = Vox2_Dataset()
