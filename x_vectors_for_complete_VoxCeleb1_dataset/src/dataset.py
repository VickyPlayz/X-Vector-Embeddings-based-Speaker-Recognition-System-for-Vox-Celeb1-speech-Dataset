import os
import torch
import torchaudio
from torch.utils.data import Dataset
import random

class VoxCelebDataset(Dataset):
    def __init__(self, root_dir, min_duration=2.0, transform=None, subset_ratio=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.min_duration = min_duration
        self.subset_ratio = subset_ratio
        self.samples = []
        self.speaker_to_id = {}
        self._load_data()

    def _load_data(self):
        print(f"Scanning dataset at {self.root_dir}...")
        speakers = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        if self.subset_ratio < 1.0:
            original_count = len(speakers)
            subset_count = int(original_count * self.subset_ratio)
            speakers = speakers[:subset_count]
            print(f"Subsetting dataset: using {subset_count}/{original_count} speakers ({self.subset_ratio*100}%)")
        
        self.speaker_to_id = {spk: idx for idx, spk in enumerate(speakers)}
        
        for spk in speakers:
            spk_dir = os.path.join(self.root_dir, spk)
            spk_idx = self.speaker_to_id[spk]
            
            for root, _, files in os.walk(spk_dir):
                for file in files:
                    if file.endswith('.wav'):
                        path = os.path.join(root, file)
                        self.samples.append((path, spk_idx))
        
        print(f"Found {len(self.samples)} samples from {len(speakers)} speakers.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(path)
        
        # Fixed length for batching (3 seconds = 48000 samples at 16k)
        target_length = 48000
        
        # Resample if not 16k is handled implicitly or requires explicit check.
        # Assuming 16k for now.
        
        num_samples = waveform.size(1)
        
        if num_samples < target_length:
            # Pad (repeat)
            num_repeats = (target_length // num_samples) + 1
            waveform = waveform.repeat(1, num_repeats)
            num_samples = waveform.size(1)
            
        # Random crop
        if num_samples > target_length:
            start_idx = random.randint(0, num_samples - target_length)
            waveform = waveform[:, start_idx : start_idx + target_length]
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label, path
