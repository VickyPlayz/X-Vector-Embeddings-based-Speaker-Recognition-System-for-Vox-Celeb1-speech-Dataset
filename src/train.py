import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml

from src.dataset import VoxCelebDataset
from src.model import XVectorNet
from src.features import FeatureExtractor

# Configuration usually loaded from yaml, but hardcoding for simplicity now
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = r'G:\3 x vector with 10 percent dataset\Dataset\VoxCeleb\vox1_dev_wav'

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset & Loader
    # Create a simple collate function to handle variable lengths if needed
    # For TDNN, we usually train on fixed chunks (e.g., 2-4 seconds)
    # The loader returns raw waveform, we need to process it.
    
    dataset = VoxCelebDataset(root_dir=DATA_ROOT, subset_ratio=0.1) # Use 10% as requested
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0) # Windows issues with num_workers?
    
    # 2. Model
    # We need to know num_speakers from dataset first
    # Or strict mapping if 1251 speakers.
    num_speakers = len(dataset.speaker_to_id)
    print(f"Number of speakers: {num_speakers}")
    
    model = XVectorNet(num_speakers=num_speakers).to(DEVICE)
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Weight decay important for regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    
    # 4. Feature Extractor
    # Initialize once
    extractor = FeatureExtractor().mfcc_transform.to(DEVICE)
    # Note: features.py has compute_features but that was CPU/numpy centric maybe?
    # Let's use the torchaudio transform directly on GPU
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for waveforms, labels, _ in pbar:
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Feature Extraction on GPU
            # Expecting waveforms to be [Batch, 1, Time]
            # MFCC Transform expects [Batch, Time] or [..., Time]
            # Since dataset returns [1, Time], stack makes it [Batch, 1, Time]
            # Squeeze channel for MFCC
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)
                
            features = extractor(waveforms) # [Batch, n_mfcc, Time]
            # CMN
            features = features - features.mean(dim=2, keepdim=True)
            
            # Forward
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': running_loss/total, 'Acc': 100.*correct/total})
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {100.*correct/total:.2f}%")
        scheduler.step(epoch_loss)
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoints/xvector_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    # Check if we can access the dataset path
    if not os.path.exists(DATA_ROOT):
        print(f"WARNING: Dataset path {DATA_ROOT} does not exist or is not accessible.")
        print("Please check the path or drive connection.")
    else:
        train()
