import torch
import numpy as np
from src.model import XVectorNet
from src.features import FeatureExtractor
from src.dataset import VoxCelebDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'G:\\3 x vector with 10 percent dataset\\Dataset'

def load_model(checkpoint_path, num_speakers):
    model = XVectorNet(num_speakers=num_speakers).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def extract_vectors(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    extractor = FeatureExtractor().mfcc_transform.to(DEVICE)
    
    embeddings = {} # {path: embedding}
    labels = {}     # {path: speaker_id}
    
    print("Extracting vectors...")
    with torch.no_grad():
        for waveform, label, path in tqdm(dataloader):
            waveform = waveform.to(DEVICE)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            
            features = extractor(waveform)
            features = features - features.mean(dim=2, keepdim=True)
            
            # Get embedding
            embedding = model(features, return_embedding=True) # [1, 512]
            
            embeddings[path[0]] = embedding.cpu().numpy().flatten()
            labels[path[0]] = label.item()
            
    return embeddings, labels

def process_embeddings(embeddings_dict):
    """
    Apply Global Mean Subtraction and Length Normalization
    """
    all_vecs = np.array(list(embeddings_dict.values()))
    global_mean = np.mean(all_vecs, axis=0)
    
    normalized_embeddings = {}
    for path, vec in embeddings_dict.items():
        vec = vec - global_mean
        vec = vec / np.linalg.norm(vec)
        normalized_embeddings[path] = vec
        
    return normalized_embeddings

if __name__ == "__main__":
    # Example usage
    # Need to train first to get a checkpoint
    pass
