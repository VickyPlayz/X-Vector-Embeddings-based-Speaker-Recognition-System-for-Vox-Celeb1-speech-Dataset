import argparse
import sys
import os
import numpy as np
from src.train import train

def main():
    parser = argparse.ArgumentParser(description="Speaker Recognition X-Vector System (Full Dataset)")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract x-vectors')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train()
    elif args.command == 'extract':
        from src.extract_vectors import load_model, extract_vectors, process_embeddings
        from src.dataset import VoxCelebDataset
        from src.train import DATA_ROOT
        
        # Load dataset
        dataset = VoxCelebDataset(root_dir=DATA_ROOT) # Full dataset
        num_spks = len(dataset.speaker_to_id)
        
        # Load model using the last checkpoint
        checkpoint = "checkpoints/xvector_epoch_20.pth"
        if not os.path.exists(checkpoint):
            print("Checkpoint not found!")
            return

        model = load_model(checkpoint, num_spks)
        emb_dict, labels = extract_vectors(model, dataset)
        norm_embs = process_embeddings(emb_dict)
        
        # Save embeddings logic could be here, but let's pass to eval directly
        
        # Generate Trials
        from src.trials import generate_trials
        trials = generate_trials(dataset)
        print(f"Generated {len(trials)} verification trials.")
        
        # Score
        from src.plda import PLDABackend
        # Using Cosine Score (simplest backend) for now
        scores = []
        gt_labels = []
        for p1, p2, label in trials:
            if p1 in norm_embs and p2 in norm_embs:
                vec1 = norm_embs[p1]
                vec2 = norm_embs[p2]
                score = np.dot(vec1, vec2) # Cosine trial
                scores.append(score)
                gt_labels.append(label)
                
        # Compute EER
        from src.evaluate import compute_eer
        eer, thresh = compute_eer(scores, gt_labels)
        print(f"Validation EER: {eer*100:.2f}% at threshold {thresh:.4f}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
