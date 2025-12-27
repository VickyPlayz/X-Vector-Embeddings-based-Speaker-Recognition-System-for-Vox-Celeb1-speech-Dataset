import os
import random
import itertools

def generate_trials(dataset, num_trials=1000):
    """
    Generate a list of (enroll_path, test_path, label)
    label: 1 for same speaker, 0 for different
    """
    # Group by speaker
    spk_to_paths = {}
    for path, label in dataset.samples:
        if label not in spk_to_paths:
            spk_to_paths[label] = []
        spk_to_paths[label].append(path)
        
    trials = []
    
    # Generate Positive Trials (Same Speaker)
    for spk, paths in spk_to_paths.items():
        if len(paths) < 2: continue
        # Create a few pairs
        pairs = list(itertools.combinations(paths, 2))
        # Limit per speaker to avoid imbalance
        selected = random.sample(pairs, min(len(pairs), 5))
        for p1, p2 in selected:
            trials.append((p1, p2, 1))
            
    num_positive = len(trials)
    
    # Generate Negative Trials (Diff Speaker)
    # Be careful to generate roughly equal amount
    spk_ids = list(spk_to_paths.keys())
    while len(trials) < num_positive * 2:
        s1, s2 = random.sample(spk_ids, 2)
        p1 = random.choice(spk_to_paths[s1])
        p2 = random.choice(spk_to_paths[s2])
        trials.append((p1, p2, 0))
        
    return trials

def write_trial_file(trials, filename="trials.txt"):
    with open(filename, 'w') as f:
        for p1, p2, label in trials:
            f.write(f"{label} {p1} {p2}\n")
