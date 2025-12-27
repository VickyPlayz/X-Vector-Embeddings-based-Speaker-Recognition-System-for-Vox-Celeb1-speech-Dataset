import torch
import torchaudio

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=24, win_length=0.025, hop_length=0.010):
        self.sample_rate = sample_rate
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": int(sample_rate * win_length),
                "win_length": int(sample_rate * win_length),
                "hop_length": int(sample_rate * hop_length),
            }
        )

    def compute_features(self, waveform):
        """
        Compute MFCC features and apply Cepstral Mean Normalization (CMN).
        waveform: [1, T]
        Returns: [D, T_frames]
        """
        # Ensure waveform is on the same device as the transform (if moved to GPU)
        # For now, we assume CPU or handle manually.
        
        mfcc = self.mfcc_transform(waveform)
        
        # CMN: Subtract mean over time dimension (dim=2)
        # mfcc shape: [channel, n_mfcc, time] -> usually [1, 24, T]
        mean = mfcc.mean(dim=-1, keepdim=True)
        mfcc_cmn = mfcc - mean
        
        # Remove channel dim if 1
        return mfcc_cmn.squeeze(0)

def apply_vad(waveform, sample_rate):
    # Simple energy-based VAD or use torchaudio's vad
    # For now, returning waveform as-is or implementing simple threshold
    return waveform
