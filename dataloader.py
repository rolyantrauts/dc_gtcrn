import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import soundfile as sf
import numpy as np

class DynamicDataset(Dataset):
    def __init__(self, clean_json, noise_json, snr_range=(0, 20)):
        with open(clean_json) as f: self.clean = json.load(f)
        with open(noise_json) as f: self.noise = json.load(f)
        self.snr_min, self.snr_max = snr_range
        self.len_samples = 16000 * 3  # 3 Seconds

    def set_snr(self, min_db, max_db):
        self.snr_min = min_db
        self.snr_max = max_db

    def load_audio(self, path):
        try:
            info = sf.info(path)
            if info.frames <= self.len_samples:
                # Pad
                data, _ = sf.read(path, dtype='float32')
                if data.ndim > 1: data = data[:, 0]
                pad = self.len_samples - len(data)
                return np.pad(data, (0, pad))
            else:
                # Random Crop
                start = random.randint(0, info.frames - self.len_samples)
                data, _ = sf.read(path, start=start, frames=self.len_samples, dtype='float32')
                if data.ndim > 1: data = data[:, 0]
                return data
        except:
            return np.zeros(self.len_samples, dtype='float32')

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        # 1. Load Clean
        clean = self.load_audio(self.clean[idx])
        
        # 2. Load Noise (Random)
        noise = self.load_audio(random.choice(self.noise))
        
        # 3. Dynamic Mix
        clean_rms = np.sqrt(np.mean(clean**2) + 1e-9)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-9)
        
        snr = random.uniform(self.snr_min, self.snr_max)
        target_rms = clean_rms / (10**(snr/20) + 1e-9)
        
        scaled_noise = noise * (target_rms / (noise_rms + 1e-9))
        noisy = clean + scaled_noise
        
        # 4. Normalize (Critical)
        peak = max(np.max(np.abs(noisy)), np.max(np.abs(clean))) + 1e-9
        noisy = noisy / peak * 0.95
        clean = clean / peak * 0.95
        
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)

def create_loader(batch_size=32, num_workers=4):
    train_ds = DynamicDataset("data_index/train_clean.json", "data_index/noise_list.json")
    val_ds = DynamicDataset("data_index/val_clean.json", "data_index/noise_list.json")
    
    # Mac/MPS compatibility check for pin_memory
    use_pin = not torch.backends.mps.is_available()
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=use_pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers)
    
    return train_dl, val_dl, train_ds