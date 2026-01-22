import os
import json
import glob
import random
import soundfile as sf
from tqdm import tqdm

# CONFIG
# A "Chunk" is 3 seconds. A 30s file counts as 10 chunks.
CHUNK_LEN = 3.0 
EXTENSIONS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus']

def scan_folder(base_paths, desc):
    if isinstance(base_paths, str): base_paths = [base_paths]
    files = []
    print(f"\n>>> Scanning {desc}...")
    for base in base_paths:
        for ext in EXTENSIONS:
            found = glob.glob(f"{base}/**/*{ext}", recursive=True)
            files.extend(found)
    return files

def balance_files(file_list):
    """
    Opens every file to check duration.
    Returns a weighted list where longer files appear multiple times.
    """
    weighted_list = []
    print(f"    Balancing {len(file_list)} files by duration (this may take a minute)...")
    
    for fpath in tqdm(file_list):
        try:
            info = sf.info(fpath)
            duration = info.duration
            # Calculate weight: e.g., 10s / 3s = 3.3 -> weight 3
            # Ensure every file appears at least once
            weight = max(1, int(duration // CHUNK_LEN))
            
            # Add file to list N times
            weighted_list.extend([fpath] * weight)
        except:
            continue
            
    return weighted_list

def prep():
    # 1. Clean Speech (LibriSpeech)
    clean_raw = scan_folder(["./LibriSpeech"], "Clean Speech")
    clean_balanced = balance_files(clean_raw)
    
    # 2. Noise (MUSAN + Custom ./noise)
    # CRITICAL: We combine them into one massive noise pool
    noise_raw = scan_folder(["./musan", "./noise"], "Noise & Texture")
    noise_balanced = balance_files(noise_raw)
    
    # 3. Validation Split
    # We split by unique files to prevent leakage
    unique_clean = list(set(clean_balanced))
    unique_clean.sort()
    random.seed(42)
    random.shuffle(unique_clean)
    
    # 5% Validation
    val_size = int(len(unique_clean) * 0.05)
    val_files = set(unique_clean[:val_size])
    
    # Reconstruct weighted lists based on the split
    train_clean = [f for f in clean_balanced if f not in val_files]
    val_clean   = [f for f in clean_balanced if f in val_files]
    
    # Save
    os.makedirs("data_index", exist_ok=True)
    with open("data_index/train_clean.json", "w") as f: json.dump(train_clean, f, indent=2)
    with open("data_index/val_clean.json", "w") as f: json.dump(val_clean, f, indent=2)
    with open("data_index/noise_list.json", "w") as f: json.dump(noise_balanced, f, indent=2)
    
    print("\n>>> Dataset Ready.")
    print(f"    Train Samples (Weighted): {len(train_clean)}")
    print(f"    Val Samples (Weighted):   {len(val_clean)}")
    print(f"    Noise Samples (Weighted): {len(noise_balanced)}")

if __name__ == "__main__":
    prep()