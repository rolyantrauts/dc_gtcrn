import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
import glob
import random
from tqdm import tqdm

def generate_subset(speech_files, noise_files, output_dir, num_samples, min_snr, max_snr, command_files=None):
    """
    Generates a dataset with strict 4-second (64000 sample) length enforcement.
    """
    os.makedirs(os.path.join(output_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
    
    print(f"Generating {num_samples} samples in {output_dir} (SNR: {min_snr}dB to {max_snr}dB)...")

    # TARGET: Exact 4 seconds at 16kHz
    TARGET_LEN = 64000 

    for i in tqdm(range(num_samples)):
        try:
            # --- 1. Simulation Setup ---
            # Random room (3-6m) with random reverb (RT60: 0.2-0.5s)
            room_dim = np.random.uniform(3, 6, size=2)
            room_dim = np.append(room_dim, np.random.uniform(2.4, 3.0))
            rt60 = np.random.uniform(0.2, 0.5)
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            
            room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
            
            # Mic Array (Stereo, 45mm spacing)
            cx = np.random.uniform(1.0, room_dim[0]-1.0)
            cy = np.random.uniform(1.0, room_dim[1]-1.0)
            h = np.random.uniform(1.0, 1.5)
            mic_locs = np.c_[[cx-0.0225, cy, h], [cx+0.0225, cy, h]]
            room.add_microphone_array(mic_locs)
            
            # Helper: Get valid random source position
            def get_pos():
                for _ in range(20):
                    ang, dist = np.random.uniform(0, 2*np.pi), np.random.uniform(0.5, 3.5)
                    sx, sy = cx + dist*np.cos(ang), cy + dist*np.sin(ang)
                    if 0.5 < sx < room_dim[0]-0.5 and 0.5 < sy < room_dim[1]-0.5:
                        return [sx, sy, np.random.uniform(1.0, 1.8)]
                return [cx+0.5, cy+0.5, 1.5]

            # --- 2. Add Speech ---
            if command_files and np.random.random() < 0.2:
                s_path = random.choice(command_files)
            else:
                s_path = random.choice(speech_files)
            
            speech, _ = sf.read(s_path)
            
            # Loop/Crop to ensure we have enough raw audio before simulation
            while len(speech) < TARGET_LEN: speech = np.concatenate([speech, speech])
            if len(speech) > TARGET_LEN:
                start = np.random.randint(0, len(speech) - TARGET_LEN)
                speech = speech[start : start + TARGET_LEN]
            else:
                speech = speech[:TARGET_LEN]
                
            room.add_source(get_pos(), signal=speech)

            # --- 3. Add Noise ---
            if np.random.random() < 0.95:
                n_path = random.choice(noise_files)
                noise, _ = sf.read(n_path)
                while len(noise) < TARGET_LEN: noise = np.concatenate([noise, noise])
                if len(noise) > TARGET_LEN:
                    start = np.random.randint(0, len(noise) - TARGET_LEN)
                    noise = noise[start : start + TARGET_LEN]
                else:
                    noise = noise[:TARGET_LEN]
                
                # SNR Scaling
                snr = np.random.uniform(min_snr, max_snr)
                s_pwr = np.mean(speech**2) + 1e-9
                n_pwr = np.mean(noise**2) + 1e-9
                scale = np.sqrt(s_pwr / (n_pwr * (10**(snr/10))))
                
                room.add_source(get_pos(), signal=noise * scale)

            # --- 4. Simulate ---
            room.simulate()
            mixed = room.mic_array.signals # [Channels, Time + Tail]
            
            # Clean Reference (Direct path/Early reflections only)
            room_clean = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
            room_clean.add_microphone_array(mic_locs)
            room_clean.add_source(get_pos(), signal=speech) 
            room_clean.simulate()
            clean = room_clean.mic_array.signals[0, :] # [Time + Tail]

            # --- 5. CRITICAL FIX: The "Scissors" ---
            # The simulator adds a reverb tail. We MUST crop it to exactly TARGET_LEN samples.
            mixed = mixed.T # [Time, Chan]
            
            # Crop Mixed
            if mixed.shape[0] > TARGET_LEN:
                mixed = mixed[:TARGET_LEN, :]
            elif mixed.shape[0] < TARGET_LEN:
                diff = TARGET_LEN - mixed.shape[0]
                mixed = np.pad(mixed, ((0, diff), (0, 0)))

            # Crop Clean
            if clean.shape[0] > TARGET_LEN:
                clean = clean[:TARGET_LEN]
            elif clean.shape[0] < TARGET_LEN:
                diff = TARGET_LEN - clean.shape[0]
                clean = np.pad(clean, (0, diff))

            # --- 6. Normalize & Save ---
            peak = max(np.max(np.abs(mixed)), np.max(np.abs(clean))) + 1e-9
            mixed = mixed / peak * 0.9
            clean = clean / peak * 0.9

            sf.write(f"{output_dir}/noisy/sample_{i:05d}.wav", mixed, 16000)
            sf.write(f"{output_dir}/clean/sample_{i:05d}.wav", clean, 16000)

        except Exception: continue

if __name__ == "__main__":
    # --- PATH CONFIG ---
    SPEECH_ROOT = "./LibriSpeech/dev-clean"
    NOISE_ROOT = "./noise"
    COMMANDS_ROOT = "./commands"
    OUT = "./data"
    
    # 1. Indexing
    print("Indexing...")
    speech = glob.glob(f"{SPEECH_ROOT}/**/*.flac", recursive=True) + glob.glob(f"{SPEECH_ROOT}/**/*.wav", recursive=True)
    noise = glob.glob(f"{NOISE_ROOT}/**/*.wav", recursive=True)
    cmds = glob.glob(f"{COMMANDS_ROOT}/*.wav") if os.path.exists(COMMANDS_ROOT) else []
    
    if not speech or not noise:
        print("Error: Files not found.")
        exit()
    
    random.shuffle(speech); random.shuffle(noise); random.shuffle(cmds)
    
    # Split Train/Val
    split_s, split_n = int(len(speech)*0.1), int(len(noise)*0.1)
    sp_v, sp_t = speech[:split_s], speech[split_s:]
    no_v, no_t = noise[:split_n], noise[split_n:]
    cm_v, cm_t = cmds[:int(len(cmds)*0.1)], cmds[int(len(cmds)*0.1):]

    # 2. Generate Curriculum
    # Phase 1: Easy (15-25dB)
    generate_subset(sp_t, no_t, f"{OUT}/train_easy", 2000, 15, 25, cm_t)
    # Phase 2: Medium (5-15dB)
    generate_subset(sp_t, no_t, f"{OUT}/train_medium", 2000, 5, 15, cm_t)
    # Phase 3: Hard (-5-5dB)
    generate_subset(sp_t, no_t, f"{OUT}/train_hard", 2000, -5, 5, cm_t)
    
    # Validation: Standard (0-20dB)
    generate_subset(sp_v, no_v, f"{OUT}/val", 200, 0, 20, cm_v)