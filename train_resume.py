import torch
import torch.nn as nn
import torch.optim as optim
from models.dc_gtcrn import DC_GTCRN
from dataloader import create_loader
import os
import time
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def train():
    args = get_args()
    
    # --- CONFIG ---
    EPOCHS = 200
    BATCH_SIZE = 32
    LR = 0.001
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {DEVICE}...")

    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. SETUP DATA
    print("Loading Data Index...")
    train_loader, val_loader, train_ds = create_loader(BATCH_SIZE, num_workers=4)
    # Access the Validation Dataset object inside the loader
    val_ds = val_loader.dataset 
    
    # 2. SETUP MODEL
    model = DC_GTCRN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    # Pre-calculate Window
    win = torch.hann_window(960).to(DEVICE)
    
    # State tracking
    start_epoch = 0
    best_loss = float('inf')
    
    # Stage Tracker (To detect curriculum changes)
    current_stage = 0 

    # --- 3. RESUME LOGIC ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            
            # Infer stage from resumed epoch
            if start_epoch < 50: current_stage = 0
            elif start_epoch < 100: current_stage = 1
            else: current_stage = 2
            
            print(f"Resumed from Epoch {checkpoint['epoch']} (Best Loss: {best_loss:.6f})")
        else:
            print(f"Checkpoint '{args.resume}' not found. Starting from scratch.")

    # --- HELPER ---
    def get_spec(x):
        nb, nc, nt = x.shape
        stft = torch.stft(x.reshape(nb*nc, nt), 960, 480, 960, window=win, return_complex=False)
        stft = stft.reshape(nb, 481, -1, 2).permute(0, 2, 3, 1)
        return torch.cat([stft, stft], dim=2)

    # --- 4. LOOP ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        start = time.time()
        
        # --- CURRICULUM CONTROLLER ---
        # Update BOTH Train and Val datasets
        # And RESET best_loss if stage changes to avoid "Moving Goalpost" bug
        
        new_stage = 0
        if epoch < 50:
            train_ds.set_snr_range(15, 30)
            val_ds.set_snr_range(15, 30) # <--- SYNCED VALIDATION
            stage_name = "EASY (15-30dB)"
            new_stage = 0
        elif epoch < 100:
            train_ds.set_snr_range(5, 15)
            val_ds.set_snr_range(5, 15)  # <--- SYNCED VALIDATION
            stage_name = "MED (5-15dB)"
            new_stage = 1
        else:
            train_ds.set_snr_range(-5, 5)
            val_ds.set_snr_range(-5, 5)  # <--- SYNCED VALIDATION
            stage_name = "HARD (-5-5dB)"
            new_stage = 2
            
        # CHECK FOR STAGE TRANSITION
        if new_stage != current_stage:
            print(f"\n[!] CURRICULUM ADVANCE: {current_stage} -> {new_stage}")
            print("[!] Resetting Best Loss Tracker for new difficulty.")
            best_loss = float('inf')
            current_stage = new_stage
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{stage_name}]")
        
        train_loss = 0
        
        for i, (noisy, clean) in enumerate(pbar):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            noisy_spec = get_spec(noisy)
            clean_spec = get_spec(clean)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    est, _ = model(noisy_spec)
                    loss = criterion(est, clean_spec)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                est, _ = model(noisy_spec)
                loss = criterion(est, clean_spec)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
            loss_val = loss.item()
            train_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.6f}"})

        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                noisy_spec = get_spec(noisy)
                clean_spec = get_spec(clean)
                est, _ = model(noisy_spec)
                val_loss += criterion(est, clean_spec).item()
                
        avg_val = val_loss / len(val_loader)
        avg_train = train_loss / len(train_loader)
        
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   [Summary] Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.6f} | Time: {time.time()-start:.1f}s")
        
        # --- CHECKPOINTING ---
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }
        if scaler: checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

        # Always save 'last.pth'
        torch.save(checkpoint_dict, "checkpoints/last.pth")
        
        # Save Best (Resets per stage)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(checkpoint_dict, "checkpoints/best_model.pth")
            print("   [!] New Best Model Saved (For this Stage)")

        # Historic Save
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, f"checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
