import torch
import torch.nn as nn
import torch.optim as optim
from models.dc_gtcrn import DC_GTCRN
from dataloader import create_loader
import os
import time
from tqdm import tqdm

def train():
    EPOCHS = 200
    BATCH_SIZE = 32
    LR = 0.001
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {DEVICE}...")

    os.makedirs("checkpoints", exist_ok=True)
    
    # Data
    print("Loading Data Index...")
    train_loader, val_loader, train_ds = create_loader(BATCH_SIZE, num_workers=4)
    
    # Model
    model = DC_GTCRN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    # Pre-calculate Window to save time (move out of loop)
    win = torch.hann_window(960).to(DEVICE)
    
    best_loss = float('inf')

    # --- HELPER FUNCTION (Defined once, dynamic shapes) ---
    def get_spec(x):
        # Dynamically get shape from input x
        nb, nc, nt = x.shape
        
        # STFT
        stft = torch.stft(x.reshape(nb*nc, nt), 960, 480, 960, window=win, return_complex=False)
        
        # Reshape to [Batch, Time, 2, Freq]
        stft = stft.reshape(nb, 481, -1, 2).permute(0, 2, 3, 1)
        
        # Duplicate to Stereo [Batch, Time, 4, Freq]
        return torch.cat([stft, stft], dim=2)

    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        
        # Curriculum Strategy
        if epoch < 50:
            train_ds.set_snr(15, 30)
            stage = "EASY (15-30dB)"
        elif epoch < 100:
            train_ds.set_snr(5, 15)
            stage = "MED (5-15dB)"
        else:
            train_ds.set_snr(-5, 5)
            stage = "HARD (-5-5dB)"
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{stage}]")
        
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
                
                # Now safe to call: uses dynamic shapes
                noisy_spec = get_spec(noisy)
                clean_spec = get_spec(clean)
                
                est, _ = model(noisy_spec)
                val_loss += criterion(est, clean_spec).item()
                
        avg_val = val_loss / len(val_loader)
        avg_train = train_loss / len(train_loader)
        
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   [Summary] Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {current_lr:.6f} | Time: {time.time()-start:.1f}s")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("   [!] New Best Model Saved")

        # Save Checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()