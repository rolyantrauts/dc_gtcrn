import torch
import torch.nn as nn

class DC_GTCRN(nn.Module):
    def __init__(self, n_fft=960, hop_length=480, hidden_dim=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # --- ENCODER ---
        # Input: [Batch, Channels, Freq, Time]
        # We use Kernel=(5, 1) so we look at 5 Freq bins, but only 1 Time step.
        # This makes the Conv layer "Stateless" in time (Perfect for streaming).
        
        # Layer 1: Downsample Freq (481 -> 241)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: Deep Features (241 -> 241)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- GRU (Temporal Processor) ---
        # Input: 64 Channels * 241 Freq Bins = 15424 Features
        self.gru = nn.GRU(64 * 241, hidden_dim, num_layers=2, batch_first=True)
        
        # --- DECODER ---
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.bn3 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(32, 4, kernel_size=(5, 1), stride=(2, 1), output_padding=(0, 0), padding=(2, 0))
        
        self.fc_out = nn.Linear(hidden_dim, 64 * 241)

    def forward(self, x, hidden=None):
        # x: [Batch, Time, 4, Freq]
        b, t, c, f = x.shape
        
        # Permute for Conv2d: [B, C, F, T]
        x = x.permute(0, 2, 3, 1)
        
        # Encoder
        e1 = torch.relu(self.bn1(self.conv1(x)))      # [B, 32, 241, T]
        e2 = torch.relu(self.bn2(self.conv2(e1)))     # [B, 64, 241, T]
        
        # Reshape for GRU: [B, T, Features]
        # We permute to [B, T, C, F] then flatten C*F
        gru_in = e2.permute(0, 3, 1, 2).reshape(b, t, -1)
        
        # GRU
        gru_out, new_hidden = self.gru(gru_in, hidden)
        
        # Project back
        dec_in = torch.relu(self.fc_out(gru_out))
        dec_in = dec_in.reshape(b, t, 64, 241).permute(0, 2, 3, 1) # [B, 64, 241, T]
        
        # Decoder (with Skip Connections)
        d1 = torch.relu(self.bn3(self.deconv1(dec_in + e2)))
        d2 = self.deconv2(d1 + e1)
        
        # Output: [B, T, 4, F]
        out = d2.permute(0, 3, 1, 2)
        
        return out, new_hidden