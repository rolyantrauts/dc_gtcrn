import torch
from models.dc_gtcrn import DC_GTCRN
import os

def export():
    print("Loading Best Model...")
    model = DC_GTCRN()
    ckpt = "checkpoints/best_model.pth"
    
    if not os.path.exists(ckpt):
        print("Error: No checkpoint found.")
        return
        
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    
    # Input: [Batch=1, Time=1, Channels=4, Freq=481]
    dummy_input = torch.randn(1, 1, 4, 481)
    
    # Hidden: [Layers=2, Batch=1, Dim=256]
    dummy_hidden = torch.zeros(2, 1, 256)
    
    print("Exporting ONNX...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_hidden),
        "dc_gtcrn_stream.onnx",
        input_names=['input_audio', 'hidden_in'],
        output_names=['output_spec', 'hidden_out'],
        opset_version=16
    )
    print("Done: dc_gtcrn_stream.onnx")

if __name__ == "__main__":
    export()