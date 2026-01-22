import numpy as np
import onnxruntime as ort
import soundfile as sf

class Streamer:
    def __init__(self, onnx_path, block_len=960, hop_len=480, hidden_dim=256):
        self.block_len = block_len
        self.hop_len = hop_len
        self.hidden_dim = hidden_dim
        
        print(f"Loading Model: {onnx_path}...")
        self.session = ort.InferenceSession(onnx_path)
        
        self.in_buffer = np.zeros((self.block_len, 2), dtype=np.float32)
        self.out_buffer = np.zeros((self.block_len,), dtype=np.float32)
        self.window = np.hanning(self.block_len).astype(np.float32)
        self.hidden = np.zeros((2, 1, self.hidden_dim), dtype=np.float32)
        
        # DC Filter State
        self.last_x = 0.0
        self.last_y = 0.0
        self.R = 0.995

    def dc_block(self, audio_chunk):
        output = np.zeros_like(audio_chunk)
        prev_x = self.last_x
        prev_y = self.last_y
        for i in range(len(audio_chunk)):
            x = audio_chunk[i]
            y = x - prev_x + self.R * prev_y
            output[i] = y
            prev_x = x
            prev_y = y
        self.last_x = prev_x
        self.last_y = prev_y
        return output

    def process_frame(self, chunk):
        # 1. Update Input Buffer
        self.in_buffer = np.roll(self.in_buffer, -self.hop_len, axis=0)
        self.in_buffer[-self.hop_len:, :] = chunk
        
        # 2. Analysis
        mic1_spec = np.fft.rfft(self.in_buffer[:, 0] * self.window, n=self.block_len)
        mic2_spec = np.fft.rfft(self.in_buffer[:, 1] * self.window, n=self.block_len)
        
        # 3. Inference
        input_tensor = np.zeros((1, 1, 4, 481), dtype=np.float32)
        input_tensor[0, 0, 0] = mic1_spec.real
        input_tensor[0, 0, 1] = mic1_spec.imag
        input_tensor[0, 0, 2] = mic2_spec.real
        input_tensor[0, 0, 3] = mic2_spec.imag
        
        outputs = self.session.run(None, {'input_audio': input_tensor, 'hidden_in': self.hidden})
        est_spec, self.hidden = outputs
        
        # KILL DC (Fixes the "wavy" drift from Phase 2)
        est_spec[:, :, :, 0] = 0.0 

        # 4. Synthesis
        est_complex = est_spec[0, 0, 0] + 1j * est_spec[0, 0, 1]
        out_frame = np.fft.irfft(est_complex, n=self.block_len)
        
        # 5. OLA
        out_frame = out_frame * self.window
        self.out_buffer += out_frame
        
        output_chunk = self.out_buffer[:self.hop_len].copy()
        
        self.out_buffer = np.roll(self.out_buffer, -self.hop_len)
        self.out_buffer[-self.hop_len:] = 0.0
        
        # Time-Domain DC Block
        output_chunk = self.dc_block(output_chunk)
        
        return output_chunk

def infer_file(onnx_path, input_path, output_path):
    streamer = Streamer(onnx_path)
    audio, sr = sf.read(input_path)
    if sr != 16000: return

    if len(audio.shape) == 1: audio = np.stack([audio, audio], axis=1)
    
    processed_audio = []
    hop = streamer.hop_len
    
    print("Streaming (Phase 2 Model + DC Fix)...")
    for i in range(0, len(audio), hop):
        chunk = audio[i : i+hop]
        if len(chunk) < hop:
            pad_len = hop - len(chunk)
            chunk = np.pad(chunk, ((0, pad_len), (0, 0)))
            
        out_chunk = streamer.process_frame(chunk)
        processed_audio.extend(out_chunk)
        
    processed_audio = np.array(processed_audio)
    
    # Normalize (Fixes the "Quiet" issue from Phase 2)
    peak = np.max(np.abs(processed_audio))
    if peak > 0: processed_audio = processed_audio / peak * 0.95
    
    sf.write(output_path, processed_audio, 16000)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    # Ensure you re-exported best_model.pth!
    infer_file("dc_gtcrn_stream.onnx", "data/train_hard/noisy/sample_00005.wav", "cleaned_phase2_fixed.wav")