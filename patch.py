import onnx

def patch_opset():
    filename = "dc_gtcrn_stream.onnx"
    print(f"Reading {filename}...")
    model = onnx.load(filename)
    
    # Check current version
    current_version = model.opset_import[0].version
    print(f"Current Header Version: {current_version}")
    
    # Force it to 17 (Stable)
    if current_version == 18:
        print("PATCHING: Rewriting header from 18 to 17...")
        model.opset_import[0].version = 17
        onnx.save(model, filename)
        print("Success! File patched.")
    else:
        print(f"Version is {current_version}. No patch needed.")

if __name__ == "__main__":
    patch_opset()
