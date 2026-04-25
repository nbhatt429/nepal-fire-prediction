# Run in Colab — saves grid_static_scaled.npy to Drive
import numpy as np
DRIVE = '/content/drive/MyDrive'

# Already computed — just confirm it exists
import os
path = f'{DRIVE}/grid_static_scaled.npy'
if os.path.exists(path):
    arr = np.load(path)
    print(f"grid_static_scaled.npy: {arr.shape}")
    print(f"Mean: {arr.mean():.4f}")
    print(f"Ready to upload to GitHub")
else:
    print("Not found — rerun the KD-tree matching code")
