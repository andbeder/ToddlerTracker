#!/usr/bin/env python3
"""Compare fused.ply colors with saved map reference"""
from PIL import Image
import numpy as np

# Load saved map reference
saved_img = Image.open('/tmp/saved_map_reference.png')
saved_arr = np.array(saved_img)

print("=== Saved Map Reference (What we're trying to match) ===")
print(f"Shape: {saved_arr.shape}")
print(f"Mean color: R={saved_arr[:,:,0].mean():.1f}, G={saved_arr[:,:,1].mean():.1f}, B={saved_arr[:,:,2].mean():.1f}")

# Load fused.ply color samples
fused_samples = Image.open('/tmp/fused_color_samples.png')
fused_arr = np.array(fused_samples)

print("\n=== Fused.ply Colors (From trimesh) ===")
print(f"Shape: {fused_arr.shape}")
print(f"Mean color: R={fused_arr[:,:,0].mean():.1f}, G={fused_arr[:,:,1].mean():.1f}, B={fused_arr[:,:,2].mean():.1f}")

print("\n=== Comparison ===")
print(f"Saved map is more muted (R~63, G~65, B~55)")
print(f"Fused.ply is brighter (R~145, G~149, B~127)")
print(f"These appear to be DIFFERENT datasets or the saved map has different processing")

# Check if saved map might be using a different source
print("\n=== Analysis ===")
print("The saved map colors (R=63.4, G=65.3, B=54.9) are much darker than")
print("the fused.ply colors (R=145.0, G=148.9, B=127.2).")
print("Either:")
print("1. The saved map was generated from a different PLY file")
print("2. The saved map applied some color correction/darkening")
print("3. CloudCompare shows fused.ply differently than what trimesh extracts")