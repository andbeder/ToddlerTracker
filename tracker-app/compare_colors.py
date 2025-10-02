#!/usr/bin/env python3
"""Compare colors between saved map and CUDA output"""
import sqlite3
import numpy as np
from PIL import Image
import io

# Load saved map from database
conn = sqlite3.connect('yard.db')
cursor = conn.cursor()
cursor.execute('SELECT image_data FROM yard_maps WHERE name = "Backyard" ORDER BY created_at DESC LIMIT 1')
result = cursor.fetchone()

if result:
    # Extract and load image
    image_data = result[0]
    saved_img = Image.open(io.BytesIO(image_data))
    saved_arr = np.array(saved_img)

    print("=== Saved Map (Correct Colors) ===")
    print(f"Shape: {saved_arr.shape}")
    print(f"Mean color: R={saved_arr[:,:,0].mean():.1f}, G={saved_arr[:,:,1].mean():.1f}, B={saved_arr[:,:,2].mean():.1f}")
    print(f"Std:  R={saved_arr[:,:,0].std():.1f}, G={saved_arr[:,:,1].std():.1f}, B={saved_arr[:,:,2].std():.1f}")
    print(f"Range: R[{saved_arr[:,:,0].min()}-{saved_arr[:,:,0].max()}], "
          f"G[{saved_arr[:,:,1].min()}-{saved_arr[:,:,1].max()}], "
          f"B[{saved_arr[:,:,2].min()}-{saved_arr[:,:,2].max()}]")

    # Sample some pixels
    print("\nSample pixels (10 random):")
    h, w = saved_arr.shape[:2]
    for i in range(10):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        r, g, b = saved_arr[y, x, 0], saved_arr[y, x, 1], saved_arr[y, x, 2]
        print(f"  ({x},{y}): RGB({r}, {g}, {b})")

    # Check for specific color patterns
    print("\nColor analysis:")
    # Check if colors are greenish (correct for grass/ground)
    green_dominant = np.sum(saved_arr[:,:,1] > saved_arr[:,:,0]) / (h * w)
    print(f"  Pixels with G > R: {green_dominant*100:.1f}%")

    # Check for purple/magenta tint (incorrect)
    purple_tint = np.sum((saved_arr[:,:,0] > 100) & (saved_arr[:,:,2] > 100) & (saved_arr[:,:,1] < 100)) / (h * w)
    print(f"  Pixels with purple tint: {purple_tint*100:.1f}%")

    # Save for reference
    saved_img.save('/tmp/saved_map_reference.png')
    print("\nSaved reference image to /tmp/saved_map_reference.png")
else:
    print("No saved map found!")

conn.close()