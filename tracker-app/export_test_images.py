#!/usr/bin/env python3
"""Export test images to JPG"""
from PIL import Image

# Convert all three test images to JPG
for algo in ['simple_average', 'ground_filter', 'cpu_fallback']:
    png_path = f'/tmp/fused_test_{algo}.png'
    jpg_path = f'/tmp/fused_test_{algo}.jpg'

    img = Image.open(png_path)
    img.convert('RGB').save(jpg_path, 'JPEG', quality=95)
    print(f"Exported {jpg_path}")

# Also copy to share directory
import shutil
for algo in ['simple_average', 'ground_filter', 'cpu_fallback']:
    src = f'/tmp/fused_test_{algo}.jpg'
    dst = f'/home/andrew/share/fused_test_{algo}.jpg'
    shutil.copy(src, dst)
    print(f"Copied to {dst}")

print("\nAll images exported to /tmp and ~/share")