#!/usr/bin/env python3
"""Debug CPU fallback rendering"""
import sys
import os
sys.path.insert(0, '/home/andrew/toddler-tracker/tracker-app')

import numpy as np
import trimesh
from PIL import Image

# Load the PLY file from project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ply_path = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')
print(f"Loading {ply_path}...")
mesh = trimesh.load(ply_path)

points = mesh.vertices.astype(np.float32)
colors = mesh.visual.vertex_colors[:, :3].astype(np.uint8)

print(f"✅ Loaded {len(points):,} points")

# Calculate boundaries
x_min, x_max = np.percentile(points[:, 0], [2, 98])
z_min, z_max = np.percentile(points[:, 2], [2, 98])
center_x = (x_min + x_max) / 2
center_z = (z_min + z_max) / 2

# Manual CPU rasterization with debug output
width, height = 800, 800
width_m = (x_max - x_min) * 1.2
height_m = (z_max - z_min) * 1.2
resolution = max(width_m, height_m) / 800

print(f"\nResolution: {resolution:.4f}m/pixel")

# Calculate view bounds
half_width = (width * resolution) / 2
half_height = (height * resolution) / 2
view_x_min = center_x - half_width
view_x_max = center_x + half_width
view_y_min = center_z - half_height
view_y_max = center_z + half_height

print(f"View bounds: X[{view_x_min:.2f}, {view_x_max:.2f}], Z[{view_y_min:.2f}, {view_y_max:.2f}]")

# Project to 2D (Y,Z for looking down X-axis)
vertices_2d = points[:, [1, 2]]

print(f"2D vertices shape: {vertices_2d.shape}")
print(f"2D X range: [{vertices_2d[:, 0].min():.2f}, {vertices_2d[:, 0].max():.2f}]")
print(f"2D Y range: [{vertices_2d[:, 1].min():.2f}, {vertices_2d[:, 1].max():.2f}]")

# Filter points within view
mask = ((vertices_2d[:, 0] >= view_x_min) & (vertices_2d[:, 0] <= view_x_max) &
        (vertices_2d[:, 1] >= view_y_min) & (vertices_2d[:, 1] <= view_y_max))

visible_2d = vertices_2d[mask]
visible_colors = colors[mask]

print(f"\nVisible points: {len(visible_2d):,} / {len(points):,} ({len(visible_2d)/len(points)*100:.1f}%)")

if len(visible_2d) == 0:
    print("❌ NO VISIBLE POINTS! This is the problem.")
    print("\nDEBUG: Let's check why...")
    print(f"  center_x (used for view): {center_x:.2f}")
    print(f"  center_z (used for view): {center_z:.2f}")
    print(f"  2D vertices use columns [1,2] which are Y,Z")
    print(f"  But view bounds use center_x and center_z")
    print(f"  PROBLEM: Comparing Y coordinates against X center!")
else:
    # Rasterize
    output = np.full((height, width, 3), 255, dtype=np.uint8)  # White background
    pixel_counts = np.zeros((height, width), dtype=np.int32)
    color_accum = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(len(visible_2d)):
        px = int((visible_2d[i, 0] - view_x_min) / resolution)
        py = height - 1 - int((visible_2d[i, 1] - view_y_min) / resolution)

        if 0 <= px < width and 0 <= py < height:
            pixel_counts[py, px] += 1
            color_accum[py, px] += visible_colors[i]

    # Average colors
    mask = pixel_counts > 0
    output[mask] = (color_accum[mask] / pixel_counts[mask][:, np.newaxis]).astype(np.uint8)

    print(f"Pixels with points: {mask.sum():,}")
    print(f"Mean color: R={output[mask, 0].mean():.1f}, G={output[mask, 1].mean():.1f}, B={output[mask, 2].mean():.1f}")

    # Save
    img = Image.fromarray(output)
    img.save('/tmp/cpu_fallback_debug.png')
    print("\n✅ Saved to /tmp/cpu_fallback_debug.png")