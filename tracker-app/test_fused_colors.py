#!/usr/bin/env python3
"""Test color extraction from fused.ply using trimesh"""
import os
import numpy as np
import trimesh
from PIL import Image

# Load the PLY file from project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ply_path = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')
print(f"Loading {ply_path}...")
mesh = trimesh.load(ply_path)

# Extract points and colors
points = mesh.vertices
colors = None

if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    colors = mesh.visual.vertex_colors[:, :3]  # RGB only, drop alpha
    print(f"✅ Loaded {len(points):,} points with colors")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    print(f"Colors dtype: {colors.dtype}")

    # Color statistics
    print("\n=== Color Statistics (from trimesh) ===")
    print(f"Mean color: R={colors[:,0].mean():.1f}, G={colors[:,1].mean():.1f}, B={colors[:,2].mean():.1f}")
    print(f"Std:  R={colors[:,0].std():.1f}, G={colors[:,1].std():.1f}, B={colors[:,2].std():.1f}")
    print(f"Range: R[{colors[:,0].min()}-{colors[:,0].max()}], "
          f"G[{colors[:,1].min()}-{colors[:,1].max()}], "
          f"B[{colors[:,2].min()}-{colors[:,2].max()}]")

    # Sample some random points
    print("\n=== Random Color Samples ===")
    for i in range(10):
        idx = np.random.randint(0, len(colors))
        r, g, b = colors[idx]
        print(f"  Point {idx}: RGB({r}, {g}, {b})")

    # Color analysis
    print("\n=== Color Analysis ===")
    green_dominant = np.sum(colors[:,1] > colors[:,0]) / len(colors)
    print(f"Pixels with G > R: {green_dominant*100:.1f}%")

    purple_tint = np.sum((colors[:,0] > 100) & (colors[:,2] > 100) & (colors[:,1] < 100)) / len(colors)
    print(f"Pixels with purple tint: {purple_tint*100:.1f}%")

    # Check for greenish-brown (what we expect for grass/ground)
    # Greenish-brown: R~60-80, G~60-80, B~50-70
    greenish_brown = np.sum((colors[:,0] >= 50) & (colors[:,0] <= 100) &
                            (colors[:,1] >= 50) & (colors[:,1] <= 100) &
                            (colors[:,2] >= 40) & (colors[:,2] <= 80)) / len(colors)
    print(f"Pixels with greenish-brown: {greenish_brown*100:.1f}%")

    # Create a simple visualization - sample 1000 random colors
    sample_size = min(1000, len(colors))
    sample_indices = np.random.choice(len(colors), sample_size, replace=False)
    sample_colors = colors[sample_indices]

    # Create a 20x50 grid showing the color samples
    img_height = 20
    img_width = 50
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i in range(img_height):
        for j in range(img_width):
            idx = i * img_width + j
            if idx < len(sample_colors):
                img[i, j] = sample_colors[idx]

    # Save the color sample visualization
    img_pil = Image.fromarray(img)
    output_path = '/tmp/fused_color_samples.png'
    img_pil.save(output_path)
    print(f"\n✅ Saved color samples to {output_path}")

else:
    print("❌ No colors found in PLY file!")