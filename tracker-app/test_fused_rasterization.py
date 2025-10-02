#!/usr/bin/env python3
"""Test full rasterization pipeline with fused.ply"""
import sys
import os
sys.path.insert(0, '/home/andrew/toddler-tracker/tracker-app')

import numpy as np
import trimesh
from cuda_rasterizer_optimized import OptimizedCudaRasterizer
from PIL import Image

# Load the PLY file from project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ply_path = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')
print(f"Loading {ply_path}...")
mesh = trimesh.load(ply_path)

points = mesh.vertices.astype(np.float32)
colors = mesh.visual.vertex_colors[:, :3].astype(np.uint8)

print(f"✅ Loaded {len(points):,} points")
print(f"Colors dtype: {colors.dtype}, shape: {colors.shape}")

# Calculate boundaries
x_min, x_max = np.percentile(points[:, 0], [2, 98])
y_min, y_max = np.percentile(points[:, 1], [2, 98])
z_min, z_max = np.percentile(points[:, 2], [2, 98])

center_x = (x_min + x_max) / 2
center_z = (z_min + z_max) / 2

print(f"\nBoundaries:")
print(f"  X: [{x_min:.2f}, {x_max:.2f}], center: {center_x:.2f}")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
print(f"  Z: [{z_min:.2f}, {z_max:.2f}], center: {center_z:.2f}")

# Test all three algorithms
rasterizer = OptimizedCudaRasterizer()

algorithms = ['simple_average', 'ground_filter', 'cpu_fallback']
output_size = (800, 800)

# Calculate appropriate resolution to fit the whole point cloud
# Point cloud is ~10.5m x 9.7m, so use resolution that shows the full area
width_m = x_max - x_min
height_m = z_max - z_min
# Add 20% padding
width_m *= 1.2
height_m *= 1.2
# Choose resolution based on larger dimension
resolution = max(width_m, height_m) / 800
print(f"\nCalculated resolution: {resolution:.4f}m/pixel (to fit {width_m:.2f}m x {height_m:.2f}m view)")

for algo in algorithms:
    print(f"\n{'='*50}")
    print(f"Testing algorithm: {algo}")
    print('='*50)

    try:
        result = rasterizer.rasterize_point_cloud(
            points=points,
            colors=colors,
            center_x=center_x,
            center_y=center_z,
            rotation=0.0,
            resolution=resolution,
            output_size=output_size,
            algorithm=algo
        )

        # Save the output
        output_path = f'/tmp/fused_test_{algo}.png'
        img = Image.fromarray(result)
        img.save(output_path)

        # Analyze colors
        non_white = result[np.any(result != [255, 255, 255], axis=2)]
        if len(non_white) > 0:
            mean_r = non_white[:, 0].mean()
            mean_g = non_white[:, 1].mean()
            mean_b = non_white[:, 2].mean()

            print(f"✅ Saved to {output_path}")
            print(f"Non-white pixels: {len(non_white):,}")
            print(f"Mean color: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")

            # Sample some pixels
            if len(non_white) >= 10:
                sample_indices = np.random.choice(len(non_white), 10, replace=False)
                print("\nSample pixels:")
                for idx in sample_indices:
                    r, g, b = non_white[idx]
                    print(f"  RGB({r}, {g}, {b})")
        else:
            print(f"⚠️  All white pixels - no data rendered!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("Test complete! Check /tmp/fused_test_*.png files")