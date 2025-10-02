#!/usr/bin/env python3
"""Debug NPY color storage"""
import numpy as np
import json

npy_dir = '/home/andrew/toddler-tracker/tracker-app/npy_storage/20250930_051727_fused'

# Load colors from NPY
colors = np.load(f'{npy_dir}/colors.npy', mmap_mode='r')

print(f"NPY Colors shape: {colors.shape}, dtype: {colors.dtype}")
print(f"Mean: R={colors[:,0].mean():.1f}, G={colors[:,1].mean():.1f}, B={colors[:,2].mean():.1f}")

# Sample points
print("\nSample colors from NPY:")
for i in range(10):
    idx = np.random.randint(0, len(colors))
    print(f"  Point {idx}: RGB({colors[idx, 0]}, {colors[idx, 1]}, {colors[idx, 2]})")

# Now load the original PLY with trimesh
import trimesh
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_PATH = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')
mesh = trimesh.load(PLY_PATH)
trimesh_colors = mesh.visual.vertex_colors[:, :3]

print(f"\n=== Trimesh Colors (from PLY) ===")
print(f"Shape: {trimesh_colors.shape}, dtype: {trimesh_colors.dtype}")
print(f"Mean: R={trimesh_colors[:,0].mean():.1f}, G={trimesh_colors[:,1].mean():.1f}, B={trimesh_colors[:,2].mean():.1f}")

# Compare
print("\n=== COMPARISON ===")
print(f"NPY:     R={colors[:,0].mean():.1f}, G={colors[:,1].mean():.1f}, B={colors[:,2].mean():.1f}")
print(f"Trimesh: R={trimesh_colors[:,0].mean():.1f}, G={trimesh_colors[:,1].mean():.1f}, B={trimesh_colors[:,2].mean():.1f}")

if (colors[:,0].mean() != trimesh_colors[:,0].mean()):
    print("\n❌ COLORS DON'T MATCH! NPY storage is corrupting colors!")
    print("Checking if it's a BGR/RGB swap...")
    if abs(colors[:,0].mean() - trimesh_colors[:,2].mean()) < 1:
        print("✅ YES! NPY has BGR, trimesh has RGB")
else:
    print("\n✅ Colors match")