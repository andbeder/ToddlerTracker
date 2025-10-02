#!/usr/bin/env python3
"""
Simple PLY rasterizer with correct XZ projection and Z-flip.
This produces the correct top-down view looking down the Y-axis.
"""
import numpy as np
import trimesh
from PIL import Image

def simple_rasterize_xz(ply_path, output_path, resolution=0.01, image_size=(1920, 1080)):
    """
    Simple rasterization with XZ projection (looking down Y-axis) and Z-flip.

    Args:
        ply_path: Path to fused.ply
        output_path: Where to save output image
        resolution: Meters per pixel
        image_size: Output image size (width, height)
    """
    print(f"Loading PLY file: {ply_path}")
    mesh = trimesh.load(ply_path)

    points = mesh.vertices  # Nx3 array [X, Y, Z]
    colors = mesh.visual.vertex_colors[:, :3]  # Nx3 array [R, G, B]

    print(f"Loaded {len(points)} points")
    print(f"Color range: R[{colors[:,0].min()}-{colors[:,0].max()}], "
          f"G[{colors[:,1].min()}-{colors[:,1].max()}], "
          f"B[{colors[:,2].min()}-{colors[:,2].max()}]")

    # Project to 2D: use X and Z (looking down Y-axis)
    # X is horizontal, Z is depth, Y is height (vertical)
    coords_2d = points[:, [0, 2]].copy()  # [X, Z]
    coords_2d[:, 1] = -coords_2d[:, 1]  # Flip Z vertically for correct orientation

    # Find bounds (use percentiles to exclude outliers)
    x_min, x_max = np.percentile(coords_2d[:, 0], [2, 98])
    z_min, z_max = np.percentile(coords_2d[:, 1], [2, 98])

    print(f"Bounds (2-98 percentile): X[{x_min:.2f}, {x_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")

    # Calculate center
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2

    # Calculate view bounds based on image size and resolution
    width, height = image_size
    half_width = (width * resolution) / 2
    half_height = (height * resolution) / 2

    view_x_min = center_x - half_width
    view_x_max = center_x + half_width
    view_z_min = center_z - half_height
    view_z_max = center_z + half_height

    print(f"View bounds: X[{view_x_min:.2f}, {view_x_max:.2f}], Z[{view_z_min:.2f}, {view_z_max:.2f}]")
    print(f"Resolution: {resolution}m/pixel, Image size: {width}x{height}")

    # Create image (black background)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    pixel_counts = np.zeros((height, width), dtype=np.int32)
    color_accumulator = np.zeros((height, width, 3), dtype=np.float64)

    # Rasterize: map each point to a pixel
    print("Rasterizing points...")
    points_in_view = 0

    for i in range(len(points)):
        x, z = coords_2d[i]

        # Check if point is in view
        if not (view_x_min <= x <= view_x_max and view_z_min <= z <= view_z_max):
            continue

        points_in_view += 1

        # Map to pixel coordinates
        # X maps to column (px), Z maps to row (py)
        px = int((x - view_x_min) / resolution)
        py = int((z - view_z_min) / resolution)

        # Clamp to image bounds
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))

        # Accumulate color
        color_accumulator[py, px] += colors[i]
        pixel_counts[py, px] += 1

    print(f"Points in view: {points_in_view} ({points_in_view/len(points)*100:.1f}%)")

    # Average colors
    print("Averaging colors...")
    mask = pixel_counts > 0
    for c in range(3):
        image[mask, c] = (color_accumulator[mask, c] / pixel_counts[mask]).astype(np.uint8)

    # Calculate statistics
    data_pixels = np.sum(mask)
    print(f"Data pixels: {data_pixels} ({data_pixels/(width*height)*100:.1f}%)")
    print(f"Black pixels: {width*height - data_pixels} ({(width*height - data_pixels)/(width*height)*100:.1f}%)")

    if data_pixels > 0:
        data_colors = image[mask]
        print(f"Data pixel colors: R={data_colors[:,0].mean():.1f}, G={data_colors[:,1].mean():.1f}, B={data_colors[:,2].mean():.1f}")
        print(f"Data pixel std: R={data_colors[:,0].std():.1f}, G={data_colors[:,1].std():.1f}, B={data_colors[:,2].std():.1f}")

    # Save image
    print(f"Saving to {output_path}")
    img = Image.fromarray(image)
    img.save(output_path)
    print("Done!")

    return image


if __name__ == '__main__':
    import os
    # Use project PLY path
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLY_PATH = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')

    image = simple_rasterize_xz(
        ply_path=PLY_PATH,
        output_path='/home/andrew/share/algorithm1.png',
        resolution=0.01,
        image_size=(1920, 1080)
    )
