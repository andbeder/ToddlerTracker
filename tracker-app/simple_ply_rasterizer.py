#!/usr/bin/env python3
"""
Simple PLY rasterizer - reads fused.ply and creates a basic top-down view.
No flipping, no complex algorithms - just average colors per pixel with black background.
"""
import numpy as np
import trimesh
from PIL import Image

def simple_rasterize(ply_path, output_path, resolution=0.01, image_size=(1920, 1080)):
    """
    Simple rasterization: read PLY, project to 2D, average colors per pixel.

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

    # Project to 2D: use Y and Z (looking down X-axis)
    # NO FLIPPING - just use the raw coordinates
    coords_2d = points[:, [1, 2]]  # [Y, Z]

    # Find bounds
    y_min, y_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    z_min, z_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

    print(f"Bounds: Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")

    # Calculate center
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2

    # Calculate view bounds based on image size and resolution
    width, height = image_size
    half_width = (width * resolution) / 2
    half_height = (height * resolution) / 2

    view_y_min = center_y - half_width
    view_y_max = center_y + half_width
    view_z_min = center_z - half_height
    view_z_max = center_z + half_height

    print(f"View bounds: Y[{view_y_min:.2f}, {view_y_max:.2f}], Z[{view_z_min:.2f}, {view_z_max:.2f}]")
    print(f"Resolution: {resolution}m/pixel, Image size: {width}x{height}")

    # Create image (black background)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    pixel_counts = np.zeros((height, width), dtype=np.int32)
    color_accumulator = np.zeros((height, width, 3), dtype=np.float64)

    # Rasterize: map each point to a pixel
    print("Rasterizing points...")
    points_in_view = 0

    for i in range(len(points)):
        y, z = coords_2d[i]

        # Check if point is in view
        if not (view_y_min <= y <= view_y_max and view_z_min <= z <= view_z_max):
            continue

        points_in_view += 1

        # Map to pixel coordinates
        # Y maps to column (x), Z maps to row (y)
        px = int((y - view_y_min) / resolution)
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
    image = simple_rasterize(
        ply_path='ply_storage/fused.ply',
        output_path='/tmp/simple_rasterized.png',
        resolution=0.01,
        image_size=(1920, 1080)
    )