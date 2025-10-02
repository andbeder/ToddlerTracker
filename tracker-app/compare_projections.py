#!/usr/bin/env python3
"""
Standalone script to compare CPU and CUDA projection outputs.
Runs both versions with identical inputs and reports differences.
"""

import numpy as np
import json
import sqlite3
import sys
import os
import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_projection import CameraProjector
from camera_projection_cupy import CameraProjectorCuPy


def load_camera_pose(camera_name):
    """Load camera pose from database."""
    conn = sqlite3.connect('poses.db')
    cursor = conn.execute(
        'SELECT intrinsics, extrinsics FROM camera_poses WHERE camera_name = ?',
        (camera_name,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"Camera {camera_name} not found in database")

    intrinsics = json.loads(row[0])
    extrinsics = json.loads(row[1])

    return intrinsics, extrinsics


def load_yard_map():
    """Load the active yard map."""
    conn = sqlite3.connect('yard.db')
    cursor = conn.execute(
        'SELECT id, name, boundaries, center_x, center_z, width, height, rotation, resolution_x, resolution_y FROM yard_maps WHERE is_used = 1'
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError("No active yard map found")

    # Parse the row (id, name, boundaries, center_x, center_z, width, height, rotation, resolution_x, resolution_y)
    yard_map = {
        'id': row[0],
        'name': row[1],
        'boundaries': json.loads(row[2]) if row[2] else {},
        'center_x': row[3],
        'center_z': row[4],
        'width': row[5],
        'height': row[6],
        'rotation': row[7],
        'resolution_x': row[8],
        'resolution_y': row[9]
    }

    return yard_map


def build_camera_pose(intrinsics, extrinsics, camera_width, camera_height):
    """Build camera pose dict with scaled intrinsics and orientation correction."""
    # Scale intrinsics
    colmap_width = intrinsics['width']
    colmap_height = intrinsics['height']
    scale_x = camera_width / colmap_width
    scale_y = camera_height / colmap_height

    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics['fx'] = intrinsics['fx'] * scale_x
    scaled_intrinsics['fy'] = intrinsics['fy'] * scale_y
    scaled_intrinsics['cx'] = intrinsics['cx'] * scale_x
    scaled_intrinsics['cy'] = intrinsics['cy'] * scale_y
    scaled_intrinsics['width'] = camera_width
    scaled_intrinsics['height'] = camera_height

    # Extract camera position
    camera_to_world = extrinsics['camera_to_world']
    cam_pos_x = camera_to_world[0][3]
    cam_pos_y = camera_to_world[1][3]
    cam_pos_z = camera_to_world[2][3]

    # Apply 180° rotation correction
    rotation_matrix = np.array(extrinsics['rotation_matrix'])
    flip_180_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    corrected_rotation = flip_180_z @ rotation_matrix

    return {
        'position_x': cam_pos_x,
        'position_y': cam_pos_y,
        'position_z': cam_pos_z,
        'rotation_matrix': corrected_rotation.tolist(),
        'quaternion': extrinsics['quaternion'],
        'intrinsics': scaled_intrinsics
    }


def compare_projections(camera_name='garage', camera_width=2560, camera_height=1920, sample_rate=20):
    """Run both CPU and CUDA projections and compare results."""
    print(f"\n{'='*80}")
    print(f"PROJECTION COMPARISON: {camera_name}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading camera pose and yard map...")
    intrinsics, extrinsics = load_camera_pose(camera_name)
    yard_map = load_yard_map()

    # Normalize boundaries format
    boundaries = yard_map['boundaries']
    if 'x_min' in boundaries and 'min_x' not in boundaries:
        boundaries = {
            'min_x': boundaries['x_min'],
            'max_x': boundaries['x_max'],
            'min_y': boundaries['z_min'],
            'max_y': boundaries['z_max']
        }
        yard_map['boundaries'] = boundaries

    print(f"  Camera: {camera_name}")
    print(f"  Resolution: {camera_width}x{camera_height}")
    print(f"  Map: {yard_map['name']} ({yard_map['resolution_x']}x{yard_map['resolution_y']})")
    print(f"  Sample rate: {sample_rate}\n")

    # Build camera pose
    camera_pose = build_camera_pose(intrinsics, extrinsics, camera_width, camera_height)

    print(f"Camera position: ({camera_pose['position_x']:.2f}, {camera_pose['position_y']:.2f}, {camera_pose['position_z']:.2f})")
    print(f"Intrinsics: fx={camera_pose['intrinsics']['fx']:.1f}, fy={camera_pose['intrinsics']['fy']:.1f}\n")

    # Point cloud path
    ply_path = '/home/andrew/toddler-tracker/tracker-app/ply_storage/fused.ply'
    if not os.path.exists(ply_path):
        print(f"ERROR: Point cloud not found at {ply_path}")
        return

    # Run CPU projection
    print(f"\n{'='*80}")
    print("RUNNING CPU PROJECTION")
    print(f"{'='*80}\n")

    cpu_projector = CameraProjector(ply_path)
    cpu_result = cpu_projector.project_camera_to_map(
        camera_pose=camera_pose,
        map_info=yard_map,
        camera_width=camera_width,
        camera_height=camera_height,
        sample_rate=sample_rate
    )

    print(f"\nCPU Results:")
    print(f"  Pixels mapped: {cpu_result['pixel_count']}")
    print(f"  Coverage: {cpu_result['coverage_percent']}%")
    print(f"  Bounds: {cpu_result['bounds']}")
    print(f"  Compute time: {cpu_result['compute_time']}s")

    if cpu_result['projected_pixels']:
        cpu_pixels = np.array(cpu_result['projected_pixels'])
        print(f"  First 5 pixels: {cpu_pixels[:5].tolist()}")
        print(f"  Mean position: ({cpu_pixels[:, 0].mean():.1f}, {cpu_pixels[:, 1].mean():.1f})")
        print(f"  Std dev: ({cpu_pixels[:, 0].std():.1f}, {cpu_pixels[:, 1].std():.1f})")

    # Run CUDA projection
    print(f"\n{'='*80}")
    print("RUNNING CUDA PROJECTION")
    print(f"{'='*80}\n")

    try:
        cuda_projector = CameraProjectorCuPy(ply_path)
        cuda_result = cuda_projector.project_camera_to_map(
            camera_pose=camera_pose,
            map_info=yard_map,
            camera_width=camera_width,
            camera_height=camera_height,
            sample_rate=sample_rate
        )

        print(f"\nCUDA Results:")
        print(f"  Pixels mapped: {cuda_result['pixel_count']}")
        print(f"  Coverage: {cuda_result['coverage_percent']}%")
        print(f"  Bounds: {cuda_result['bounds']}")
        print(f"  Compute time: {cuda_result['compute_time']}s")

        if cuda_result['projected_pixels']:
            cuda_pixels = np.array(cuda_result['projected_pixels'])
            print(f"  First 5 pixels: {cuda_pixels[:5].tolist()}")
            print(f"  Mean position: ({cuda_pixels[:, 0].mean():.1f}, {cuda_pixels[:, 1].mean():.1f})")
            print(f"  Std dev: ({cuda_pixels[:, 0].std():.1f}, {cuda_pixels[:, 1].std():.1f})")

        # Compare results
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}\n")

        print(f"Pixel count difference: {cuda_result['pixel_count'] - cpu_result['pixel_count']}")
        print(f"Coverage difference: {cuda_result['coverage_percent'] - cpu_result['coverage_percent']:.2f}%")
        print(f"Speedup: {cpu_result['compute_time'] / cuda_result['compute_time']:.1f}x\n")

        if cpu_result['projected_pixels'] and cuda_result['projected_pixels']:
            cpu_pixels = np.array(cpu_result['projected_pixels'])
            cuda_pixels = np.array(cuda_result['projected_pixels'])

            # Compare bounds
            print("Bounds comparison:")
            print(f"  CPU:  min=({cpu_result['bounds']['min_x']}, {cpu_result['bounds']['min_y']}), max=({cpu_result['bounds']['max_x']}, {cpu_result['bounds']['max_y']})")
            print(f"  CUDA: min=({cuda_result['bounds']['min_x']}, {cuda_result['bounds']['min_y']}), max=({cuda_result['bounds']['max_x']}, {cuda_result['bounds']['max_y']})")

            # Check if bounds are identical (all pixels at same location)
            cpu_same = (cpu_result['bounds']['min_x'] == cpu_result['bounds']['max_x'] and
                       cpu_result['bounds']['min_y'] == cpu_result['bounds']['max_y'])
            cuda_same = (cuda_result['bounds']['min_x'] == cuda_result['bounds']['max_x'] and
                        cuda_result['bounds']['min_y'] == cuda_result['bounds']['max_y'])

            print(f"\n  CPU pixels all at same location: {cpu_same}")
            print(f"  CUDA pixels all at same location: {cuda_same}")

            if cuda_same:
                print("\n  ⚠️  WARNING: CUDA pixels are all converging to a single point!")
                print("  This indicates a coordinate transformation bug in the CUDA version.")

            # Compare means
            cpu_mean = cpu_pixels.mean(axis=0)
            cuda_mean = cuda_pixels.mean(axis=0)
            mean_diff = np.linalg.norm(cuda_mean - cpu_mean)

            print(f"\nMean position difference: {mean_diff:.1f} pixels")
            print(f"  CPU mean:  ({cpu_mean[0]:.1f}, {cpu_mean[1]:.1f})")
            print(f"  CUDA mean: ({cuda_mean[0]:.1f}, {cuda_mean[1]:.1f})")

            if mean_diff > 10:
                print(f"  ⚠️  Large difference detected (>{mean_diff:.1f} pixels)")

    except ImportError as e:
        print(f"CUDA projection not available: {e}")
        print("Skipping CUDA comparison.")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare CPU and CUDA projections')
    parser.add_argument('--camera', default='garage', help='Camera name (default: garage)')
    parser.add_argument('--width', type=int, default=2560, help='Camera width (default: 2560)')
    parser.add_argument('--height', type=int, default=1920, help='Camera height (default: 1920)')
    parser.add_argument('--sample-rate', type=int, default=20, help='Sample rate (default: 20)')

    args = parser.parse_args()

    compare_projections(
        camera_name=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        sample_rate=args.sample_rate
    )
