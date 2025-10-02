#!/usr/bin/env python3
"""
Analyze the projection pattern to detect banding or gaps.
"""

import sys
import logging
import numpy as np
from camera_projection_cupy import CameraProjectorCuPy
from pose_manager import PoseManager
from yard_manager import YardManager, PLY_FILE_PATH

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_pattern():
    """Analyze projection pattern for gaps/banding."""

    yard_manager = YardManager()
    used_map = yard_manager.get_used_map()

    pose_manager = PoseManager()
    pose = pose_manager.get_camera_pose('backyard')

    # Prepare camera pose
    extrinsics = pose['extrinsics']
    intrinsics = pose['intrinsics']
    camera_to_world = extrinsics['camera_to_world']

    cam_pos_x = camera_to_world[0][3]
    cam_pos_y = camera_to_world[1][3]
    cam_pos_z = camera_to_world[2][3]

    rotation_matrix = np.array(extrinsics['rotation_matrix'])
    flip_180_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    corrected_rotation = flip_180_z @ rotation_matrix

    camera_width = 2560
    camera_height = 1920
    colmap_width = intrinsics['width']
    colmap_height = intrinsics['height']
    scale_x = camera_width / colmap_width
    scale_y = camera_height / colmap_height

    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics['fx'] = intrinsics['fx'] * scale_x
    scaled_intrinsics['fy'] = intrinsics['fy'] * scale_y
    scaled_intrinsics['cx'] = intrinsics['cx'] * scale_x
    scaled_intrinsics['cy'] = intrinsics['cy'] * scale_y

    camera_pose_transformed = {
        'position_x': cam_pos_x,
        'position_y': cam_pos_y,
        'position_z': cam_pos_z,
        'rotation_matrix': corrected_rotation.tolist(),
        'intrinsics': scaled_intrinsics
    }

    boundaries = used_map['boundaries']
    if 'x_min' in boundaries:
        boundaries = {
            'min_x': boundaries['x_min'],
            'max_x': boundaries['x_max'],
            'min_y': boundaries['z_min'],
            'max_y': boundaries['z_max']
        }
        used_map['boundaries'] = boundaries

    # Run projection
    projector = CameraProjectorCuPy(PLY_FILE_PATH)

    logger.info("Running projection with sample_rate=1...")
    result = projector.project_camera_to_map(
        camera_pose=camera_pose_transformed,
        map_info=used_map,
        camera_width=camera_width,
        camera_height=camera_height,
        sample_rate=1
    )

    # Analyze the pixel_mappings
    pixel_mappings = result['pixel_mappings']
    logger.info(f"\nTotal camera pixels mapped: {len(pixel_mappings):,}")

    # Check camera pixel coverage
    cam_x_set = set()
    cam_y_set = set()
    map_pixel_set = set()

    for cam_x, cam_y, map_x, map_y in pixel_mappings:
        cam_x_set.add(cam_x)
        cam_y_set.add(cam_y)
        map_pixel_set.add((map_x, map_y))

    logger.info(f"\nCamera pixel analysis:")
    logger.info(f"  Unique camera X coordinates: {len(cam_x_set)} (expected: {camera_width})")
    logger.info(f"  Unique camera Y coordinates: {len(cam_y_set)} (expected: {camera_height})")
    logger.info(f"  Unique map pixels covered: {len(map_pixel_set)}")

    # Check for gaps in camera rows
    cam_y_list = sorted(cam_y_set)
    gaps = []
    for i in range(len(cam_y_list) - 1):
        if cam_y_list[i+1] - cam_y_list[i] > 1:
            gaps.append((cam_y_list[i], cam_y_list[i+1]))

    if gaps:
        logger.warning(f"\n⚠️  Found {len(gaps)} gaps in camera Y coordinates:")
        for start, end in gaps[:10]:  # Show first 10
            logger.warning(f"    Gap between Y={start} and Y={end} (missing {end-start-1} rows)")
    else:
        logger.info("\n✓ No gaps in camera Y coordinates - all rows scanned!")

    # Check camera X gaps
    cam_x_list = sorted(cam_x_set)
    x_gaps = []
    for i in range(len(cam_x_list) - 1):
        if cam_x_list[i+1] - cam_x_list[i] > 1:
            x_gaps.append((cam_x_list[i], cam_x_list[i+1]))

    if x_gaps:
        logger.warning(f"\n⚠️  Found {len(x_gaps)} gaps in camera X coordinates:")
        for start, end in x_gaps[:10]:
            logger.warning(f"    Gap between X={start} and X={end} (missing {end-start-1} columns)")
    else:
        logger.info("✓ No gaps in camera X coordinates - all columns scanned!")

    # Create density map
    logger.info(f"\nMap pixel density analysis:")
    density_map = {}
    for cam_x, cam_y, map_x, map_y in pixel_mappings:
        key = (map_x, map_y)
        density_map[key] = density_map.get(key, 0) + 1

    densities = list(density_map.values())
    logger.info(f"  Min camera pixels per map pixel: {min(densities)}")
    logger.info(f"  Max camera pixels per map pixel: {max(densities)}")
    logger.info(f"  Avg camera pixels per map pixel: {np.mean(densities):.1f}")
    logger.info(f"  Median camera pixels per map pixel: {np.median(densities):.1f}")

    # Show distribution
    hist, bins = np.histogram(densities, bins=[1, 10, 100, 1000, 10000, 100000])
    logger.info(f"\nDensity distribution:")
    logger.info(f"  1-10 camera pixels: {hist[0]} map pixels")
    logger.info(f"  10-100 camera pixels: {hist[1]} map pixels")
    logger.info(f"  100-1000 camera pixels: {hist[2]} map pixels")
    logger.info(f"  1000-10000 camera pixels: {hist[3]} map pixels")
    logger.info(f"  >10000 camera pixels: {hist[4]} map pixels")

if __name__ == '__main__':
    analyze_pattern()
