#!/usr/bin/env python3
"""
Test script to verify CUDA projection is working correctly.
"""

import sys
import json
import logging
from camera_projection_cupy import CameraProjectorCuPy, CUPY_AVAILABLE
from pose_manager import PoseManager
from yard_manager import YardManager, PLY_FILE_PATH
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_projection():
    """Test CUDA projection with a real camera."""

    # Check prerequisites
    logger.info("=== CUDA Projection Test ===")
    logger.info(f"1. Checking CuPy availability...")
    if not CUPY_AVAILABLE:
        logger.error("❌ CuPy not available!")
        return False
    logger.info("✓ CuPy is available")

    # Check PLY file
    logger.info(f"2. Checking point cloud file...")
    if not os.path.exists(PLY_FILE_PATH):
        logger.error(f"❌ PLY file not found: {PLY_FILE_PATH}")
        return False
    logger.info(f"✓ PLY file exists: {PLY_FILE_PATH}")

    # Check yard map
    logger.info(f"3. Checking yard map...")
    yard_manager = YardManager()
    used_map = yard_manager.get_used_map()
    if not used_map:
        logger.error("❌ No yard map is set as 'used'")
        return False
    logger.info(f"✓ Active map: {used_map['name']} ({used_map['resolution_x']}x{used_map['resolution_y']})")

    # Check camera poses
    logger.info(f"4. Checking camera poses...")
    pose_manager = PoseManager()
    all_poses = pose_manager.get_all_poses()
    if not all_poses:
        logger.error("❌ No camera poses found")
        return False
    logger.info(f"✓ Found {len(all_poses)} camera poses")

    # Select first camera for testing
    test_camera = all_poses[0]['camera_name']
    logger.info(f"5. Testing with camera: {test_camera}")

    pose = pose_manager.get_camera_pose(test_camera)
    if not pose:
        logger.error(f"❌ Could not load pose for {test_camera}")
        return False

    # Extract pose data
    extrinsics = pose['extrinsics']
    intrinsics = pose['intrinsics']
    camera_to_world = extrinsics['camera_to_world']

    # Transform to required format
    import numpy as np

    cam_pos_x = camera_to_world[0][3]
    cam_pos_y = camera_to_world[1][3]
    cam_pos_z = camera_to_world[2][3]

    rotation_matrix = np.array(extrinsics['rotation_matrix'])
    flip_180_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    corrected_rotation = flip_180_z @ rotation_matrix

    # Scale intrinsics to test resolution (using 2560x1920 as typical)
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

    # Normalize boundaries
    boundaries = used_map['boundaries']
    if 'x_min' in boundaries and 'min_x' not in boundaries:
        boundaries = {
            'min_x': boundaries['x_min'],
            'max_x': boundaries['x_max'],
            'min_y': boundaries['z_min'],
            'max_y': boundaries['z_max']
        }
        used_map['boundaries'] = boundaries

    logger.info(f"6. Creating CuPy projector...")
    try:
        projector = CameraProjectorCuPy(PLY_FILE_PATH)
        logger.info("✓ Projector created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create projector: {e}")
        return False

    logger.info(f"7. Running projection (this may take a few seconds)...")
    logger.info(f"   Camera: {camera_width}x{camera_height}, sample_rate=20")

    try:
        result = projector.project_camera_to_map(
            camera_pose=camera_pose_transformed,
            map_info=used_map,
            camera_width=camera_width,
            camera_height=camera_height,
            sample_rate=20
        )

        logger.info("✓ Projection completed!")
        logger.info("")
        logger.info("=== Results ===")
        logger.info(f"Mapped pixels: {result['pixel_count']}")
        logger.info(f"Coverage: {result['coverage_percent']}%")
        logger.info(f"Compute time: {result['compute_time']}s")
        logger.info(f"CUDA accelerated: {result.get('cuda_accelerated', False)}")
        logger.info(f"CuPy accelerated: {result.get('cupy_accelerated', False)}")

        if result.get('grid_build_time'):
            logger.info(f"Grid build time: {result['grid_build_time']}s")
        if result.get('ray_trace_time'):
            logger.info(f"Ray trace time: {result['ray_trace_time']}s")

        logger.info(f"Bounds: X=[{result['bounds']['min_x']}, {result['bounds']['max_x']}], "
                   f"Y=[{result['bounds']['min_y']}, {result['bounds']['max_y']}]")

        if result['pixel_count'] > 0:
            logger.info("")
            logger.info("✓✓✓ CUDA PROJECTION WORKING! ✓✓✓")
            return True
        else:
            logger.warning("⚠ Projection ran but no pixels were mapped")
            return False

    except Exception as e:
        logger.error(f"❌ Projection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_cuda_projection()
    sys.exit(0 if success else 1)
