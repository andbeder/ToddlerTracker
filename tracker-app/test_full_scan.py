#!/usr/bin/env python3
"""
Test CUDA projection with sample_rate=1 (all pixels).
This will process ~5 million rays for 2560x1920.
"""

import sys
import logging
from camera_projection_cupy import CameraProjectorCuPy, CUPY_AVAILABLE
from pose_manager import PoseManager
from yard_manager import YardManager, PLY_FILE_PATH
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_scan():
    """Test CUDA projection with all pixels."""

    if not CUPY_AVAILABLE:
        logger.error("❌ CuPy not available!")
        return False

    logger.info("=== Testing Full Pixel Scan (sample_rate=1) ===")

    # Get data
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

    # Normalize boundaries
    boundaries = used_map['boundaries']
    if 'x_min' in boundaries:
        boundaries = {
            'min_x': boundaries['x_min'],
            'max_x': boundaries['x_max'],
            'min_y': boundaries['z_min'],
            'max_y': boundaries['z_max']
        }
        used_map['boundaries'] = boundaries

    logger.info(f"Camera: {camera_width}x{camera_height}")
    logger.info(f"Total rays with sample_rate=1: {camera_width * camera_height:,}")

    # Create projector
    projector = CameraProjectorCuPy(PLY_FILE_PATH)

    logger.info("Running full scan projection...")
    try:
        result = projector.project_camera_to_map(
            camera_pose=camera_pose_transformed,
            map_info=used_map,
            camera_width=camera_width,
            camera_height=camera_height,
            sample_rate=1  # ALL PIXELS
        )

        logger.info("")
        logger.info("=== FULL SCAN RESULTS ===")
        logger.info(f"Mapped pixels: {result['pixel_count']:,}")
        logger.info(f"Coverage: {result['coverage_percent']}%")
        logger.info(f"Compute time: {result['compute_time']}s")
        logger.info(f"CUDA accelerated: {result.get('cuda_accelerated', False)}")

        if result.get('grid_build_time'):
            logger.info(f"Grid build: {result['grid_build_time']}s")
        if result.get('ray_trace_time'):
            logger.info(f"Ray trace: {result['ray_trace_time']}s")

        logger.info(f"Bounds: X=[{result['bounds']['min_x']}, {result['bounds']['max_x']}], "
                   f"Y=[{result['bounds']['min_y']}, {result['bounds']['max_y']}]")

        if result.get('cuda_accelerated'):
            logger.info("")
            logger.info("✓✓✓ FULL SCAN WITH CUDA WORKING! ✓✓✓")
            return True
        else:
            logger.warning("⚠ Fell back to CPU")
            return False

    except Exception as e:
        logger.error(f"❌ Full scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_full_scan()
    sys.exit(0 if success else 1)
