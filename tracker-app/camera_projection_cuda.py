"""
CUDA-Accelerated Camera-to-Map Projection Module

Provides massive speedup over CPU ray-tracing using parallel GPU processing.
Expected performance: 95s -> ~2-5s (20-50x speedup)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

try:
    from numba import cuda, float32, int32
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None

logger = logging.getLogger(__name__)


@cuda.jit
def build_ground_grid_kernel(points, point_count,
                             min_x, max_x, min_z, max_z,
                             resolution_x, resolution_y,
                             grid_indices, grid_y_values, grid_counts):
    """
    CUDA kernel to build spatial grid of ground points.
    Each thread processes one point and assigns it to a grid cell.
    """
    idx = cuda.grid(1)

    if idx >= point_count:
        return

    x = points[idx, 0]
    y = points[idx, 1]
    z = points[idx, 2]

    # Check if point is within map boundaries
    if x < min_x or x > max_x or z < min_z or z > max_z:
        return

    # Calculate grid cell
    cell_size_x = (max_x - min_x) / resolution_x
    cell_size_z = (max_z - min_z) / resolution_y

    cell_x = int((x - min_x) / cell_size_x)
    cell_z = int((z - min_z) / cell_size_z)

    # Clamp to grid boundaries
    if cell_x < 0: cell_x = 0
    if cell_x >= resolution_x: cell_x = resolution_x - 1
    if cell_z < 0: cell_z = 0
    if cell_z >= resolution_y: cell_z = resolution_y - 1

    # Linear index for 2D grid
    grid_idx = cell_z * resolution_x + cell_x

    # Atomic operations to update grid
    # Store index of point in this cell (limit to 10 points per cell)
    old_count = cuda.atomic.add(grid_counts, grid_idx, 1)
    if old_count < 10:  # Limit points per cell for memory
        grid_indices[grid_idx * 10 + old_count] = idx
        grid_y_values[grid_idx * 10 + old_count] = y


@cuda.jit
def ray_trace_kernel(rays_origin, rays_direction, ray_count,
                     points, point_count,
                     grid_indices, grid_y_values, grid_counts,
                     min_x, max_x, min_z, max_z,
                     resolution_x, resolution_y,
                     max_distance, step_size,
                     output_world_pos, output_valid):
    """
    CUDA kernel for parallel ray tracing.
    Each thread traces one ray through the point cloud.
    """
    ray_idx = cuda.grid(1)

    if ray_idx >= ray_count:
        return

    # Get ray parameters
    origin_x = rays_origin[ray_idx, 0]
    origin_y = rays_origin[ray_idx, 1]
    origin_z = rays_origin[ray_idx, 2]

    dir_x = rays_direction[ray_idx, 0]
    dir_y = rays_direction[ray_idx, 1]
    dir_z = rays_direction[ray_idx, 2]

    cell_size_x = (max_x - min_x) / resolution_x
    cell_size_z = (max_z - min_z) / resolution_y

    # March along ray
    distance = 0.0
    while distance < max_distance:
        # Current point on ray
        point_x = origin_x + dir_x * distance
        point_y = origin_y + dir_y * distance
        point_z = origin_z + dir_z * distance

        # Check if within map bounds
        if point_x < min_x or point_x > max_x or point_z < min_z or point_z > max_z:
            distance += step_size
            continue

        # Get grid cell
        cell_x = int((point_x - min_x) / cell_size_x)
        cell_z = int((point_z - min_z) / cell_size_z)

        # Clamp
        if cell_x < 0: cell_x = 0
        if cell_x >= resolution_x: cell_x = resolution_x - 1
        if cell_z < 0: cell_z = 0
        if cell_z >= resolution_y: cell_z = resolution_y - 1

        grid_idx = cell_z * resolution_x + cell_x
        count = grid_counts[grid_idx]

        if count > 0:
            # Check ground points in this cell (bottom 40%)
            min_y = 1e10
            for i in range(min(count, 10)):
                y_val = grid_y_values[grid_idx * 10 + i]
                if y_val < min_y:
                    min_y = y_val

            # Calculate 40th percentile threshold
            threshold = min_y + 0.4 * (point_y - min_y)

            # Check if ray is near ground
            if point_y <= threshold + 0.2:  # Within 20cm of ground
                output_world_pos[ray_idx, 0] = point_x
                output_world_pos[ray_idx, 1] = point_y
                output_world_pos[ray_idx, 2] = point_z
                output_valid[ray_idx] = 1
                return

        distance += step_size

    # No intersection found
    output_valid[ray_idx] = 0


class CameraProjectorCUDA:
    """CUDA-accelerated camera pixel to map projection."""

    def __init__(self, point_cloud_path: str):
        """
        Initialize projector with point cloud data.

        Args:
            point_cloud_path: Path to PLY file
        """
        self.point_cloud_path = point_cloud_path
        self.points = None
        self.colors = None
        self.cuda_available = CUDA_AVAILABLE

        if not self.cuda_available:
            logger.warning("CUDA not available, will fall back to CPU")

        self._load_point_cloud()

    def _load_point_cloud(self):
        """Load point cloud from PLY file."""
        try:
            import trimesh
            logger.info(f"Loading point cloud from {self.point_cloud_path}")
            start = time.time()
            mesh = trimesh.load(self.point_cloud_path)
            self.points = np.ascontiguousarray(mesh.vertices.astype(np.float32))
            if hasattr(mesh.visual, 'vertex_colors'):
                self.colors = mesh.visual.vertex_colors[:, :3]
            load_time = time.time() - start
            logger.info(f"Loaded {len(self.points):,} points in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Error loading point cloud: {e}")
            raise

    def project_camera_to_map(
        self,
        camera_pose: Dict,
        map_info: Dict,
        camera_width: int,
        camera_height: int,
        sample_rate: int = 10
    ) -> Dict:
        """
        Project camera pixels onto map using CUDA-accelerated ray tracing.

        Args:
            camera_pose: Camera position and orientation with intrinsics
            map_info: Map boundaries, center, resolution
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels
            sample_rate: Sample every Nth pixel

        Returns:
            Dict with projection results
        """
        if not self.cuda_available:
            logger.warning("CUDA not available, using CPU fallback")
            from camera_projection import CameraProjector
            cpu_projector = CameraProjector(self.point_cloud_path)
            return cpu_projector.project_camera_to_map(
                camera_pose, map_info, camera_width, camera_height, sample_rate
            )

        start_time = time.time()

        logger.info(f"Starting CUDA-accelerated projection: {camera_width}x{camera_height}, sample_rate={sample_rate}")

        # Extract camera parameters
        camera_pos = np.array([
            camera_pose['position_x'],
            camera_pose['position_y'],
            camera_pose['position_z']
        ], dtype=np.float32)

        rotation_matrix = np.array(camera_pose['rotation_matrix'], dtype=np.float32).reshape(3, 3)

        # Get intrinsics
        intrinsics = camera_pose.get('intrinsics', {})
        if intrinsics and 'fx' in intrinsics:
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']
            logger.info(f"Using calibrated intrinsics: fx={fx:.2f}, fy={fy:.2f}, fov={intrinsics.get('fov_x', 0):.1f}°")
        else:
            fx = fy = camera_width * 1.2
            cx = camera_width / 2
            cy = camera_height / 2
            logger.warning("No intrinsics, using approximation")

        # Generate sampled rays
        logger.info("Generating camera rays...")
        rays_origin_list = []
        rays_direction_list = []
        ray_pixels = []  # Store (cam_x, cam_y) for each ray

        for cam_y in range(0, camera_height, sample_rate):
            for cam_x in range(0, camera_width, sample_rate):
                # Ray direction in camera space
                ray_dir_cam = np.array([
                    (cam_x - cx) / fx,
                    (cam_y - cy) / fy,
                    1.0
                ], dtype=np.float32)
                ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

                # Transform to world space
                ray_dir_world = rotation_matrix @ ray_dir_cam

                rays_origin_list.append(camera_pos)
                rays_direction_list.append(ray_dir_world)
                ray_pixels.append((cam_x, cam_y))

        rays_origin = np.array(rays_origin_list, dtype=np.float32)
        rays_direction = np.array(rays_direction_list, dtype=np.float32)
        ray_count = len(rays_origin)

        logger.info(f"Generated {ray_count:,} rays")

        # Build ground grid on GPU
        logger.info("Building spatial grid on GPU...")
        grid_start = time.time()

        boundaries = map_info['boundaries']
        min_x, max_x = float(boundaries['min_x']), float(boundaries['max_x'])
        min_z, max_z = float(boundaries['min_y']), float(boundaries['max_y'])
        resolution_x = int(map_info['resolution_x'])
        resolution_y = int(map_info['resolution_y'])

        grid_size = resolution_x * resolution_y
        max_points_per_cell = 10  # Reduced from 100 to save memory

        logger.info(f"Grid: {resolution_x}x{resolution_y} = {grid_size:,} cells, {max_points_per_cell} pts/cell")
        logger.info(f"Memory: {grid_size * max_points_per_cell * 8 / 1024 / 1024:.1f} MB for grid arrays")

        try:
            # Allocate grid arrays on GPU
            d_points = cuda.to_device(self.points)
            d_grid_indices = cuda.device_array((grid_size * max_points_per_cell,), dtype=np.int32)
            d_grid_y_values = cuda.device_array((grid_size * max_points_per_cell,), dtype=np.float32)
            d_grid_counts = cuda.to_device(np.zeros(grid_size, dtype=np.int32))
        except Exception as e:
            logger.error(f"Failed to allocate GPU memory: {e}")
            raise

        # Launch grid building kernel
        threads_per_block = 256
        blocks = (len(self.points) + threads_per_block - 1) // threads_per_block

        try:
            # Close any existing CUDA context and create fresh one
            cuda.close()
            cuda.select_device(0)

            build_ground_grid_kernel[blocks, threads_per_block](
                d_points, len(self.points),
                min_x, max_x, min_z, max_z,
                resolution_x, resolution_y,
                d_grid_indices, d_grid_y_values, d_grid_counts
            )
            cuda.synchronize()
        except Exception as e:
            logger.error(f"CUDA grid building failed: {e}")
            logger.info("Falling back to CPU projection")
            from camera_projection import CameraProjector
            cpu_projector = CameraProjector(self.point_cloud_path)
            return cpu_projector.project_camera_to_map(
                camera_pose, map_info, camera_width, camera_height, sample_rate
            )

        grid_time = time.time() - grid_start
        logger.info(f"Grid built in {grid_time:.2f}s")

        # Ray trace on GPU
        logger.info("Ray tracing on GPU...")
        trace_start = time.time()

        d_rays_origin = cuda.to_device(rays_origin)
        d_rays_direction = cuda.to_device(rays_direction)
        d_output_world_pos = cuda.device_array((ray_count, 3), dtype=np.float32)
        d_output_valid = cuda.device_array(ray_count, dtype=np.int32)

        # Launch ray tracing kernel
        blocks = (ray_count + threads_per_block - 1) // threads_per_block
        ray_trace_kernel[blocks, threads_per_block](
            d_rays_origin, d_rays_direction, ray_count,
            d_points, len(self.points),
            d_grid_indices, d_grid_y_values, d_grid_counts,
            min_x, max_x, min_z, max_z,
            resolution_x, resolution_y,
            50.0,  # max_distance
            0.5,   # step_size
            d_output_world_pos, d_output_valid
        )
        cuda.synchronize()

        trace_time = time.time() - trace_start
        logger.info(f"Ray tracing completed in {trace_time:.2f}s")

        # Copy results back to CPU
        output_world_pos = d_output_world_pos.copy_to_host()
        output_valid = d_output_valid.copy_to_host()

        # Process results
        logger.info("Processing results...")
        pixel_mappings = []
        projected_pixels = []

        for i in range(ray_count):
            if output_valid[i] == 1:
                world_pos = output_world_pos[i]
                cam_x, cam_y = ray_pixels[i]

                # Convert world to map pixel
                pixel_x = int((world_pos[0] - min_x) / (max_x - min_x) * resolution_x)
                pixel_y = int((world_pos[2] - min_z) / (max_z - min_z) * resolution_y)

                # Clamp
                pixel_x = max(0, min(pixel_x, resolution_x - 1))
                pixel_y = max(0, min(pixel_y, resolution_y - 1))

                pixel_mappings.append((cam_x, cam_y, pixel_x, pixel_y))
                projected_pixels.append([pixel_x, pixel_y])

        # Compute bounds and coverage
        if projected_pixels:
            projected_arr = np.array(projected_pixels)
            bounds = {
                'min_x': int(projected_arr[:, 0].min()),
                'max_x': int(projected_arr[:, 0].max()),
                'min_y': int(projected_arr[:, 1].min()),
                'max_y': int(projected_arr[:, 1].max())
            }
            map_total_pixels = resolution_x * resolution_y
            coverage_percent = (len(projected_pixels) / map_total_pixels) * 100
        else:
            bounds = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
            coverage_percent = 0.0

        compute_time = time.time() - start_time

        logger.info(f"✓ CUDA projection complete: {len(pixel_mappings):,} pixels mapped in {compute_time:.2f}s")
        logger.info(f"  Grid build: {grid_time:.2f}s, Ray trace: {trace_time:.2f}s")
        logger.info(f"  Coverage: {coverage_percent:.2f}%")

        return {
            'pixel_mappings': pixel_mappings,
            'projected_pixels': projected_pixels,
            'bounds': bounds,
            'pixel_count': len(pixel_mappings),
            'coverage_percent': round(coverage_percent, 2),
            'compute_time': round(compute_time, 2),
            'cuda_accelerated': True,
            'grid_build_time': round(grid_time, 2),
            'ray_trace_time': round(trace_time, 2)
        }
