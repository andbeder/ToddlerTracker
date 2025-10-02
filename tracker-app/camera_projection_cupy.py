"""
CuPy-based CUDA-Accelerated Camera-to-Map Projection Module

Uses CuPy for more reliable CUDA context management than Numba.
Expected performance: 95s -> ~2-5s (20-50x speedup)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class CameraProjectorCuPy:
    """CuPy-accelerated camera pixel to map projection."""

    def __init__(self, point_cloud_path: str):
        """
        Initialize projector with point cloud data.

        Args:
            point_cloud_path: Path to PLY file
        """
        self.point_cloud_path = point_cloud_path
        self.points = None
        self.colors = None
        self.cupy_available = CUPY_AVAILABLE

        if not self.cupy_available:
            logger.warning("CuPy not available, will fall back to CPU")

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

    def _build_ground_grid_cupy(self, points_cpu, min_x, max_x, min_z, max_z,
                                resolution_x, resolution_y):
        """
        Build spatial grid on GPU using CuPy.
        Returns grid with ground point Y-values for each cell.
        OPTIMIZED: Filter points on CPU before GPU transfer to save memory.
        """
        # Filter points within map boundaries ON CPU to reduce GPU memory usage
        logger.info(f"Pre-filtering {len(points_cpu):,} points on CPU...")
        x_coords = points_cpu[:, 0]
        z_coords = points_cpu[:, 2]

        mask = (x_coords >= min_x) & (x_coords <= max_x) & \
               (z_coords >= min_z) & (z_coords <= max_z)

        filtered_points_cpu = points_cpu[mask]
        logger.info(f"Filtered to {len(filtered_points_cpu):,} points within map bounds")

        # GPU memory tracking for cleanup
        gpu_arrays = []

        try:
            # Transfer only filtered points to GPU
            filtered_points = cp.array(filtered_points_cpu)
            gpu_arrays.append(filtered_points)

            # Calculate cell indices for each point
            cell_size_x = (max_x - min_x) / resolution_x
            cell_size_z = (max_z - min_z) / resolution_y

            cell_x = cp.clip(((filtered_points[:, 0] - min_x) / cell_size_x).astype(cp.int32),
                            0, resolution_x - 1)
            gpu_arrays.append(cell_x)

            cell_z = cp.clip(((filtered_points[:, 2] - min_z) / cell_size_z).astype(cp.int32),
                            0, resolution_y - 1)
            gpu_arrays.append(cell_z)

            # Create linear cell indices
            cell_indices = cell_z * resolution_x + cell_x
            gpu_arrays.append(cell_indices)

            # For each cell, store ground points (bottom 40% of Y values)
            grid_size = resolution_x * resolution_y
            grid_min_y = cp.full(grid_size, cp.inf, dtype=cp.float32)
            grid_max_y = cp.full(grid_size, -cp.inf, dtype=cp.float32)
            grid_count = cp.zeros(grid_size, dtype=cp.int32)

            # Use CuPy RawKernel for atomic min/max operations
            build_grid_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void build_grid(const float* y_vals, const int* cell_indices, int n_points,
                           float* grid_min_y, float* grid_max_y) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= n_points) return;

                int cell_id = cell_indices[idx];
                float y = y_vals[idx];

                atomicMin((int*)&grid_min_y[cell_id], __float_as_int(y));
                atomicMax((int*)&grid_max_y[cell_id], __float_as_int(y));
            }
            ''', 'build_grid')

            y_vals = filtered_points[:, 1]
            gpu_arrays.append(y_vals)

            # Launch kernel
            threads_per_block = 256
            blocks = (len(filtered_points) + threads_per_block - 1) // threads_per_block
            build_grid_kernel((blocks,), (threads_per_block,),
                             (y_vals, cell_indices, len(filtered_points),
                              grid_min_y, grid_max_y))

            # Count points per cell using bincount
            counts = cp.bincount(cell_indices, minlength=grid_size)
            grid_count[:len(counts)] = counts

            # Calculate 40th percentile threshold for each cell
            # threshold = min_y + 0.4 * (max_y - min_y)
            grid_threshold = grid_min_y + 0.4 * (grid_max_y - grid_min_y)

            return grid_threshold, grid_count

        finally:
            # Free intermediate GPU memory - ALWAYS executes even on exceptions
            for arr in gpu_arrays:
                try:
                    del arr
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete GPU array: {cleanup_err}")

            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as cleanup_err:
                logger.warning(f"Failed to free GPU memory pool: {cleanup_err}")

    def _ray_trace_cupy(self, rays_origin, rays_direction,
                       grid_threshold, grid_count,
                       min_x, max_x, min_z, max_z,
                       resolution_x, resolution_y,
                       max_distance=50.0, step_size=0.5):
        """
        Ray trace using CuPy parallel operations.
        """
        ray_count = len(rays_origin)

        # Allocate output arrays
        output_valid = cp.zeros(ray_count, dtype=cp.bool_)
        output_world_x = cp.zeros(ray_count, dtype=cp.float32)
        output_world_y = cp.zeros(ray_count, dtype=cp.float32)
        output_world_z = cp.zeros(ray_count, dtype=cp.float32)

        cell_size_x = (max_x - min_x) / resolution_x
        cell_size_z = (max_z - min_z) / resolution_y

        # Ray march for each ray
        num_steps = int(max_distance / step_size)

        for step in range(1, num_steps):  # Start at 1 to skip camera origin (distance=0)
            distance = step * step_size

            # Current points on all rays
            points_x = rays_origin[:, 0] + rays_direction[:, 0] * distance
            points_y = rays_origin[:, 1] + rays_direction[:, 1] * distance
            points_z = rays_origin[:, 2] + rays_direction[:, 2] * distance

            # Check which rays are still within bounds and not yet found
            in_bounds = (points_x >= min_x) & (points_x <= max_x) & \
                       (points_z >= min_z) & (points_z <= max_z) & \
                       (~output_valid)

            if not cp.any(in_bounds):
                break

            # Get cell indices for in-bounds rays
            cell_x = cp.clip(((points_x - min_x) / cell_size_x).astype(cp.int32),
                            0, resolution_x - 1)
            cell_z = cp.clip(((points_z - min_z) / cell_size_z).astype(cp.int32),
                            0, resolution_y - 1)
            cell_idx = cell_z * resolution_x + cell_x

            # Check if ray is near ground in this cell
            has_ground = grid_count[cell_idx] > 0
            threshold = grid_threshold[cell_idx]
            near_ground = points_y <= (threshold + 0.2)  # Within 20cm

            # Mark rays that hit ground
            hit = in_bounds & has_ground & near_ground

            # Update outputs for hits
            output_valid[hit] = True
            output_world_x[hit] = points_x[hit]
            output_world_y[hit] = points_y[hit]
            output_world_z[hit] = points_z[hit]

        return output_valid, output_world_x, output_world_y, output_world_z

    def project_camera_to_map(
        self,
        camera_pose: Dict,
        map_info: Dict,
        camera_width: int,
        camera_height: int,
        sample_rate: int = 10
    ) -> Dict:
        """
        Project camera pixels onto map using CuPy-accelerated processing.

        Args:
            camera_pose: Camera position and orientation with intrinsics
            map_info: Map boundaries, center, resolution
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels
            sample_rate: Sample every Nth pixel

        Returns:
            Dict with projection results
        """
        if not self.cupy_available:
            logger.warning("CuPy not available, using CPU fallback")
            from camera_projection import CameraProjector
            cpu_projector = CameraProjector(self.point_cloud_path)
            return cpu_projector.project_camera_to_map(
                camera_pose, map_info, camera_width, camera_height, sample_rate
            )

        start_time = time.time()

        logger.info(f"Starting CuPy-accelerated projection: {camera_width}x{camera_height}, sample_rate={sample_rate}")

        # Check GPU memory availability
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            total_mem = device.mem_info[1]  # Total GPU memory
            free_mem = device.mem_info[0]   # Free GPU memory
            logger.info(f"GPU Memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
        except Exception as e:
            logger.warning(f"Could not query GPU memory: {e}")

        # Track all GPU arrays for cleanup
        gpu_allocations = []

        try:
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
            ray_pixels = []

            for cam_y in range(0, camera_height, sample_rate):
                for cam_x in range(0, camera_width, sample_rate):
                    ray_dir_cam = np.array([
                        (cam_x - cx) / fx,
                        (cam_y - cy) / fy,
                        1.0
                    ], dtype=np.float32)
                    ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
                    ray_dir_world = rotation_matrix @ ray_dir_cam

                    rays_origin_list.append(camera_pos)
                    rays_direction_list.append(ray_dir_world)
                    ray_pixels.append((cam_x, cam_y))

            rays_origin = cp.array(np.array(rays_origin_list, dtype=np.float32))
            gpu_allocations.append(rays_origin)

            rays_direction = cp.array(np.array(rays_direction_list, dtype=np.float32))
            gpu_allocations.append(rays_direction)

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

            # Pass CPU points - filtering happens inside to save GPU memory
            grid_threshold, grid_count = self._build_ground_grid_cupy(
                self.points, min_x, max_x, min_z, max_z,
                resolution_x, resolution_y
            )
            gpu_allocations.append(grid_threshold)
            gpu_allocations.append(grid_count)

            grid_time = time.time() - grid_start
            logger.info(f"Grid built in {grid_time:.2f}s")

            # Ray trace on GPU
            logger.info("Ray tracing on GPU...")
            trace_start = time.time()

            output_valid, output_x, output_y, output_z = self._ray_trace_cupy(
                rays_origin, rays_direction,
                grid_threshold, grid_count,
                min_x, max_x, min_z, max_z,
                resolution_x, resolution_y
            )
            gpu_allocations.extend([output_valid, output_x, output_y, output_z])

            trace_time = time.time() - trace_start
            logger.info(f"Ray tracing completed in {trace_time:.2f}s")

            # Copy results back to CPU
            output_valid_cpu = cp.asnumpy(output_valid)
            output_x_cpu = cp.asnumpy(output_x)
            output_y_cpu = cp.asnumpy(output_y)
            output_z_cpu = cp.asnumpy(output_z)

            # Process results - VECTORIZED for speed
            logger.info("Processing results...")

            # Count valid rays
            valid_count = int(cp.sum(output_valid))
            logger.info(f"Valid rays (hit ground): {valid_count:,} / {ray_count:,} ({100*valid_count/ray_count:.1f}%)")

            # Vectorized processing of valid hits
            valid_mask = output_valid_cpu

            # Extract valid world coordinates
            world_x_valid = output_x_cpu[valid_mask]
            world_z_valid = output_z_cpu[valid_mask]

            # Apply 90° rotation (vectorized)
            rotated_x = world_z_valid
            rotated_y = -world_x_valid

            # Map to pixel coordinates (vectorized)
            pixel_x = ((rotated_x - min_z) / (max_z - min_z) * resolution_x).astype(np.int32)
            pixel_y = ((rotated_y - (-max_x)) / ((-min_x) - (-max_x)) * resolution_y).astype(np.int32)

            # Filter for valid pixel bounds
            in_bounds = (pixel_x >= 0) & (pixel_x < resolution_x) & \
                       (pixel_y >= 0) & (pixel_y < resolution_y)

            pixel_x = pixel_x[in_bounds]
            pixel_y = pixel_y[in_bounds]

            # Get camera pixels for valid hits
            ray_pixels_array = np.array(ray_pixels, dtype=np.int32)
            cam_pixels_valid = ray_pixels_array[valid_mask][in_bounds]

            # Build pixel_mappings and projected_pixels
            # Convert to lists of Python ints for JSON serialization
            pixel_mappings = list(zip(
                [int(x) for x in cam_pixels_valid[:, 0]],
                [int(y) for y in cam_pixels_valid[:, 1]],
                [int(x) for x in pixel_x],
                [int(y) for y in pixel_y]
            ))
            projected_pixels = [[int(x), int(y)] for x, y in zip(pixel_x, pixel_y)]

            # Debug output - show first few
            logger.info(f"First 5 hits:")
            for i in range(min(5, len(pixel_mappings))):
                cam_x, cam_y, px, py = pixel_mappings[i]
                wx, wz = world_x_valid[i], world_z_valid[i]
                logger.info(f"  Hit {i}: world=({wx:.3f}, {wz:.3f}) -> rotated=({rotated_x[i]:.3f}, {rotated_y[i]:.3f}) -> pixel=({px}, {py})")

            # Count unique world coordinates (sampled for performance)
            sample_size = min(10000, len(world_x_valid))
            sample_idx = np.random.choice(len(world_x_valid), sample_size, replace=False) if len(world_x_valid) > sample_size else range(len(world_x_valid))
            unique_world_coords = len(set(zip(
                np.round(world_x_valid[sample_idx], 2),
                np.round(world_z_valid[sample_idx], 2)
            )))
            logger.info(f"Unique world coordinates (sampled): ~{unique_world_coords} (from {sample_size} samples)")

            # Compute bounds and coverage
            if projected_pixels:
                projected_arr = np.array(projected_pixels)
                bounds = {
                    'min_x': int(projected_arr[:, 0].min()),
                    'max_x': int(projected_arr[:, 0].max()),
                    'min_y': int(projected_arr[:, 1].min()),
                    'max_y': int(projected_arr[:, 1].max())
                }

                # Get unique map pixels for visualization (reduces data transfer)
                unique_map_pixels = list(set(map(tuple, projected_pixels)))
                unique_pixel_count = len(unique_map_pixels)

                map_total_pixels = resolution_x * resolution_y
                coverage_percent = (unique_pixel_count / map_total_pixels) * 100

                logger.info(f"  Bounds: X=[{bounds['min_x']}, {bounds['max_x']}], Y=[{bounds['min_y']}, {bounds['max_y']}]")
                logger.info(f"  Unique map pixels: {unique_pixel_count:,} (from {len(projected_pixels):,} rays)")
            else:
                bounds = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
                coverage_percent = 0.0
                unique_map_pixels = []

            compute_time = time.time() - start_time

            logger.info(f"✓ CuPy projection complete: {len(pixel_mappings):,} camera pixels mapped in {compute_time:.2f}s")
            logger.info(f"  Grid build: {grid_time:.2f}s, Ray trace: {trace_time:.2f}s")
            logger.info(f"  Coverage: {coverage_percent:.2f}%")

            return {
                'pixel_mappings': pixel_mappings,
                'projected_pixels': unique_map_pixels,  # Return only unique pixels for visualization
                'bounds': bounds,
                'pixel_count': len(pixel_mappings),
                'coverage_percent': round(coverage_percent, 2),
                'compute_time': round(compute_time, 2),
                'cuda_accelerated': True,
                'cupy_accelerated': True,
                'grid_build_time': round(grid_time, 2),
                'ray_trace_time': round(trace_time, 2)
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"CuPy projection failed: {error_msg}")

            # Check if it was an OOM error
            if "Out of memory" in error_msg or "out of memory" in error_msg:
                logger.warning("⚠️  GPU ran out of memory. Your RTX 3050 has 6GB but the point cloud is large.")
                logger.warning("💡 Suggestions:")
                logger.warning("   1. Use CPU mode (slower but works)")
                logger.warning("   2. Downsample your point cloud")
                logger.warning("   3. Reduce map resolution")

            logger.info("Falling back to CPU projection")
            from camera_projection import CameraProjector
            cpu_projector = CameraProjector(self.point_cloud_path)
            return cpu_projector.project_camera_to_map(
                camera_pose, map_info, camera_width, camera_height, sample_rate
            )

        finally:
            # CRITICAL: Clean up ALL GPU memory - always executes even on exceptions
            logger.debug(f"Cleaning up {len(gpu_allocations)} GPU allocations")

            for i, arr in enumerate(gpu_allocations):
                try:
                    del arr
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete GPU array {i}: {cleanup_err}")

            # Free all memory blocks from the GPU memory pool
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                logger.debug("GPU memory pool freed successfully")
            except Exception as cleanup_err:
                logger.error(f"Failed to free GPU memory pool: {cleanup_err}")
