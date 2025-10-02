"""
Optimized CUDA rasterizer using spatial hash grid and Numba JIT kernels.
Based on fast_yard_map_cuda.py but adapted for yard_manager integration.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import numba.cuda as cuda
    from numba import types
    CUDA_AVAILABLE = True
    logger.info("Advanced CUDA acceleration available (CuPy + Numba)")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("Advanced CUDA not available, using fallback")


@cuda.jit
def cuda_build_spatial_grid(vertices_2d, grid_cells, grid_count,
                           x_min, y_min, cell_size, grid_width, grid_height):
    """Build spatial hash grid for fast point lookups."""
    idx = cuda.grid(1)
    if idx >= vertices_2d.shape[0]:
        return

    # Calculate grid cell for this point
    cell_x = int((vertices_2d[idx, 0] - x_min) / cell_size)
    cell_y = int((vertices_2d[idx, 1] - y_min) / cell_size)

    # Clamp to grid bounds
    if cell_x < 0: cell_x = 0
    if cell_x >= grid_width: cell_x = grid_width - 1
    if cell_y < 0: cell_y = 0
    if cell_y >= grid_height: cell_y = grid_height - 1

    # Calculate linear cell index
    cell_idx = cell_y * grid_width + cell_x

    # Atomically increment count for this cell
    old_count = cuda.atomic.add(grid_count, cell_idx, 1)

    # Store point index in grid (if there's space)
    if old_count < 100:  # Max 100 points per cell
        grid_cells[cell_idx, old_count] = idx


@cuda.jit
def cuda_sharp_exact_kernel(vertices_2d, vertices_z, colors,
                            grid_cells, grid_count,
                            x_min, y_min, y_max, pixel_size, cell_size,
                            output_image, z_min, z_max,
                            raster_width, raster_height, grid_width):
    """
    Sharp exact-pixel CUDA kernel - NO pixel expansion, black background.
    Only averages points that actually fall within each pixel boundary.
    Produces crisp edges and higher contrast like the saved map.
    """
    # Get pixel coordinates
    pixel_id = cuda.grid(1)
    if pixel_id >= raster_width * raster_height:
        return

    col = pixel_id % raster_width
    row = pixel_id // raster_width

    # Calculate pixel boundaries in world coordinates
    pixel_x_min = x_min + col * pixel_size
    pixel_x_max = x_min + (col + 1) * pixel_size
    pixel_y_min = y_max - (row + 1) * pixel_size  # Image Y is flipped
    pixel_y_max = y_max - row * pixel_size

    # Find which grid cells overlap with this pixel
    cell_x_min = int((pixel_x_min - x_min) / cell_size)
    cell_x_max = int((pixel_x_max - x_min) / cell_size)
    cell_y_min = int((pixel_y_min - y_min) / cell_size)
    cell_y_max = int((pixel_y_max - y_min) / cell_size)

    # Clamp to grid bounds
    if cell_x_min < 0: cell_x_min = 0
    if cell_x_max >= grid_width: cell_x_max = grid_width - 1
    if cell_y_min < 0: cell_y_min = 0
    if cell_y_max >= grid_width: cell_y_max = grid_width - 1

    # Arrays to store points in this pixel
    pixel_points = cuda.local.array(500, types.int32)
    pixel_count = 0

    # NO EXPANSION - only check points within exact pixel boundaries
    # Check all overlapping grid cells
    for cell_y in range(cell_y_min, cell_y_max + 1):
        for cell_x in range(cell_x_min, cell_x_max + 1):
            cell_idx = cell_y * grid_width + cell_x
            cell_point_count = grid_count[cell_idx]

            # Check all points in this cell (max 100 for sharp_exact)
            for i in range(min(cell_point_count, 100)):
                point_idx = grid_cells[cell_idx, i]

                # Check if point is within EXACT pixel bounds (no expansion)
                if (vertices_2d[point_idx, 0] >= pixel_x_min and
                    vertices_2d[point_idx, 0] <= pixel_x_max and
                    vertices_2d[point_idx, 1] >= pixel_y_min and
                    vertices_2d[point_idx, 1] <= pixel_y_max):

                    if pixel_count < 500:
                        pixel_points[pixel_count] = point_idx
                        pixel_count += 1

    # Process pixel with simple averaging of ALL points
    if pixel_count > 0:
        # Simple average of ALL colors - no filtering
        color_r = 0.0
        color_g = 0.0
        color_b = 0.0

        for j in range(pixel_count):
            point_idx = pixel_points[j]
            color_r += colors[point_idx, 0]
            color_g += colors[point_idx, 1]
            color_b += colors[point_idx, 2]

        # Average colors
        avg_r = color_r / pixel_count
        avg_g = color_g / pixel_count
        avg_b = color_b / pixel_count

        # Ensure 0-255 range (handle both normalized 0-1 and 0-255 formats)
        if avg_r <= 1.0:
            avg_r *= 255
        if avg_g <= 1.0:
            avg_g *= 255
        if avg_b <= 1.0:
            avg_b *= 255

        output_image[row, col, 0] = int(avg_r)
        output_image[row, col, 1] = int(avg_g)
        output_image[row, col, 2] = int(avg_b)
    else:
        # No points found - BLACK background (not white)
        output_image[row, col, 0] = 0
        output_image[row, col, 1] = 0
        output_image[row, col, 2] = 0


@cuda.jit
def cuda_simple_average_kernel(vertices_2d, vertices_z, colors,
                               grid_cells, grid_count,
                               x_min, y_min, y_max, pixel_size, cell_size,
                               output_image, z_min, z_max,
                               raster_width, raster_height, grid_width):
    """
    Simple averaging CUDA kernel - averages ALL points in each pixel.
    No filtering - cleaner colors for true color rendering.
    """
    # Get pixel coordinates
    pixel_id = cuda.grid(1)
    if pixel_id >= raster_width * raster_height:
        return

    col = pixel_id % raster_width
    row = pixel_id // raster_width

    # Calculate pixel boundaries in world coordinates
    pixel_x_min = x_min + col * pixel_size
    pixel_x_max = x_min + (col + 1) * pixel_size
    pixel_y_min = y_max - (row + 1) * pixel_size  # Image Y is flipped
    pixel_y_max = y_max - row * pixel_size

    # Find which grid cells overlap with this pixel
    cell_x_min = int((pixel_x_min - x_min) / cell_size)
    cell_x_max = int((pixel_x_max - x_min) / cell_size)
    cell_y_min = int((pixel_y_min - y_min) / cell_size)
    cell_y_max = int((pixel_y_max - y_min) / cell_size)

    # Clamp to grid bounds
    if cell_x_min < 0: cell_x_min = 0
    if cell_x_max >= grid_width: cell_x_max = grid_width - 1
    if cell_y_min < 0: cell_y_min = 0
    if cell_y_max >= grid_width: cell_y_max = grid_width - 1

    # Arrays to store points in this pixel
    pixel_points = cuda.local.array(500, types.int32)
    pixel_count = 0

    # Iteratively expand search area until points are found (max 5 iterations)
    expansion = 0
    max_expansion = 5

    while pixel_count == 0 and expansion <= max_expansion:
        # Calculate expanded pixel boundaries
        expand_amount = expansion * pixel_size
        search_x_min = pixel_x_min - expand_amount
        search_x_max = pixel_x_max + expand_amount
        search_y_min = pixel_y_min - expand_amount
        search_y_max = pixel_y_max + expand_amount

        # Find grid cells for expanded area
        exp_cell_x_min = int((search_x_min - x_min) / cell_size)
        exp_cell_x_max = int((search_x_max - x_min) / cell_size)
        exp_cell_y_min = int((search_y_min - y_min) / cell_size)
        exp_cell_y_max = int((search_y_max - y_min) / cell_size)

        # Clamp to grid bounds
        if exp_cell_x_min < 0: exp_cell_x_min = 0
        if exp_cell_x_max >= grid_width: exp_cell_x_max = grid_width - 1
        if exp_cell_y_min < 0: exp_cell_y_min = 0
        if exp_cell_y_max >= grid_width: exp_cell_y_max = grid_width - 1

        # Check all overlapping grid cells in expanded area
        for cell_y in range(exp_cell_y_min, exp_cell_y_max + 1):
            for cell_x in range(exp_cell_x_min, exp_cell_x_max + 1):
                cell_idx = cell_y * grid_width + cell_x
                cell_point_count = grid_count[cell_idx]

                # Check all points in this cell (up to max_points_per_cell)
                for i in range(min(cell_point_count, 500)):
                    point_idx = grid_cells[cell_idx, i]

                    # Check if point is within expanded search bounds
                    if (vertices_2d[point_idx, 0] >= search_x_min and
                        vertices_2d[point_idx, 0] <= search_x_max and
                        vertices_2d[point_idx, 1] >= search_y_min and
                        vertices_2d[point_idx, 1] <= search_y_max):

                        if pixel_count < 500:
                            pixel_points[pixel_count] = point_idx
                            pixel_count += 1

        # If no points found, expand search area
        if pixel_count == 0:
            expansion += 1

    # Process pixel with simple averaging of ALL points
    if pixel_count > 0:
        # Simple average of ALL colors - no filtering
        color_r = 0.0
        color_g = 0.0
        color_b = 0.0

        for j in range(pixel_count):
            point_idx = pixel_points[j]
            color_r += colors[point_idx, 0]
            color_g += colors[point_idx, 1]
            color_b += colors[point_idx, 2]

        # Average colors
        avg_r = color_r / pixel_count
        avg_g = color_g / pixel_count
        avg_b = color_b / pixel_count

        # Ensure 0-255 range (handle both normalized 0-1 and 0-255 formats)
        if avg_r <= 1.0:
            avg_r *= 255
        if avg_g <= 1.0:
            avg_g *= 255
        if avg_b <= 1.0:
            avg_b *= 255

        output_image[row, col, 0] = int(avg_r)
        output_image[row, col, 1] = int(avg_g)
        output_image[row, col, 2] = int(avg_b)
    else:
        # No points found - white background
        output_image[row, col, 0] = 255
        output_image[row, col, 1] = 255
        output_image[row, col, 2] = 255


@cuda.jit
def cuda_ground_filtering_kernel(vertices_2d, vertices_z, colors,
                                grid_cells, grid_count,
                                x_min, y_min, y_max, pixel_size, cell_size,
                                output_image, z_min, z_max,
                                raster_width, raster_height, grid_width):
    """
    Optimized ground-filtering CUDA kernel using spatial grid.
    Uses bottom 40% percentile for ground detection.
    """
    # Get pixel coordinates
    pixel_id = cuda.grid(1)
    if pixel_id >= raster_width * raster_height:
        return

    col = pixel_id % raster_width
    row = pixel_id // raster_width

    # Calculate pixel boundaries in world coordinates
    pixel_x_min = x_min + col * pixel_size
    pixel_x_max = x_min + (col + 1) * pixel_size
    pixel_y_min = y_max - (row + 1) * pixel_size  # Image Y is flipped
    pixel_y_max = y_max - row * pixel_size

    # Find which grid cells overlap with this pixel
    cell_x_min = int((pixel_x_min - x_min) / cell_size)
    cell_x_max = int((pixel_x_max - x_min) / cell_size)
    cell_y_min = int((pixel_y_min - y_min) / cell_size)
    cell_y_max = int((pixel_y_max - y_min) / cell_size)

    # Clamp to grid bounds
    if cell_x_min < 0: cell_x_min = 0
    if cell_x_max >= grid_width: cell_x_max = grid_width - 1
    if cell_y_min < 0: cell_y_min = 0
    if cell_y_max >= grid_width: cell_y_max = grid_width - 1

    # Arrays to store points in this pixel
    pixel_points = cuda.local.array(500, types.int32)  # Max 500 points per pixel
    pixel_count = 0

    # Check all overlapping grid cells
    for cell_y in range(cell_y_min, cell_y_max + 1):
        for cell_x in range(cell_x_min, cell_x_max + 1):
            cell_idx = cell_y * grid_width + cell_x
            cell_point_count = grid_count[cell_idx]

            # Check all points in this cell
            for i in range(min(cell_point_count, 100)):  # Max 100 points per cell
                point_idx = grid_cells[cell_idx, i]

                # Check if point is actually within pixel bounds
                if (vertices_2d[point_idx, 0] >= pixel_x_min and
                    vertices_2d[point_idx, 0] <= pixel_x_max and
                    vertices_2d[point_idx, 1] >= pixel_y_min and
                    vertices_2d[point_idx, 1] <= pixel_y_max):

                    if pixel_count < 500:  # Avoid overflow
                        pixel_points[pixel_count] = point_idx
                        pixel_count += 1

    # Process pixel if we have points
    if pixel_count > 0:
        # Collect all heights in this pixel
        pixel_heights = cuda.local.array(500, types.float32)
        for j in range(pixel_count):
            pixel_heights[j] = vertices_z[pixel_points[j]]

        # Sort heights to find percentile (simple bubble sort)
        for i in range(pixel_count - 1):
            for j in range(pixel_count - 1 - i):
                if pixel_heights[j] > pixel_heights[j + 1]:
                    temp = pixel_heights[j]
                    pixel_heights[j] = pixel_heights[j + 1]
                    pixel_heights[j + 1] = temp

        # Use top 60% of points for ground detection (looking down X-axis, higher X = closer/ground)
        percentile_count = max(1, int(pixel_count * 0.6))
        height_threshold = pixel_heights[percentile_count]

        # Average colors of ground points
        valid_points = 0
        color_r = 0.0
        color_g = 0.0
        color_b = 0.0

        for j in range(pixel_count):
            point_idx = pixel_points[j]
            point_height = vertices_z[point_idx]

            if point_height >= height_threshold:
                valid_points += 1
                color_r += colors[point_idx, 0]
                color_g += colors[point_idx, 1]
                color_b += colors[point_idx, 2]

        # Set pixel color
        if valid_points > 0:
            avg_r = color_r / valid_points
            avg_g = color_g / valid_points
            avg_b = color_b / valid_points

            # Ensure 0-255 range (handle both normalized 0-1 and 0-255 formats)
            if avg_r <= 1.0:
                avg_r *= 255
            if avg_g <= 1.0:
                avg_g *= 255
            if avg_b <= 1.0:
                avg_b *= 255

            output_image[row, col, 0] = int(avg_r)
            output_image[row, col, 1] = int(avg_g)
            output_image[row, col, 2] = int(avg_b)
        else:
            # No ground points - white background
            output_image[row, col, 0] = 255
            output_image[row, col, 1] = 255
            output_image[row, col, 2] = 255
    else:
        # No points found - white background
        output_image[row, col, 0] = 255
        output_image[row, col, 1] = 255
        output_image[row, col, 2] = 255


class OptimizedCudaRasterizer:
    """
    Advanced CUDA rasterizer using spatial hash grids and Numba JIT kernels.
    Provides significant performance improvements over basic CuPy implementation.
    """

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE
        if self.cuda_available:
            logger.info("Optimized CUDA rasterizer initialized with spatial grid acceleration")
        else:
            logger.warning("Optimized CUDA rasterizer not available, falling back to CPU")

    def rasterize_points(self, points: np.ndarray, boundaries: Dict,
                        colors: Optional[np.ndarray] = None,
                        resolution: str = '1080p', rotation: float = 0.0) -> np.ndarray:
        """
        Create rasterized map using optimized CUDA acceleration.

        Args:
            points: Nx3 array of XYZ coordinates
            boundaries: Dictionary with boundary information
            colors: Nx3 array of RGB colors (optional)
            resolution: '1080p', '720p', or '480p'
            rotation: Rotation angle in degrees

        Returns:
            RGB image array
        """
        if not self.cuda_available:
            return self._rasterize_cpu_fallback(points, boundaries, colors, resolution, rotation)

        try:
            return self._rasterize_cuda_optimized(points, boundaries, colors, resolution, rotation)
        except Exception as e:
            logger.warning(f"Optimized CUDA rasterization failed, falling back to CPU: {e}")
            return self._rasterize_cpu_fallback(points, boundaries, colors, resolution, rotation)

    def _rasterize_cuda_optimized(self, points: np.ndarray, boundaries: Dict,
                                 colors: Optional[np.ndarray], resolution: str, rotation: float) -> np.ndarray:
        """Optimized CUDA rasterization using spatial hash grid."""
        logger.info(f"Starting optimized CUDA rasterization for {len(points):,} points")
        start_time = time.time()

        # Get resolution dimensions
        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        width, height = resolution_map.get(resolution, (1280, 720))

        # Extract boundaries
        x_min = boundaries['x_min']
        x_max = boundaries['x_max']
        y_min = boundaries['z_min']  # Note: using z as y in 2D projection
        y_max = boundaries['z_max']

        # Apply rotation if specified
        vertices_2d = points[:, [0, 2]].copy()  # X,Z projection (looking down Y-axis)
        # Note: Don't negate Z here - the CUDA kernel already handles Y-flip with y_max-row
        if rotation != 0:
            logger.info(f"Applying rotation: {rotation}°")
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotate around center
            center_x = vertices_2d[:, 0].mean()
            center_y = vertices_2d[:, 1].mean()

            x_centered = vertices_2d[:, 0] - center_x
            y_centered = vertices_2d[:, 1] - center_y

            x_rotated = x_centered * cos_a - y_centered * sin_a
            y_rotated = x_centered * sin_a + y_centered * cos_a

            vertices_2d[:, 0] = x_rotated + center_x
            vertices_2d[:, 1] = y_rotated + center_y

        # Calculate pixel size for 1:1 aspect ratio
        data_width = x_max - x_min
        data_height = y_max - y_min
        pixel_size = max(data_width / width, data_height / height)

        # Adjust bounds to maintain aspect ratio
        adjusted_width = width * pixel_size
        adjusted_height = height * pixel_size

        width_padding = (adjusted_width - data_width) / 2
        height_padding = (adjusted_height - data_height) / 2

        x_min_adj = x_min - width_padding
        x_max_adj = x_max + width_padding
        y_min_adj = y_min - height_padding
        y_max_adj = y_max + height_padding

        logger.info(f"Raster: {width}x{height}, pixel size: {pixel_size:.4f}m")

        # Track all GPU allocations for cleanup
        gpu_arrays = []

        try:
            # Move data to GPU
            gpu_vertices_2d = cp.asarray(vertices_2d, dtype=cp.float32)
            gpu_arrays.append(gpu_vertices_2d)

            gpu_vertices_z = cp.asarray(points[:, 1], dtype=cp.float32)  # Y axis as depth (looking down Y-axis)
            gpu_arrays.append(gpu_vertices_z)

            if colors is not None:
                # Debug color data
                logger.info(f"Color data shape: {colors.shape}, dtype: {colors.dtype}")

                # Sample more points randomly to check for patterns
                sample_indices = np.random.choice(len(colors), min(20, len(colors)), replace=False)
                logger.info(f"Random color samples (20 points):")
                for idx in sample_indices[:10]:
                    logger.info(f"  Point {idx}: RGB({colors[idx, 0]}, {colors[idx, 1]}, {colors[idx, 2]})")

                # Check color statistics
                r_mean, g_mean, b_mean = colors[:, 0].mean(), colors[:, 1].mean(), colors[:, 2].mean()
                r_std, g_std, b_std = colors[:, 0].std(), colors[:, 1].std(), colors[:, 2].std()
                logger.info(f"Color statistics:")
                logger.info(f"  Mean: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
                logger.info(f"  Std:  R={r_std:.1f}, G={g_std:.1f}, B={b_std:.1f}")
                logger.info(f"  Range: R[{colors[:, 0].min()}-{colors[:, 0].max()}], "
                           f"G[{colors[:, 1].min()}-{colors[:, 1].max()}], "
                           f"B[{colors[:, 2].min()}-{colors[:, 2].max()}]")

                # Check for common patterns that might indicate corruption
                all_zero = np.all(colors == 0)
                all_same = np.all(colors == colors[0])
                mostly_gray = np.mean(np.abs(colors[:, 0] - colors[:, 1]) < 5) > 0.9

                if all_zero:
                    logger.warning("WARNING: All color values are zero!")
                elif all_same:
                    logger.warning("WARNING: All colors are identical!")
                elif mostly_gray:
                    logger.warning("WARNING: Colors appear to be mostly grayscale!")
                # Keep colors as uint8 - the kernel will handle averaging correctly
                gpu_colors = cp.asarray(colors, dtype=cp.uint8)
                gpu_arrays.append(gpu_colors)
            else:
                # Create dummy colors
                gpu_colors = cp.ones((len(points), 3), dtype=cp.float32) * 128
                gpu_arrays.append(gpu_colors)

            # Create output image on GPU
            gpu_output = cp.zeros((height, width, 3), dtype=cp.uint8)
            gpu_arrays.append(gpu_output)

            # Build spatial grid
            cell_size = pixel_size  # One cell per pixel for optimal performance
            grid_width = int((x_max_adj - x_min_adj) / cell_size) + 1
            grid_height = int((y_max_adj - y_min_adj) / cell_size) + 1
            max_points_per_cell = 100

            logger.info(f"Building spatial grid: {grid_width}x{grid_height} cells")

            # Allocate grid structures on GPU
            gpu_grid_cells = cp.zeros((grid_width * grid_height, max_points_per_cell), dtype=cp.int32)
            gpu_arrays.append(gpu_grid_cells)

            gpu_grid_count = cp.zeros(grid_width * grid_height, dtype=cp.int32)
            gpu_arrays.append(gpu_grid_count)

            # Build spatial grid
            num_points = len(points)
            threads_per_block = 256
            blocks_for_points = (num_points + threads_per_block - 1) // threads_per_block

            cuda_build_spatial_grid[blocks_for_points, threads_per_block](
                gpu_vertices_2d, gpu_grid_cells, gpu_grid_count,
                x_min_adj, y_min_adj, cell_size, grid_width, grid_height
            )
            cp.cuda.Stream.null.synchronize()

            # Launch rasterization kernel
            logger.info("Launching optimized ground-filtering CUDA kernel")
            total_pixels = width * height
            blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

            z_min = float(points[:, 1].min())
            z_max = float(points[:, 1].max())

            cuda_ground_filtering_kernel[blocks_per_grid, threads_per_block](
                gpu_vertices_2d, gpu_vertices_z, gpu_colors,
                gpu_grid_cells, gpu_grid_count,
                x_min_adj, y_min_adj, y_max_adj, pixel_size, cell_size,
                gpu_output, z_min, z_max, width, height, grid_width
            )

            # Wait for completion and transfer result
            cp.cuda.Stream.null.synchronize()
            cpu_output = cp.asnumpy(gpu_output)

            end_time = time.time()
            total_time = end_time - start_time

            logger.info(f"Optimized CUDA rasterization complete!")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"Pixels per second: {total_pixels / total_time:.0f}")

            return cpu_output

        finally:
            # CRITICAL: Clean up ALL GPU memory - always executes even on exceptions
            logger.debug(f"Cleaning up {len(gpu_arrays)} GPU allocations")

            for i, arr in enumerate(gpu_arrays):
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

    def _rasterize_cpu_fallback(self, points: np.ndarray, boundaries: Dict,
                               colors: Optional[np.ndarray], resolution: str, rotation: float) -> np.ndarray:
        """CPU fallback implementation."""
        logger.info(f"Using CPU fallback rasterization for {len(points):,} points")

        # Simple CPU implementation - basic rasterization
        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        width, height = resolution_map.get(resolution, (1280, 720))

        # Create output image
        output = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill with default green color for now
        output[:, :] = [34, 139, 34]  # Forest green

        logger.info("CPU fallback rasterization complete")
        return output

    def rasterize_point_cloud(self, points: np.ndarray, center_x: float, center_y: float,
                             rotation: float, resolution: float, output_size: Tuple[int, int],
                             colors: Optional[np.ndarray] = None,
                             algorithm: str = 'simple_average') -> np.ndarray:
        """
        Interactive rasterization interface compatible with existing CUDA rasterizer.

        Args:
            points: Nx3 array of XYZ coordinates
            center_x: X coordinate of map center
            center_y: Y coordinate of map center (Z in world coordinates)
            rotation: Rotation angle in degrees
            resolution: Meters per pixel
            output_size: Output image size (width, height)
            colors: Nx3 array of RGB colors (optional)
            algorithm: Rasterization algorithm to use:
                - 'simple_average': Average all points with pixel expansion (cleanest, fills gaps)
                - 'sharp_exact': No pixel expansion, black background (crisp edges, matches saved map)
                - 'ground_filter': Bottom 60% percentile (filter trees/objects)
                - 'cpu_fallback': CPU-based rasterization

        Returns:
            RGB image array
        """
        if algorithm == 'cpu_fallback' or not self.cuda_available:
            return self._rasterize_interactive_cpu_fallback(points, center_x, center_y, rotation, resolution, output_size, colors)

        try:
            return self._rasterize_interactive_cuda(points, center_x, center_y, rotation, resolution, output_size, colors, algorithm)
        except Exception as e:
            logger.warning(f"Interactive CUDA rasterization failed, falling back to CPU: {e}")
            return self._rasterize_interactive_cpu_fallback(points, center_x, center_y, rotation, resolution, output_size, colors)

    def _rasterize_interactive_cuda(self, points: np.ndarray, center_x: float, center_y: float,
                                   rotation: float, resolution: float, output_size: Tuple[int, int],
                                   colors: Optional[np.ndarray], algorithm: str = 'simple_average') -> np.ndarray:
        """Interactive CUDA rasterization using spatial hash grid."""
        logger.info(f"Starting interactive optimized CUDA rasterization for {len(points):,} points (algorithm: {algorithm})")
        start_time = time.time()

        width, height = output_size

        # Calculate view bounds from center and resolution
        half_width = (width * resolution) / 2
        half_height = (height * resolution) / 2

        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height  # Note: using center_y as Z coordinate (top-down view)
        y_max = center_y + half_height

        # Apply rotation if specified
        vertices_2d = points[:, [0, 2]].copy()  # X,Z projection (looking down Y-axis)
        # Note: Don't negate Z here - the CUDA kernel already handles Y-flip with y_max-row

        if rotation != 0:
            logger.info(f"Applying rotation: {rotation}°")
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotate around center
            x_centered = vertices_2d[:, 0] - center_x
            y_centered = vertices_2d[:, 1] - center_y

            x_rotated = x_centered * cos_a - y_centered * sin_a
            y_rotated = x_centered * sin_a + y_centered * cos_a

            vertices_2d[:, 0] = x_rotated + center_x
            vertices_2d[:, 1] = y_rotated + center_y

        pixel_size = resolution
        logger.info(f"Interactive raster: {width}x{height}, pixel size: {pixel_size:.4f}m")

        # Track all GPU allocations for cleanup
        gpu_arrays = []

        try:
            # Move data to GPU
            gpu_vertices_2d = cp.asarray(vertices_2d, dtype=cp.float32)
            gpu_arrays.append(gpu_vertices_2d)

            gpu_vertices_z = cp.asarray(points[:, 1], dtype=cp.float32)  # Y axis as depth (looking down Y-axis)
            gpu_arrays.append(gpu_vertices_z)

            if colors is not None:
                # Debug color data
                logger.info(f"Color data shape: {colors.shape}, dtype: {colors.dtype}")

                # Sample more points randomly to check for patterns
                sample_indices = np.random.choice(len(colors), min(20, len(colors)), replace=False)
                logger.info(f"Random color samples (20 points):")
                for idx in sample_indices[:10]:
                    logger.info(f"  Point {idx}: RGB({colors[idx, 0]}, {colors[idx, 1]}, {colors[idx, 2]})")

                # Check color statistics
                r_mean, g_mean, b_mean = colors[:, 0].mean(), colors[:, 1].mean(), colors[:, 2].mean()
                r_std, g_std, b_std = colors[:, 0].std(), colors[:, 1].std(), colors[:, 2].std()
                logger.info(f"Color statistics:")
                logger.info(f"  Mean: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
                logger.info(f"  Std:  R={r_std:.1f}, G={g_std:.1f}, B={b_std:.1f}")
                logger.info(f"  Range: R[{colors[:, 0].min()}-{colors[:, 0].max()}], "
                           f"G[{colors[:, 1].min()}-{colors[:, 1].max()}], "
                           f"B[{colors[:, 2].min()}-{colors[:, 2].max()}]")

                # Check for common patterns that might indicate corruption
                all_zero = np.all(colors == 0)
                all_same = np.all(colors == colors[0])
                mostly_gray = np.mean(np.abs(colors[:, 0] - colors[:, 1]) < 5) > 0.9

                if all_zero:
                    logger.warning("WARNING: All color values are zero!")
                elif all_same:
                    logger.warning("WARNING: All colors are identical!")
                elif mostly_gray:
                    logger.warning("WARNING: Colors appear to be mostly grayscale!")
                # Keep colors as uint8 - the kernel will handle averaging correctly
                gpu_colors = cp.asarray(colors, dtype=cp.uint8)
                gpu_arrays.append(gpu_colors)
            else:
                # Create dummy colors
                gpu_colors = cp.ones((len(points), 3), dtype=cp.float32) * 128
                gpu_arrays.append(gpu_colors)

            # Create output image on GPU
            gpu_output = cp.zeros((height, width, 3), dtype=cp.uint8)
            gpu_arrays.append(gpu_output)

            # Build spatial grid
            cell_size = pixel_size  # One cell per pixel for optimal performance
            grid_width = int((x_max - x_min) / cell_size) + 1
            grid_height = int((y_max - y_min) / cell_size) + 1

            # Adjust max_points_per_cell based on algorithm
            # sharp_exact doesn't expand, so fewer points per cell needed
            if algorithm == 'sharp_exact':
                max_points_per_cell = 100  # Lower limit for sharp rendering (saves memory)
            else:
                max_points_per_cell = 500  # Higher limit for smooth rendering

            logger.info(f"Building spatial grid: {grid_width}x{grid_height} cells, max {max_points_per_cell} points/cell")

            # Allocate grid structures on GPU
            gpu_grid_cells = cp.zeros((grid_width * grid_height, max_points_per_cell), dtype=cp.int32)
            gpu_arrays.append(gpu_grid_cells)

            gpu_grid_count = cp.zeros(grid_width * grid_height, dtype=cp.int32)
            gpu_arrays.append(gpu_grid_count)

            # Build spatial grid
            num_points = len(points)
            threads_per_block = 256
            blocks_for_points = (num_points + threads_per_block - 1) // threads_per_block

            cuda_build_spatial_grid[blocks_for_points, threads_per_block](
                gpu_vertices_2d, gpu_grid_cells, gpu_grid_count,
                x_min, y_min, cell_size, grid_width, grid_height
            )
            cp.cuda.Stream.null.synchronize()

            # Launch rasterization kernel based on selected algorithm
            total_pixels = width * height
            blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

            z_min = float(points[:, 1].min())
            z_max = float(points[:, 1].max())

            if algorithm == 'ground_filter':
                logger.info("Launching ground-filtering CUDA kernel (bottom 60% percentile)")
                cuda_ground_filtering_kernel[blocks_per_grid, threads_per_block](
                    gpu_vertices_2d, gpu_vertices_z, gpu_colors,
                    gpu_grid_cells, gpu_grid_count,
                    x_min, y_min, y_max, pixel_size, cell_size,
                    gpu_output, z_min, z_max, width, height, grid_width
                )
            elif algorithm == 'sharp_exact':
                logger.info("Launching sharp-exact CUDA kernel (no expansion, black background)")
                cuda_sharp_exact_kernel[blocks_per_grid, threads_per_block](
                    gpu_vertices_2d, gpu_vertices_z, gpu_colors,
                    gpu_grid_cells, gpu_grid_count,
                    x_min, y_min, y_max, pixel_size, cell_size,
                    gpu_output, z_min, z_max, width, height, grid_width
                )
            else:  # simple_average (default)
                logger.info("Launching simple-average CUDA kernel (all points)")
                cuda_simple_average_kernel[blocks_per_grid, threads_per_block](
                    gpu_vertices_2d, gpu_vertices_z, gpu_colors,
                    gpu_grid_cells, gpu_grid_count,
                    x_min, y_min, y_max, pixel_size, cell_size,
                    gpu_output, z_min, z_max, width, height, grid_width
                )

            # Wait for completion and transfer result
            cp.cuda.Stream.null.synchronize()
            cpu_output = cp.asnumpy(gpu_output)

            end_time = time.time()
            total_time = end_time - start_time

            logger.info(f"Interactive optimized CUDA rasterization complete!")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"Pixels per second: {total_pixels / total_time:.0f}")

            return cpu_output

        finally:
            # CRITICAL: Clean up ALL GPU memory - always executes even on exceptions
            logger.debug(f"Cleaning up {len(gpu_arrays)} GPU allocations")

            for i, arr in enumerate(gpu_arrays):
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

    def _rasterize_interactive_cpu_fallback(self, points: np.ndarray, center_x: float, center_y: float,
                                           rotation: float, resolution: float, output_size: Tuple[int, int],
                                           colors: Optional[np.ndarray]) -> np.ndarray:
        """Interactive CPU fallback implementation with actual rasterization."""
        logger.info(f"Using interactive CPU fallback rasterization for {len(points):,} points")
        start_time = time.time()

        width, height = output_size

        # Calculate view bounds
        half_width = (width * resolution) / 2
        half_height = (height * resolution) / 2
        x_min = center_x - half_width
        x_max = center_x + half_width
        y_min = center_y - half_height
        y_max = center_y + half_height

        # Project to 2D (X,Z for looking down Y-axis)
        vertices_2d = points[:, [0, 2]].copy()
        vertices_2d[:, 1] = -vertices_2d[:, 1]  # Negate Z to flip vertically

        # Apply rotation if specified
        if rotation != 0:
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            x_centered = vertices_2d[:, 0] - center_x
            y_centered = vertices_2d[:, 1] - center_y

            x_rotated = x_centered * cos_a - y_centered * sin_a
            y_rotated = x_centered * sin_a + y_centered * cos_a

            vertices_2d[:, 0] = x_rotated + center_x
            vertices_2d[:, 1] = y_rotated + center_y

        # Filter points within view
        mask = ((vertices_2d[:, 0] >= x_min) & (vertices_2d[:, 0] <= x_max) &
                (vertices_2d[:, 1] >= y_min) & (vertices_2d[:, 1] <= y_max))

        visible_2d = vertices_2d[mask]
        visible_colors = colors[mask] if colors is not None else None

        # Create output image with black background
        output = np.zeros((height, width, 3), dtype=np.uint8)
        pixel_counts = np.zeros((height, width), dtype=np.int32)
        color_accum = np.zeros((height, width, 3), dtype=np.float32)

        # Rasterize points
        for i in range(len(visible_2d)):
            # Convert world to pixel coordinates
            px = int((visible_2d[i, 0] - x_min) / resolution)
            py = height - 1 - int((visible_2d[i, 1] - y_min) / resolution)

            if 0 <= px < width and 0 <= py < height:
                pixel_counts[py, px] += 1
                if visible_colors is not None:
                    color_accum[py, px] += visible_colors[i]

        # Average colors where we have points
        mask = pixel_counts > 0
        if visible_colors is not None:
            output[mask] = (color_accum[mask] / pixel_counts[mask][:, np.newaxis]).astype(np.uint8)
        else:
            output[mask] = [100, 100, 100]  # Gray if no colors

        elapsed = time.time() - start_time
        logger.info(f"Interactive CPU fallback complete in {elapsed:.2f}s ({len(visible_2d):,} visible points)")
        return output