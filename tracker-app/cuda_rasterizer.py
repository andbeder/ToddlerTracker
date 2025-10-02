"""
CUDA-accelerated point cloud rasterization for improved performance.
Uses CuPy for GPU acceleration of point cloud processing operations.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    # Test if CuPy actually works with basic operations
    test_arr = cp.array([1, 2, 3])
    test_result = cp.sum(test_arr)
    CUDA_AVAILABLE = True
    logger.info("CUDA acceleration available")
except Exception:
    CUDA_AVAILABLE = False
    logger.warning("CUDA not available, using optimized CPU processing")


class CudaPointCloudRasterizer:
    """GPU-accelerated point cloud rasterization using CUDA."""

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE

    def rasterize_point_cloud(self,
                             points: np.ndarray,
                             center_x: float,
                             center_y: float,
                             rotation: float = 0.0,
                             resolution: float = 0.01,  # meters per pixel
                             output_size: Tuple[int, int] = (800, 600),
                             colors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create top-down rasterized view of point cloud with CUDA acceleration.

        Args:
            points: Nx3 array of XYZ coordinates
            center_x: X coordinate of map center
            center_y: Y coordinate of map center (Z in world coordinates)
            rotation: Rotation angle in degrees
            resolution: Meters per pixel
            output_size: Output image size (width, height)
            colors: Optional Nx3 array of RGB colors

        Returns:
            RGB image array
        """
        # For very large datasets, CUDA has significant overhead
        # Use optimized CPU processing which is faster for our use case
        if len(points) > 1000000:  # 1M+ points
            logger.info(f"Using optimized CPU rasterization for large dataset ({len(points):,} points)")
            return self._rasterize_cpu(points, center_x, center_y, rotation,
                                     resolution, output_size, colors)

        if not self.cuda_available:
            return self._rasterize_cpu(points, center_x, center_y, rotation,
                                     resolution, output_size, colors)

        try:
            logger.info(f"Using CUDA rasterization for dataset ({len(points):,} points)")
            return self._rasterize_cuda(points, center_x, center_y, rotation,
                                      resolution, output_size, colors)
        except Exception as e:
            logger.warning(f"CUDA rasterization failed, falling back to optimized CPU: {e}")
            return self._rasterize_cpu(points, center_x, center_y, rotation,
                                     resolution, output_size, colors)

    def _rasterize_cuda(self,
                       points: np.ndarray,
                       center_x: float,
                       center_y: float,
                       rotation: float,
                       resolution: float,
                       output_size: Tuple[int, int],
                       colors: Optional[np.ndarray]) -> np.ndarray:
        """CUDA-accelerated rasterization."""
        width, height = output_size

        # Transfer data to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)
        if colors is not None:
            colors_gpu = cp.asarray(colors, dtype=cp.uint8)

        # Create output arrays on GPU
        image_gpu = cp.zeros((height, width, 3), dtype=cp.uint8)
        density_gpu = cp.zeros((height, width), dtype=cp.float32)
        height_map_gpu = cp.zeros((height, width), dtype=cp.float32)
        height_count_gpu = cp.zeros((height, width), dtype=cp.int32)

        # Apply rotation if needed
        if abs(rotation) > 0.01:
            angle_rad = cp.radians(rotation)
            cos_a = cp.cos(angle_rad)
            sin_a = cp.sin(angle_rad)

            # Rotate points around center
            x_centered = points_gpu[:, 0] - center_x
            z_centered = points_gpu[:, 2] - center_y  # Y becomes Z in world

            x_rot = cos_a * x_centered - sin_a * z_centered + center_x
            z_rot = sin_a * x_centered + cos_a * z_centered + center_y

            points_gpu[:, 0] = x_rot
            points_gpu[:, 2] = z_rot

        # Calculate world bounds
        world_width = width * resolution
        world_height = height * resolution

        # World coordinate bounds centered on center_x, center_y
        x_min = center_x - world_width / 2
        x_max = center_x + world_width / 2
        z_min = center_y - world_height / 2  # Y becomes Z
        z_max = center_y + world_height / 2

        # Filter points within view bounds
        mask = ((points_gpu[:, 0] >= x_min) & (points_gpu[:, 0] <= x_max) &
                (points_gpu[:, 2] >= z_min) & (points_gpu[:, 2] <= z_max))

        filtered_points = points_gpu[mask]
        if colors is not None:
            filtered_colors = colors_gpu[mask]

        if len(filtered_points) == 0:
            logger.warning("No points in view bounds")
            return cp.asnumpy(image_gpu)

        # Convert world coordinates to pixel coordinates
        px = ((filtered_points[:, 0] - x_min) / resolution).astype(cp.int32)
        py = (height - 1 - ((filtered_points[:, 2] - z_min) / resolution)).astype(cp.int32)

        # Clip to image bounds
        valid_mask = ((px >= 0) & (px < width) & (py >= 0) & (py < height))
        px = px[valid_mask]
        py = py[valid_mask]
        filtered_points_valid = filtered_points[valid_mask]

        if colors is not None:
            filtered_colors_valid = filtered_colors[valid_mask]

        # Use CUDA kernels for efficient rasterization
        if len(px) > 0:
            # Create density and height maps
            self._accumulate_points_cuda(px, py, filtered_points_valid[:, 1],
                                       density_gpu, height_map_gpu, height_count_gpu)

            # Generate colors
            if colors is not None and len(filtered_colors_valid) > 0:
                self._apply_colors_cuda(px, py, filtered_colors_valid, image_gpu)
            else:
                self._generate_height_colors_cuda(density_gpu, height_map_gpu,
                                                height_count_gpu, image_gpu,
                                                cp.min(filtered_points_valid[:, 1]),
                                                cp.max(filtered_points_valid[:, 1]))

        # Apply Gaussian smoothing
        image_gpu = self._gaussian_smooth_cuda(image_gpu)

        # Transfer result back to CPU
        return cp.asnumpy(image_gpu)

    def _rasterize_cuda_basic(self,
                             points: np.ndarray,
                             center_x: float,
                             center_y: float,
                             rotation: float,
                             resolution: float,
                             output_size: Tuple[int, int],
                             colors: Optional[np.ndarray]) -> np.ndarray:
        """Basic CUDA rasterization that works without NVRTC."""
        width, height = output_size

        # For very large point clouds, use chunked processing
        chunk_size = 1000000  # 1M points per chunk
        if len(points) > chunk_size:
            logger.info(f"Processing {len(points):,} points in chunks for rasterization")
            return self._rasterize_chunked(points, center_x, center_y, rotation,
                                         resolution, output_size, colors, chunk_size)

        # Transfer data to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)
        if colors is not None:
            colors_gpu = cp.asarray(colors, dtype=cp.uint8)

        # Create output arrays
        image_gpu = cp.zeros((height, width, 3), dtype=cp.uint8)
        density_gpu = cp.zeros((height, width), dtype=cp.float32)
        height_map_gpu = cp.zeros((height, width), dtype=cp.float32)

        # Apply rotation if needed
        if abs(rotation) > 0.01:
            angle_rad = cp.radians(rotation)
            cos_a = cp.cos(angle_rad)
            sin_a = cp.sin(angle_rad)

            x_centered = points_gpu[:, 0] - center_x
            z_centered = points_gpu[:, 2] - center_y

            x_rot = cos_a * x_centered - sin_a * z_centered + center_x
            z_rot = sin_a * x_centered + cos_a * z_centered + center_y

            points_gpu[:, 0] = x_rot
            points_gpu[:, 2] = z_rot

        # Calculate world bounds
        world_width = width * resolution
        world_height = height * resolution
        x_min = center_x - world_width / 2
        x_max = center_x + world_width / 2
        z_min = center_y - world_height / 2
        z_max = center_y + world_height / 2

        # Filter points within view bounds
        mask = ((points_gpu[:, 0] >= x_min) & (points_gpu[:, 0] <= x_max) &
                (points_gpu[:, 2] >= z_min) & (points_gpu[:, 2] <= z_max))

        filtered_points = points_gpu[mask]
        if colors is not None:
            filtered_colors = colors_gpu[mask]

        if len(filtered_points) == 0:
            logger.warning("No points in view bounds")
            return cp.asnumpy(image_gpu)

        # Convert to pixel coordinates
        px = ((filtered_points[:, 0] - x_min) / resolution).astype(cp.int32)
        py = (height - 1 - ((filtered_points[:, 2] - z_min) / resolution)).astype(cp.int32)

        # Clip to image bounds
        valid_mask = ((px >= 0) & (px < width) & (py >= 0) & (py < height))
        px = px[valid_mask]
        py = py[valid_mask]
        filtered_points_valid = filtered_points[valid_mask]

        if colors is not None:
            filtered_colors_valid = filtered_colors[valid_mask]

        # Simple rasterization using vectorized operations
        if len(px) > 0:
            # Create linear indices for faster access
            indices = py * width + px

            # Count points per pixel using add.at
            cp.add.at(density_gpu.ravel(), indices, 1)

            # Height accumulation
            cp.add.at(height_map_gpu.ravel(), indices, filtered_points_valid[:, 1])

            # Color assignment (simple: use last color for each pixel)
            if colors is not None and len(filtered_colors_valid) > 0:
                # Use the last color that lands on each pixel
                for i in range(len(px)):
                    y_coord, x_coord = int(py[i]), int(px[i])
                    if 0 <= y_coord < height and 0 <= x_coord < width:
                        image_gpu[y_coord, x_coord] = filtered_colors_valid[i]
            else:
                # Generate height-based colors
                self._generate_height_colors_basic(density_gpu, height_map_gpu, image_gpu,
                                                 filtered_points_valid)

        logger.info(f"CUDA basic rasterization completed for {len(points):,} points")
        return cp.asnumpy(image_gpu)

    def _rasterize_chunked(self,
                          points: np.ndarray,
                          center_x: float,
                          center_y: float,
                          rotation: float,
                          resolution: float,
                          output_size: Tuple[int, int],
                          colors: Optional[np.ndarray],
                          chunk_size: int) -> np.ndarray:
        """Process very large point clouds in chunks."""
        width, height = output_size

        # Create output arrays on GPU
        image_gpu = cp.zeros((height, width, 3), dtype=cp.uint8)
        density_gpu = cp.zeros((height, width), dtype=cp.float32)
        height_map_gpu = cp.zeros((height, width), dtype=cp.float32)

        logger.info(f"Rasterizing {len(points):,} points in chunks of {chunk_size:,}")

        for i in range(0, len(points), chunk_size):
            chunk_end = min(i + chunk_size, len(points))
            chunk_points = points[i:chunk_end]
            chunk_colors = colors[i:chunk_end] if colors is not None else None

            # Process this chunk using basic method
            chunk_result = self._rasterize_cuda_basic(chunk_points, center_x, center_y,
                                                    rotation, resolution, output_size,
                                                    chunk_colors)

            # Accumulate results (blend with existing image)
            chunk_gpu = cp.asarray(chunk_result)
            mask = cp.any(chunk_gpu > 0, axis=2)  # Where chunk has content
            image_gpu[mask] = chunk_gpu[mask]

        logger.info(f"Chunked rasterization completed")
        return cp.asnumpy(image_gpu)

    def _generate_height_colors_basic(self, density_map, height_map, image, filtered_points):
        """Generate height-based colors using basic operations."""
        # Normalize height map where we have points
        point_mask = density_map > 0
        if not cp.any(point_mask):
            return

        # Calculate average height per pixel
        height_avg = cp.where(point_mask, height_map / density_map, 0)

        # Get height range from filtered points
        y_min = float(cp.min(filtered_points[:, 1]))
        y_max = float(cp.max(filtered_points[:, 1]))
        y_range = y_max - y_min

        if y_range <= 0:
            # All same height, use a default color
            image[point_mask] = [100, 100, 100]  # Gray
            return

        # Normalize heights
        height_norm = (height_avg - y_min) / y_range
        height_norm = cp.clip(height_norm, 0, 1)

        # Create color gradient (blue -> green -> red)
        red = cp.where(height_norm > 0.5, 255 * (height_norm - 0.5) * 2, 0)
        green = cp.where(height_norm < 0.5, 255 * height_norm * 2,
                        255 * (1 - (height_norm - 0.5) * 2))
        blue = cp.where(height_norm < 0.5, 255 * (1 - height_norm * 2), 0)

        # Apply colors where we have points
        image[point_mask, 0] = red[point_mask].astype(cp.uint8)
        image[point_mask, 1] = green[point_mask].astype(cp.uint8)
        image[point_mask, 2] = blue[point_mask].astype(cp.uint8)

    def _accumulate_points_cuda(self, px, py, heights, density_map, height_map, height_count):
        """Accumulate point data using CUDA."""
        # Use CuPy's built-in functions for efficient accumulation
        indices = py * density_map.shape[1] + px

        # Count points per pixel
        cp.add.at(density_map.ravel(), indices, 1)

        # Sum heights per pixel
        cp.add.at(height_map.ravel(), indices, heights)
        cp.add.at(height_count.ravel(), indices, 1)

    def _apply_colors_cuda(self, px, py, colors, image):
        """Apply point colors using CUDA."""
        # Simple color assignment (last color wins for each pixel)
        for i in range(len(px)):
            y, x = int(py[i]), int(px[i])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                image[y, x] = colors[i]

    def _generate_height_colors_cuda(self, density_map, height_map, height_count,
                                   image, y_min, y_max):
        """Generate height-based colors using CUDA."""
        # Normalize height map
        mask = height_count > 0
        height_map[mask] = height_map[mask] / height_count[mask]

        y_range = y_max - y_min
        if y_range <= 0:
            return

        # Create height-based color gradient
        height_normalized = (height_map - y_min) / y_range
        height_normalized = cp.clip(height_normalized, 0, 1)

        # Blue to green to red gradient
        # Blue component (high when normalized height is low)
        blue = cp.where(height_normalized < 0.5,
                       255 * (1 - height_normalized * 2), 0)

        # Green component (high in middle range)
        green = cp.where(height_normalized < 0.5,
                        255 * height_normalized * 2,
                        255 * (1 - (height_normalized - 0.5) * 2))

        # Red component (high when normalized height is high)
        red = cp.where(height_normalized > 0.5,
                      255 * ((height_normalized - 0.5) * 2), 0)

        # Apply density modulation
        max_density = cp.max(density_map)
        if max_density > 0:
            intensity_factor = cp.minimum(1.0, density_map / 10.0)
            red = (red * intensity_factor).astype(cp.uint8)
            green = (green * intensity_factor).astype(cp.uint8)
            blue = (blue * intensity_factor).astype(cp.uint8)

        # Set colors where we have points
        point_mask = density_map > 0
        image[point_mask, 0] = red[point_mask]
        image[point_mask, 1] = green[point_mask]
        image[point_mask, 2] = blue[point_mask]

    def _gaussian_smooth_cuda(self, image):
        """Apply Gaussian smoothing using CUDA."""
        try:
            from cupyx.scipy.ndimage import gaussian_filter

            # Apply smoothing to each channel
            for c in range(3):
                image[:, :, c] = gaussian_filter(image[:, :, c], sigma=1.0)

            return image
        except ImportError:
            # Fall back to no smoothing if cupyx not available
            logger.warning("cupyx not available for Gaussian smoothing")
            return image

    def _rasterize_cpu(self,
                      points: np.ndarray,
                      center_x: float,
                      center_y: float,
                      rotation: float,
                      resolution: float,
                      output_size: Tuple[int, int],
                      colors: Optional[np.ndarray]) -> np.ndarray:
        """Optimized CPU rasterization for large point clouds."""
        width, height = output_size

        # For very large point clouds, use chunked processing
        if len(points) > 2000000:  # 2M points
            logger.info(f"Using optimized CPU rasterization for {len(points):,} points")
            return self._rasterize_cpu_chunked(points, center_x, center_y, rotation,
                                             resolution, output_size, colors)

        # Create output image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply rotation if needed
        if abs(rotation) > 0.01:
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotate around center
            x_centered = points[:, 0] - center_x
            z_centered = points[:, 2] - center_y

            x_rot = cos_a * x_centered - sin_a * z_centered + center_x
            z_rot = sin_a * x_centered + cos_a * z_centered + center_y

            points_rotated = np.copy(points)
            points_rotated[:, 0] = x_rot
            points_rotated[:, 2] = z_rot
            points = points_rotated

        # Calculate world bounds
        world_width = width * resolution
        world_height = height * resolution

        x_min = center_x - world_width / 2
        x_max = center_x + world_width / 2
        z_min = center_y - world_height / 2
        z_max = center_y + world_height / 2

        # Filter points within boundaries
        mask = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max))

        filtered_points = points[mask]
        if colors is not None:
            filtered_colors = colors[mask]
        else:
            filtered_colors = None

        # Create density and height maps
        density_map = np.zeros((height, width), dtype=np.float32)
        height_map = np.zeros((height, width), dtype=np.float32)
        height_count = np.zeros((height, width), dtype=np.int32)

        # Rasterize points
        for i, point in enumerate(filtered_points):
            # Convert world coordinates to pixel coordinates
            px = int((point[0] - x_min) / resolution)
            py = int(height - 1 - ((point[2] - z_min) / resolution))

            if 0 <= px < width and 0 <= py < height:
                density_map[py, px] += 1
                height_map[py, px] += point[1]
                height_count[py, px] += 1

                # If colors are available, use them
                if filtered_colors is not None:
                    image[py, px] = filtered_colors[i]

        # Normalize height map
        mask = height_count > 0
        height_map[mask] = height_map[mask] / height_count[mask]

        # If no colors provided, create visualization based on height
        if colors is None:
            y_min = np.min(points[:, 1]) if len(points) > 0 else 0
            y_max = np.max(points[:, 1]) if len(points) > 0 else 1
            y_range = y_max - y_min

            if y_range > 0:
                for y in range(height):
                    for x in range(width):
                        if density_map[y, x] > 0:
                            # Color based on height (blue to red gradient)
                            h_normalized = (height_map[y, x] - y_min) / y_range
                            h_normalized = np.clip(h_normalized, 0, 1)

                            # Create gradient from blue (low) to green (mid) to red (high)
                            if h_normalized < 0.5:
                                # Blue to green
                                t = h_normalized * 2
                                image[y, x, 0] = 0  # R
                                image[y, x, 1] = int(255 * t)  # G
                                image[y, x, 2] = int(255 * (1 - t))  # B
                            else:
                                # Green to red
                                t = (h_normalized - 0.5) * 2
                                image[y, x, 0] = int(255 * t)  # R
                                image[y, x, 1] = int(255 * (1 - t))  # G
                                image[y, x, 2] = 0  # B

                            # Modulate by density
                            intensity_factor = min(1.0, density_map[y, x] / 10.0)
                            image[y, x] = (image[y, x] * intensity_factor).astype(np.uint8)

        # Apply Gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter
            for c in range(3):
                image[:, :, c] = gaussian_filter(image[:, :, c], sigma=1.0)
        except ImportError:
            pass

        return image

    def _rasterize_cpu_chunked(self,
                              points: np.ndarray,
                              center_x: float,
                              center_y: float,
                              rotation: float,
                              resolution: float,
                              output_size: Tuple[int, int],
                              colors: Optional[np.ndarray]) -> np.ndarray:
        """Chunked CPU rasterization for very large point clouds."""
        width, height = output_size

        logger.info(f"Chunked CPU rasterization for {len(points):,} points")

        # Create output arrays
        image = np.zeros((height, width, 3), dtype=np.uint8)
        density_map = np.zeros((height, width), dtype=np.float32)
        height_map = np.zeros((height, width), dtype=np.float32)
        height_count = np.zeros((height, width), dtype=np.int32)

        # Calculate world bounds once
        world_width = width * resolution
        world_height = height * resolution
        x_min = center_x - world_width / 2
        x_max = center_x + world_width / 2
        z_min = center_y - world_height / 2
        z_max = center_y + world_height / 2

        # Process in chunks to manage memory
        chunk_size = 1000000  # 1M points per chunk

        for i in range(0, len(points), chunk_size):
            chunk_end = min(i + chunk_size, len(points))
            chunk_points = points[i:chunk_end].copy()
            chunk_colors = colors[i:chunk_end] if colors is not None else None

            # Apply rotation to chunk if needed
            if abs(rotation) > 0.01:
                angle_rad = np.radians(rotation)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)

                x_centered = chunk_points[:, 0] - center_x
                z_centered = chunk_points[:, 2] - center_y

                x_rot = cos_a * x_centered - sin_a * z_centered + center_x
                z_rot = sin_a * x_centered + cos_a * z_centered + center_y

                chunk_points[:, 0] = x_rot
                chunk_points[:, 2] = z_rot

            # Filter points within view bounds
            mask = ((chunk_points[:, 0] >= x_min) & (chunk_points[:, 0] <= x_max) &
                    (chunk_points[:, 2] >= z_min) & (chunk_points[:, 2] <= z_max))

            filtered_points = chunk_points[mask]
            if chunk_colors is not None:
                filtered_colors = chunk_colors[mask]
            else:
                filtered_colors = None

            if len(filtered_points) == 0:
                continue

            # Convert to pixel coordinates
            px = ((filtered_points[:, 0] - x_min) / resolution).astype(np.int32)
            py = (height - 1 - ((filtered_points[:, 2] - z_min) / resolution)).astype(np.int32)

            # Clip to image bounds
            valid_mask = ((px >= 0) & (px < width) & (py >= 0) & (py < height))
            px = px[valid_mask]
            py = py[valid_mask]
            filtered_points_valid = filtered_points[valid_mask]

            if filtered_colors is not None:
                filtered_colors_valid = filtered_colors[valid_mask]

            # Accumulate point data using vectorized operations where possible
            if len(px) > 0:
                # Use numpy's add.at for efficient accumulation
                indices = py * width + px
                np.add.at(density_map.ravel(), indices, 1)
                np.add.at(height_map.ravel(), indices, filtered_points_valid[:, 1])
                np.add.at(height_count.ravel(), indices, 1)

                # Color assignment (simple: use last color for each pixel)
                if filtered_colors is not None:
                    for j in range(len(px)):
                        y_coord, x_coord = py[j], px[j]
                        image[y_coord, x_coord] = filtered_colors_valid[j]

        # Normalize height map
        mask = height_count > 0
        height_map[mask] = height_map[mask] / height_count[mask]

        # If no colors provided, create visualization based on height
        if colors is None:
            y_min = np.min(points[:, 1]) if len(points) > 0 else 0
            y_max = np.max(points[:, 1]) if len(points) > 0 else 1
            y_range = y_max - y_min

            if y_range > 0:
                # Vectorized height coloring for better performance
                point_mask = density_map > 0
                if np.any(point_mask):
                    height_normalized = (height_map - y_min) / y_range
                    height_normalized = np.clip(height_normalized, 0, 1)

                    # Create gradient from blue (low) to green (mid) to red (high)
                    blue_mask = height_normalized < 0.5
                    red_mask = height_normalized >= 0.5

                    # Blue to green
                    t = height_normalized * 2
                    image[point_mask & blue_mask, 0] = 0  # R
                    image[point_mask & blue_mask, 1] = (255 * t[point_mask & blue_mask]).astype(np.uint8)  # G
                    image[point_mask & blue_mask, 2] = (255 * (1 - t[point_mask & blue_mask])).astype(np.uint8)  # B

                    # Green to red
                    t = (height_normalized - 0.5) * 2
                    image[point_mask & red_mask, 0] = (255 * t[point_mask & red_mask]).astype(np.uint8)  # R
                    image[point_mask & red_mask, 1] = (255 * (1 - t[point_mask & red_mask])).astype(np.uint8)  # G
                    image[point_mask & red_mask, 2] = 0  # B

                    # Modulate by density
                    intensity_factor = np.minimum(1.0, density_map / 10.0)
                    for c in range(3):
                        image[:, :, c] = (image[:, :, c] * intensity_factor).astype(np.uint8)

        # Apply light Gaussian smoothing if available
        try:
            from scipy.ndimage import gaussian_filter
            for c in range(3):
                image[:, :, c] = gaussian_filter(image[:, :, c], sigma=0.5)
        except ImportError:
            pass

        logger.info(f"Chunked CPU rasterization completed")
        return image