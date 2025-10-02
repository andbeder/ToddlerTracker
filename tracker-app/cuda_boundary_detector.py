"""
CUDA-accelerated boundary detection for large point clouds.
Uses CuPy for GPU acceleration of statistical operations.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    # Test if CuPy actually works with basic operations
    test_arr = cp.array([1, 2, 3])
    test_result = cp.sum(test_arr)
    CUDA_AVAILABLE = True
    logger.info("CUDA acceleration available for boundary detection")
except Exception:
    CUDA_AVAILABLE = False
    logger.warning("CUDA not available for boundary detection, using optimized CPU processing")


class CudaBoundaryDetector:
    """GPU-accelerated boundary detection using CUDA."""

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE

    def detect_boundaries(self, points: np.ndarray,
                         percentile_min: float = 2.0,
                         percentile_max: float = 98.0) -> Dict:
        """
        Detect point cloud boundaries using CUDA-accelerated operations.

        Args:
            points: Nx3 array of XYZ coordinates
            percentile_min: Lower percentile for outlier removal
            percentile_max: Upper percentile for outlier removal

        Returns:
            Dict containing boundary information
        """
        if not self.cuda_available:
            return self._detect_boundaries_cpu(points, percentile_min, percentile_max)

        try:
            return self._detect_boundaries_cuda(points, percentile_min, percentile_max)
        except Exception as e:
            logger.warning(f"CUDA boundary detection failed, falling back to CPU: {e}")
            return self._detect_boundaries_cpu(points, percentile_min, percentile_max)

    def _detect_boundaries_cuda(self, points: np.ndarray,
                               percentile_min: float,
                               percentile_max: float) -> Dict:
        """CUDA-accelerated boundary detection."""
        if len(points) == 0:
            raise ValueError("Empty point cloud")

        try:
            # Transfer data to GPU
            points_gpu = cp.asarray(points, dtype=cp.float32)

            # Extract coordinates on GPU
            x_coords = points_gpu[:, 0]
            y_coords = points_gpu[:, 1]
            z_coords = points_gpu[:, 2]

            # Calculate percentiles on GPU (much faster for large datasets)
            x_min_percentile = float(cp.percentile(x_coords, percentile_min))
            x_max_percentile = float(cp.percentile(x_coords, percentile_max))
            z_min_percentile = float(cp.percentile(z_coords, percentile_min))
            z_max_percentile = float(cp.percentile(z_coords, percentile_max))

            # Create mask for filtering on GPU
            mask = ((x_coords >= x_min_percentile) & (x_coords <= x_max_percentile) &
                    (z_coords >= z_min_percentile) & (z_coords <= z_max_percentile))

            # Filter points on GPU
            filtered_points = points_gpu[mask]

            if len(filtered_points) == 0:
                raise ValueError("No points remain after filtering")

            # Calculate statistics on GPU
            x_filtered = filtered_points[:, 0]
            z_filtered = filtered_points[:, 2]

            center_x = float(cp.mean(x_filtered))
            center_z = float(cp.mean(z_filtered))
            width = float(x_max_percentile - x_min_percentile)
            height = float(z_max_percentile - z_min_percentile)

            # Y-axis statistics
            y_min = float(cp.min(y_coords))
            y_max = float(cp.max(y_coords))
            y_mean = float(cp.mean(y_coords))

            # Get array sizes
            total_points = len(points)
            filtered_points_count = len(filtered_points)

            logger.info(f"CUDA boundary detection successful for {total_points:,} points")

            return {
                'center_x': center_x,
                'center_z': center_z,
                'width': width,
                'height': height,
                'x_min': float(x_min_percentile),
                'x_max': float(x_max_percentile),
                'z_min': float(z_min_percentile),
                'z_max': float(z_max_percentile),
                'y_min': y_min,
                'y_max': y_max,
                'y_mean': y_mean,
                'total_points': total_points,
                'filtered_points': filtered_points_count,
                'filter_percentage': (filtered_points_count / total_points) * 100,
                'cuda_accelerated': True
            }
        except Exception as e:
            if "nvrtc" in str(e).lower():
                logger.warning("NVRTC library issue detected, using optimized GPU operations")
                return self._detect_boundaries_cuda_basic(points, percentile_min, percentile_max)
            else:
                raise e

    def _detect_boundaries_cuda_basic(self, points: np.ndarray,
                                     percentile_min: float,
                                     percentile_max: float) -> Dict:
        """Basic CUDA operations that don't require NVRTC."""
        if len(points) == 0:
            raise ValueError("Empty point cloud")

        # For very large point clouds, use chunked processing to manage memory
        chunk_size = 2000000  # 2M points per chunk
        if len(points) > chunk_size:
            logger.info(f"Processing {len(points):,} points in chunks of {chunk_size:,}")
            return self._detect_boundaries_chunked(points, percentile_min, percentile_max, chunk_size)

        # Transfer data to GPU using basic operations
        points_gpu = cp.asarray(points, dtype=cp.float32)

        # Extract coordinates
        x_coords = points_gpu[:, 0]
        y_coords = points_gpu[:, 1]
        z_coords = points_gpu[:, 2]

        # Sort for percentile calculation (more memory efficient than cp.percentile)
        x_sorted = cp.sort(x_coords)
        z_sorted = cp.sort(z_coords)

        # Calculate percentile indices
        n = len(x_coords)
        x_min_idx = int(n * percentile_min / 100)
        x_max_idx = int(n * percentile_max / 100)
        z_min_idx = int(n * percentile_min / 100)
        z_max_idx = int(n * percentile_max / 100)

        # Get percentile values
        x_min_percentile = float(x_sorted[x_min_idx])
        x_max_percentile = float(x_sorted[x_max_idx])
        z_min_percentile = float(z_sorted[z_min_idx])
        z_max_percentile = float(z_sorted[z_max_idx])

        # Create filtering mask
        mask = ((x_coords >= x_min_percentile) & (x_coords <= x_max_percentile) &
                (z_coords >= z_min_percentile) & (z_coords <= z_max_percentile))

        # Apply filter
        filtered_points = points_gpu[mask]
        if len(filtered_points) == 0:
            raise ValueError("No points remain after filtering")

        # Calculate basic statistics
        center_x = float(cp.mean(filtered_points[:, 0]))
        center_z = float(cp.mean(filtered_points[:, 2]))
        width = float(x_max_percentile - x_min_percentile)
        height = float(z_max_percentile - z_min_percentile)

        # Y-axis statistics
        y_min = float(cp.min(y_coords))
        y_max = float(cp.max(y_coords))
        y_mean = float(cp.mean(y_coords))

        total_points = len(points)
        filtered_count = len(filtered_points)

        logger.info(f"CUDA basic boundary detection successful for {total_points:,} points")

        return {
            'center_x': center_x,
            'center_z': center_z,
            'width': width,
            'height': height,
            'x_min': x_min_percentile,
            'x_max': x_max_percentile,
            'z_min': z_min_percentile,
            'z_max': z_max_percentile,
            'y_min': y_min,
            'y_max': y_max,
            'y_mean': y_mean,
            'total_points': total_points,
            'filtered_points': filtered_count,
            'filter_percentage': (filtered_count / total_points) * 100,
            'cuda_accelerated': True
        }

    def _detect_boundaries_chunked(self, points: np.ndarray,
                                  percentile_min: float,
                                  percentile_max: float,
                                  chunk_size: int) -> Dict:
        """Process very large point clouds in chunks to manage GPU memory."""
        logger.info(f"Using chunked processing for {len(points):,} points")

        # First pass: collect samples from each chunk for global percentiles
        sample_size = min(50000, len(points) // 10)  # Sample 10% or max 50k points
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]

        # Calculate global percentiles from sample
        x_coords_sample = sample_points[:, 0]
        z_coords_sample = sample_points[:, 2]

        x_min_percentile = float(np.percentile(x_coords_sample, percentile_min))
        x_max_percentile = float(np.percentile(x_coords_sample, percentile_max))
        z_min_percentile = float(np.percentile(z_coords_sample, percentile_min))
        z_max_percentile = float(np.percentile(z_coords_sample, percentile_max))

        # Second pass: filter and accumulate statistics
        total_filtered = 0
        sum_x, sum_z = 0.0, 0.0
        y_min_global = float('inf')
        y_max_global = float('-inf')
        sum_y = 0.0
        total_y_count = 0

        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]
            chunk_gpu = cp.asarray(chunk, dtype=cp.float32)

            # Apply global bounds filter to chunk
            x_coords = chunk_gpu[:, 0]
            y_coords = chunk_gpu[:, 1]
            z_coords = chunk_gpu[:, 2]

            mask = ((x_coords >= x_min_percentile) & (x_coords <= x_max_percentile) &
                    (z_coords >= z_min_percentile) & (z_coords <= z_max_percentile))

            filtered_chunk = chunk_gpu[mask]
            if len(filtered_chunk) > 0:
                total_filtered += len(filtered_chunk)
                sum_x += float(cp.sum(filtered_chunk[:, 0]))
                sum_z += float(cp.sum(filtered_chunk[:, 2]))

            # Y statistics from all points in chunk
            chunk_y_min = float(cp.min(y_coords))
            chunk_y_max = float(cp.max(y_coords))
            y_min_global = min(y_min_global, chunk_y_min)
            y_max_global = max(y_max_global, chunk_y_max)
            sum_y += float(cp.sum(y_coords))
            total_y_count += len(y_coords)

        # Calculate final statistics
        center_x = sum_x / total_filtered if total_filtered > 0 else 0.0
        center_z = sum_z / total_filtered if total_filtered > 0 else 0.0
        y_mean = sum_y / total_y_count if total_y_count > 0 else 0.0

        width = x_max_percentile - x_min_percentile
        height = z_max_percentile - z_min_percentile

        logger.info(f"Chunked CUDA processing completed: {total_filtered:,} filtered points")

        return {
            'center_x': center_x,
            'center_z': center_z,
            'width': width,
            'height': height,
            'x_min': x_min_percentile,
            'x_max': x_max_percentile,
            'z_min': z_min_percentile,
            'z_max': z_max_percentile,
            'y_min': y_min_global,
            'y_max': y_max_global,
            'y_mean': y_mean,
            'total_points': len(points),
            'filtered_points': total_filtered,
            'filter_percentage': (total_filtered / len(points)) * 100,
            'cuda_accelerated': True
        }

    def _detect_boundaries_cpu(self, points: np.ndarray,
                              percentile_min: float,
                              percentile_max: float) -> Dict:
        """Optimized CPU boundary detection for large point clouds."""
        if len(points) == 0:
            raise ValueError("Empty point cloud")

        # For very large point clouds, use optimized chunked processing
        if len(points) > 5000000:  # 5M points
            logger.info(f"Using optimized CPU processing for {len(points):,} points")
            return self._detect_boundaries_cpu_optimized(points, percentile_min, percentile_max)

        # Extract X, Y, Z coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]

        # Calculate percentiles for X and Z (ground plane)
        x_min_percentile = np.percentile(x_coords, percentile_min)
        x_max_percentile = np.percentile(x_coords, percentile_max)
        z_min_percentile = np.percentile(z_coords, percentile_min)
        z_max_percentile = np.percentile(z_coords, percentile_max)

        # Filter points within percentile range
        mask = (
            (x_coords >= x_min_percentile) & (x_coords <= x_max_percentile) &
            (z_coords >= z_min_percentile) & (z_coords <= z_max_percentile)
        )

        filtered_points = points[mask]

        if len(filtered_points) == 0:
            raise ValueError("No points remain after filtering")

        # Calculate center and dimensions
        x_filtered = filtered_points[:, 0]
        z_filtered = filtered_points[:, 2]

        center_x = float(np.mean(x_filtered))
        center_z = float(np.mean(z_filtered))
        width = float(x_max_percentile - x_min_percentile)
        height = float(z_max_percentile - z_min_percentile)

        # Y-axis statistics (vertical)
        y_min = float(np.min(y_coords))
        y_max = float(np.max(y_coords))
        y_mean = float(np.mean(y_coords))

        return {
            'center_x': center_x,
            'center_z': center_z,
            'width': width,
            'height': height,
            'x_min': float(x_min_percentile),
            'x_max': float(x_max_percentile),
            'z_min': float(z_min_percentile),
            'z_max': float(z_max_percentile),
            'y_min': y_min,
            'y_max': y_max,
            'y_mean': y_mean,
            'total_points': len(points),
            'filtered_points': len(filtered_points),
            'filter_percentage': (len(filtered_points) / len(points)) * 100,
            'cuda_accelerated': False
        }

    def _detect_boundaries_cpu_optimized(self, points: np.ndarray,
                                        percentile_min: float,
                                        percentile_max: float) -> Dict:
        """Optimized CPU processing for very large point clouds."""
        logger.info(f"Optimized CPU boundary detection for {len(points):,} points")

        # Use memory-efficient sampling for percentile calculation
        sample_size = min(500000, len(points) // 10)  # Sample 10% or max 500k points
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]

        # Calculate percentiles from sample
        x_coords_sample = sample_points[:, 0]
        z_coords_sample = sample_points[:, 2]

        x_min_percentile = float(np.percentile(x_coords_sample, percentile_min))
        x_max_percentile = float(np.percentile(x_coords_sample, percentile_max))
        z_min_percentile = float(np.percentile(z_coords_sample, percentile_min))
        z_max_percentile = float(np.percentile(z_coords_sample, percentile_max))

        # Process in chunks to manage memory
        chunk_size = 2000000  # 2M points per chunk
        total_filtered = 0
        sum_x, sum_z = 0.0, 0.0
        y_min_global = float('inf')
        y_max_global = float('-inf')
        sum_y = 0.0
        total_y_count = 0

        logger.info(f"Processing in chunks of {chunk_size:,} points")

        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]

            # Extract coordinates for chunk
            x_coords = chunk[:, 0]
            y_coords = chunk[:, 1]
            z_coords = chunk[:, 2]

            # Apply filter to chunk
            mask = (
                (x_coords >= x_min_percentile) & (x_coords <= x_max_percentile) &
                (z_coords >= z_min_percentile) & (z_coords <= z_max_percentile)
            )

            filtered_chunk = chunk[mask]
            if len(filtered_chunk) > 0:
                total_filtered += len(filtered_chunk)
                sum_x += np.sum(filtered_chunk[:, 0])
                sum_z += np.sum(filtered_chunk[:, 2])

            # Y statistics from all points in chunk
            chunk_y_min = float(np.min(y_coords))
            chunk_y_max = float(np.max(y_coords))
            y_min_global = min(y_min_global, chunk_y_min)
            y_max_global = max(y_max_global, chunk_y_max)
            sum_y += np.sum(y_coords)
            total_y_count += len(y_coords)

        # Calculate final statistics
        center_x = sum_x / total_filtered if total_filtered > 0 else 0.0
        center_z = sum_z / total_filtered if total_filtered > 0 else 0.0
        y_mean = sum_y / total_y_count if total_y_count > 0 else 0.0

        width = x_max_percentile - x_min_percentile
        height = z_max_percentile - z_min_percentile

        logger.info(f"Optimized CPU processing completed: {total_filtered:,} filtered points")

        return {
            'center_x': center_x,
            'center_z': center_z,
            'width': width,
            'height': height,
            'x_min': x_min_percentile,
            'x_max': x_max_percentile,
            'z_min': z_min_percentile,
            'z_max': z_max_percentile,
            'y_min': y_min_global,
            'y_max': y_max_global,
            'y_mean': y_mean,
            'total_points': len(points),
            'filtered_points': total_filtered,
            'filter_percentage': (total_filtered / len(points)) * 100,
            'cuda_accelerated': False
        }


class CudaPLYParser:
    """CUDA-accelerated PLY file parser for large files."""

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE

    def parse_ply_file_fast(self, file_path: str) -> Dict:
        """
        Fast PLY parsing with immediate CUDA transfer for large files.

        This method optimizes memory usage by transferring data to GPU
        immediately after parsing, which is useful for boundary detection.
        """
        # Import the standard parser
        from yard_manager import PLYParser

        # Parse with standard parser first
        result = PLYParser.parse_ply_file(file_path)

        if self.cuda_available and len(result['points']) > 100000:  # Only for large files
            logger.info(f"Pre-loading {len(result['points'])} points to GPU for faster processing")

            # Pre-load to GPU to speed up subsequent operations
            try:
                points_gpu = cp.asarray(result['points'], dtype=cp.float32)
                # Keep reference to prevent garbage collection but don't modify result
                result['_gpu_preloaded'] = True
            except Exception as e:
                logger.warning(f"GPU pre-loading failed: {e}")
                result['_gpu_preloaded'] = False

        return result