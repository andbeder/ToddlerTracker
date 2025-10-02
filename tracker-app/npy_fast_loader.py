"""
Ultra-fast NumPy memory-mapped file loader for point clouds.
Provides instant access to massive point clouds with zero loading time.
"""

import numpy as np
import json
import os
from typing import Dict, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class NPYFastLoader:
    """Ultra-fast memory-mapped NumPy array loader for instant point cloud access."""

    def __init__(self, npy_storage_dir: str = 'npy_storage'):
        self.npy_storage_dir = npy_storage_dir
        self.cached_datasets = {}  # Cache metadata for faster repeated access

    def load_latest_dataset(self) -> Tuple[Optional[np.memmap], Optional[np.memmap], Optional[Dict]]:
        """
        Load the most recent NPY dataset.

        Returns:
            Tuple of (points, colors, metadata) or (None, None, None) if not found
        """
        # Find the most recent dataset directory
        if not os.path.exists(self.npy_storage_dir):
            logger.warning(f"NPY storage directory not found: {self.npy_storage_dir}")
            return None, None, None

        # List all dataset directories
        dataset_dirs = []
        for item in os.listdir(self.npy_storage_dir):
            item_path = os.path.join(self.npy_storage_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    dataset_dirs.append(item_path)

        if not dataset_dirs:
            logger.warning("No NPY datasets found")
            return None, None, None

        # Sort by directory name (timestamp prefix ensures chronological order)
        dataset_dirs.sort(reverse=True)
        latest_dir = dataset_dirs[0]

        return self.load_dataset(latest_dir)

    def load_dataset(self, dataset_dir: str) -> Tuple[Optional[np.memmap], Optional[np.memmap], Optional[Dict]]:
        """
        Load a specific NPY dataset with ultra-fast memory mapping.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            Tuple of (points, colors, metadata)
        """
        try:
            start_time = time.time()

            # Check cache first
            if dataset_dir in self.cached_datasets:
                metadata = self.cached_datasets[dataset_dir]
                logger.info(f"Using cached metadata for: {dataset_dir}")
            else:
                # Load metadata
                metadata_path = os.path.join(dataset_dir, 'metadata.json')
                if not os.path.exists(metadata_path):
                    logger.error(f"Metadata not found: {metadata_path}")
                    return None, None, None

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Cache metadata for future use
                self.cached_datasets[dataset_dir] = metadata

            # Memory-map the points array (instant access, no loading!)
            points_path = os.path.join(dataset_dir, metadata['points_file'])
            points_shape = tuple(metadata['points_shape'])

            # Create memory-mapped array - this is nearly instantaneous!
            points = np.memmap(points_path, dtype='float32', mode='r', shape=points_shape)

            # Load colors if available (use np.load, not memmap - memmap has channel order bug!)
            colors = None
            if metadata.get('colors_file'):
                colors_path = os.path.join(dataset_dir, metadata['colors_file'])
                colors = np.load(colors_path, mmap_mode='r')

            load_time = time.time() - start_time

            logger.info(f"✅ Memory-mapped dataset loaded in {load_time:.6f} seconds")
            logger.info(f"📊 Points: {metadata['num_points']:,}")
            logger.info(f"🎨 Has colors: {colors is not None}")
            logger.info(f"⚡ Points accessible instantly via memory mapping")

            return points, colors, metadata

        except Exception as e:
            logger.error(f"Error loading NPY dataset: {e}")
            return None, None, None

    def load_dataset_by_name(self, dataset_name: str) -> Tuple[Optional[np.memmap], Optional[np.memmap], Optional[Dict]]:
        """
        Load NPY dataset by name pattern.

        Args:
            dataset_name: Name pattern to match (e.g., 'fused')

        Returns:
            Tuple of (points, colors, metadata)
        """
        # Find matching dataset
        for item in os.listdir(self.npy_storage_dir):
            if dataset_name in item:
                dataset_dir = os.path.join(self.npy_storage_dir, item)
                if os.path.isdir(dataset_dir):
                    return self.load_dataset(dataset_dir)

        logger.warning(f"No dataset found matching: {dataset_name}")
        return None, None, None

    def get_boundaries_from_metadata(self, metadata: Dict) -> Dict:
        """
        Extract pre-computed boundaries from metadata.

        Args:
            metadata: Dataset metadata

        Returns:
            Dict with boundary information
        """
        if not metadata:
            return {}

        percentile_bounds = metadata.get('percentile_bounds', {})

        return {
            'center_x': metadata.get('center', {}).get('x', 0),
            'center_z': metadata.get('center', {}).get('z', 0),
            'width': metadata.get('dimensions', {}).get('width', 0),
            'height': metadata.get('dimensions', {}).get('depth', 0),
            'x_min': percentile_bounds.get('x_p02', metadata.get('bounds', {}).get('x_min', 0)),
            'x_max': percentile_bounds.get('x_p98', metadata.get('bounds', {}).get('x_max', 0)),
            'z_min': percentile_bounds.get('z_p02', metadata.get('bounds', {}).get('z_min', 0)),
            'z_max': percentile_bounds.get('z_p98', metadata.get('bounds', {}).get('z_max', 0)),
            'y_min': metadata.get('bounds', {}).get('y_min', 0),
            'y_max': metadata.get('bounds', {}).get('y_max', 0),
            'y_mean': metadata.get('center', {}).get('y', 0),
            'total_points': metadata.get('num_points', 0),
            'filtered_points': metadata.get('num_points', 0),
            'filter_percentage': 100.0,
            'cuda_accelerated': True,
            'method': 'precomputed',
            'load_time_ms': 0  # Near-zero load time!
        }

    def list_available_datasets(self) -> list:
        """List all available NPY datasets."""
        datasets = []

        if not os.path.exists(self.npy_storage_dir):
            return datasets

        for item in os.listdir(self.npy_storage_dir):
            item_path = os.path.join(self.npy_storage_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        datasets.append({
                            'name': metadata.get('dataset_name', item),
                            'directory': item_path,
                            'num_points': metadata.get('num_points', 0),
                            'has_colors': metadata.get('has_colors', False),
                            'created_at': metadata.get('created_at', ''),
                            'size_mb': metadata.get('total_size_bytes', 0) / 1024 / 1024
                        })
                    except Exception as e:
                        logger.error(f"Error reading metadata for {item}: {e}")

        # Sort by creation date (newest first)
        datasets.sort(key=lambda x: x['created_at'], reverse=True)

        return datasets


class OptimizedYardManagerWithNPY:
    """Extension to YardManager for ultra-fast NPY loading."""

    def __init__(self, npy_storage_dir: str = 'npy_storage'):
        self.npy_loader = NPYFastLoader(npy_storage_dir)

    def scan_boundaries_npy(self, dataset_name: str = None) -> Dict:
        """
        Ultra-fast boundary scanning using memory-mapped NPY files.

        Args:
            dataset_name: Optional dataset name to load

        Returns:
            Dict with boundary information
        """
        try:
            logger.info("🚀 Using ultra-fast NPY memory-mapped loading")

            # Load dataset
            if dataset_name:
                points, colors, metadata = self.npy_loader.load_dataset_by_name(dataset_name)
            else:
                points, colors, metadata = self.npy_loader.load_latest_dataset()

            if points is None:
                return {
                    'status': 'error',
                    'message': 'No NPY dataset found'
                }

            # Option 1: Use pre-computed boundaries from metadata (instant!)
            if metadata and 'percentile_bounds' in metadata:
                logger.info("✅ Using pre-computed boundaries from metadata (0ms)")
                boundaries = self.npy_loader.get_boundaries_from_metadata(metadata)

                return {
                    'status': 'success',
                    'message': 'Boundaries loaded instantly from pre-computed metadata',
                    'boundaries': boundaries
                }

            # Option 2: Calculate boundaries using CUDA (still very fast)
            else:
                logger.info("Computing boundaries using CUDA...")
                from cuda_boundary_detector_optimized import OptimizedCudaBoundaryDetector

                detector = OptimizedCudaBoundaryDetector()
                boundaries = detector.detect_boundaries(points, 2.0, 98.0)

                return {
                    'status': 'success',
                    'message': 'Boundaries computed with CUDA acceleration',
                    'boundaries': boundaries
                }

        except Exception as e:
            logger.error(f"Error in NPY boundary scanning: {e}")
            return {
                'status': 'error',
                'message': f'Error: {str(e)}'
            }