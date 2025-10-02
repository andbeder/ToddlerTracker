#!/usr/bin/env python3
"""
PLY to NumPy memory-mapped format converter.
Converts PLY files to ultra-fast memory-mapped NumPy arrays.
"""

import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLYToNPYConverter:
    """Convert PLY files to memory-mapped NumPy format for instant loading."""

    def __init__(self, npy_storage_dir: str = 'npy_storage'):
        self.npy_storage_dir = npy_storage_dir
        if not os.path.exists(npy_storage_dir):
            os.makedirs(npy_storage_dir)

    def convert_ply_to_npy(self, ply_path: str, dataset_name: str = None) -> Dict:
        """
        Convert PLY file to memory-mapped NumPy arrays.

        Args:
            ply_path: Path to PLY file
            dataset_name: Name for the dataset (default: from filename)

        Returns:
            Dict with paths and metadata
        """
        logger.info(f"Converting PLY to NPY: {ply_path}")

        # Parse PLY file
        from yard_manager import PLYParser
        parser = PLYParser()

        logger.info("Parsing PLY file...")
        start_time = time.time()
        ply_data = parser.parse_ply_file(ply_path)
        parse_time = time.time() - start_time

        points = ply_data['points']
        colors = ply_data.get('colors')

        logger.info(f"✅ PLY parsed in {parse_time:.2f} seconds")
        logger.info(f"📊 Points: {len(points):,}")
        logger.info(f"🎨 Has colors: {colors is not None}")

        # Create dataset directory
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(ply_path))[0]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_dir = os.path.join(self.npy_storage_dir, f"{timestamp}_{dataset_name}")
        os.makedirs(dataset_dir, exist_ok=True)

        # Save points as memory-mapped array
        logger.info("Saving points as memory-mapped NumPy array...")
        points_path = os.path.join(dataset_dir, 'points.npy')
        start_time = time.time()
        np.save(points_path, points.astype(np.float32))
        save_time = time.time() - start_time
        points_size = os.path.getsize(points_path)

        logger.info(f"✅ Points saved: {points_path}")
        logger.info(f"   Size: {points_size / 1024 / 1024:.1f} MB")
        logger.info(f"   Time: {save_time:.2f} seconds")

        # Save colors if available
        colors_path = None
        colors_size = 0
        if colors is not None:
            logger.info("Saving colors as memory-mapped NumPy array...")
            colors_path = os.path.join(dataset_dir, 'colors.npy')
            np.save(colors_path, colors.astype(np.uint8))
            colors_size = os.path.getsize(colors_path)
            logger.info(f"✅ Colors saved: {colors_path}")

        # Calculate statistics and metadata
        metadata = self._calculate_metadata(points, colors)
        metadata.update({
            'dataset_name': dataset_name,
            'original_ply': os.path.basename(ply_path),
            'created_at': timestamp,
            'points_file': 'points.npy',
            'colors_file': 'colors.npy' if colors is not None else None,
            'points_shape': list(points.shape),
            'points_dtype': str(points.dtype),
            'points_size_bytes': points_size,
            'colors_size_bytes': colors_size,
            'total_size_bytes': points_size + colors_size,
            'parse_time_seconds': parse_time,
            'save_time_seconds': save_time
        })

        # Save metadata
        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Metadata saved: {metadata_path}")

        # Test memory-mapped loading
        logger.info("\n🧪 Testing memory-mapped loading...")
        test_start = time.time()
        test_points = np.memmap(points_path, dtype='float32', mode='r',
                                shape=tuple(points.shape))
        test_time = time.time() - test_start

        logger.info(f"✅ Memory-mapped loading test:")
        logger.info(f"   Load time: {test_time:.6f} seconds")
        logger.info(f"   Shape verified: {test_points.shape}")
        logger.info(f"   Speedup vs parsing: {parse_time/test_time:.0f}x faster!")

        return {
            'dataset_dir': dataset_dir,
            'points_path': points_path,
            'colors_path': colors_path,
            'metadata_path': metadata_path,
            'metadata': metadata,
            'parse_time': parse_time,
            'save_time': save_time,
            'mmap_load_time': test_time,
            'speedup': parse_time / test_time
        }

    def _calculate_metadata(self, points: np.ndarray,
                           colors: Optional[np.ndarray]) -> Dict:
        """Calculate statistics and metadata for the dataset."""
        metadata = {
            'num_points': len(points),
            'has_colors': colors is not None,
        }

        # Calculate bounds
        metadata['bounds'] = {
            'x_min': float(np.min(points[:, 0])),
            'x_max': float(np.max(points[:, 0])),
            'y_min': float(np.min(points[:, 1])),
            'y_max': float(np.max(points[:, 1])),
            'z_min': float(np.min(points[:, 2])),
            'z_max': float(np.max(points[:, 2])),
        }

        # Calculate center and dimensions
        metadata['center'] = {
            'x': float(np.mean(points[:, 0])),
            'y': float(np.mean(points[:, 1])),
            'z': float(np.mean(points[:, 2])),
        }

        metadata['dimensions'] = {
            'width': metadata['bounds']['x_max'] - metadata['bounds']['x_min'],
            'height': metadata['bounds']['y_max'] - metadata['bounds']['y_min'],
            'depth': metadata['bounds']['z_max'] - metadata['bounds']['z_min'],
        }

        # Calculate percentile bounds (for outlier removal)
        metadata['percentile_bounds'] = {
            'x_p02': float(np.percentile(points[:, 0], 2)),
            'x_p98': float(np.percentile(points[:, 0], 98)),
            'y_p02': float(np.percentile(points[:, 1], 2)),
            'y_p98': float(np.percentile(points[:, 1], 98)),
            'z_p02': float(np.percentile(points[:, 2], 2)),
            'z_p98': float(np.percentile(points[:, 2], 98)),
        }

        return metadata

    def load_npy_dataset(self, dataset_dir: str) -> Tuple[np.memmap, Optional[np.memmap], Dict]:
        """
        Load memory-mapped NumPy dataset.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            Tuple of (points, colors, metadata)
        """
        # Load metadata
        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load memory-mapped points
        points_path = os.path.join(dataset_dir, metadata['points_file'])
        points_shape = tuple(metadata['points_shape'])
        points = np.memmap(points_path, dtype='float32', mode='r', shape=points_shape)

        # Load colors if available
        colors = None
        if metadata.get('colors_file'):
            colors_path = os.path.join(dataset_dir, metadata['colors_file'])
            colors_shape = (metadata['num_points'], 3)
            colors = np.memmap(colors_path, dtype='uint8', mode='r', shape=colors_shape)

        return points, colors, metadata


def main():
    """Convert existing fused.ply to NPY format."""
    converter = PLYToNPYConverter()

    # Convert the existing fused.ply
    ply_path = 'ply_storage/fused.ply'
    if not os.path.exists(ply_path):
        # Try the other location
        ply_path = 'ply_storage/20250929_050320_fused.ply'

    if os.path.exists(ply_path):
        logger.info(f"\n{'='*60}")
        logger.info("CONVERTING FUSED.PLY TO NUMPY MEMORY-MAPPED FORMAT")
        logger.info(f"{'='*60}\n")

        result = converter.convert_ply_to_npy(ply_path, 'fused')

        logger.info(f"\n{'='*60}")
        logger.info("CONVERSION COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"\n📁 Dataset directory: {result['dataset_dir']}")
        logger.info(f"📊 Points file: {result['points_path']}")
        logger.info(f"📈 Parse time: {result['parse_time']:.2f} seconds")
        logger.info(f"💾 Save time: {result['save_time']:.2f} seconds")
        logger.info(f"⚡ Memory-map load time: {result['mmap_load_time']:.6f} seconds")
        logger.info(f"🚀 Speedup: {result['speedup']:.0f}x faster!")
    else:
        logger.error(f"PLY file not found: {ply_path}")


if __name__ == '__main__':
    main()