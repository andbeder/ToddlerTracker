"""
Yard point cloud processing and visualization manager.
Handles PLY file parsing, boundary detection, and top-down rasterization.
"""

import struct
import numpy as np
import sqlite3
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from io import BytesIO
from PIL import Image
import base64
import time
from cuda_rasterizer import CudaPointCloudRasterizer
from cuda_boundary_detector import CudaBoundaryDetector

logger = logging.getLogger(__name__)

# Project PLY file path - single source of truth for fused point cloud location
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_FILE_PATH = os.path.join(PROJECT_DIR, 'ply_storage', 'fused.ply')


class PLYParser:
    """Parser for PLY (Polygon File Format) point cloud files."""

    @staticmethod
    def parse_ply_file(file_path: str) -> Dict:
        """
        Parse a PLY file and extract point cloud data using trimesh.

        Returns:
            Dict containing points array and metadata
        """
        try:
            # Use trimesh for reliable PLY parsing (same as erik-tracker)
            import trimesh
            mesh = trimesh.load(file_path)

            points = mesh.vertices
            colors = None

            # Get vertex colors if available
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors[:, :3]  # RGB only, drop alpha
                logger.info(f"Loaded colors via trimesh: {colors.shape}")
            else:
                logger.warning("No vertex colors found in PLY")

            return {
                'points': points.astype(np.float32),
                'colors': colors.astype(np.uint8) if colors is not None else None,
                'vertex_count': len(points),
                'has_color': colors is not None,
                'format': 'trimesh'
            }

        except ImportError:
            logger.warning("trimesh not available, falling back to custom parser")
            # Fall back to original parser
            return PLYParser._parse_ply_custom(file_path)
        except Exception as e:
            logger.error(f"trimesh parsing failed: {e}, trying custom parser")
            return PLYParser._parse_ply_custom(file_path)

    @staticmethod
    def _parse_ply_custom(file_path: str) -> Dict:
        """Original custom PLY parser as fallback."""
        try:
            with open(file_path, 'rb') as f:
                # Parse PLY header
                header_lines = []
                vertex_count = 0
                has_color = False
                binary_format = False
                properties = []

                while True:
                    line = f.readline()
                    if not line:
                        break

                    if isinstance(line, bytes):
                        line = line.decode('ascii').strip()
                    else:
                        line = line.strip()

                    header_lines.append(line)

                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('property'):
                        parts = line.split()
                        if len(parts) >= 3:
                            prop_type = parts[1]
                            prop_name = parts[2]
                            properties.append((prop_type, prop_name))
                            if prop_name in ['red', 'green', 'blue']:
                                has_color = True
                    elif 'binary' in line:
                        binary_format = True
                    elif line == 'end_header':
                        break

                if vertex_count == 0:
                    raise ValueError("No vertices found in PLY file")

                # Determine data format
                point_size = 0
                x_idx, y_idx, z_idx = None, None, None
                r_idx, g_idx, b_idx = None, None, None

                for i, (dtype, name) in enumerate(properties):
                    if name == 'x':
                        x_idx = i
                    elif name == 'y':
                        y_idx = i
                    elif name == 'z':
                        z_idx = i
                    elif name == 'red':
                        r_idx = i
                    elif name == 'green':
                        g_idx = i
                    elif name == 'blue':
                        b_idx = i

                    # Calculate size based on data type
                    if dtype in ['float', 'float32']:
                        point_size += 4
                    elif dtype in ['double', 'float64']:
                        point_size += 8
                    elif dtype in ['uchar', 'uint8']:
                        point_size += 1
                    elif dtype in ['short', 'int16']:
                        point_size += 2
                    elif dtype in ['int', 'int32']:
                        point_size += 4

                # Read vertex data
                points = []
                colors = []

                if binary_format:
                    # Read binary data
                    for _ in range(vertex_count):
                        vertex_data = f.read(point_size)
                        offset = 0
                        values = []

                        for dtype, name in properties:
                            if dtype in ['float', 'float32']:
                                val = struct.unpack('<f', vertex_data[offset:offset+4])[0]
                                values.append(val)
                                offset += 4
                            elif dtype in ['double', 'float64']:
                                val = struct.unpack('<d', vertex_data[offset:offset+8])[0]
                                values.append(val)
                                offset += 8
                            elif dtype in ['uchar', 'uint8']:
                                val = struct.unpack('<B', vertex_data[offset:offset+1])[0]
                                values.append(val)
                                offset += 1
                            elif dtype in ['short', 'int16']:
                                val = struct.unpack('<h', vertex_data[offset:offset+2])[0]
                                values.append(val)
                                offset += 2
                            elif dtype in ['int', 'int32']:
                                val = struct.unpack('<i', vertex_data[offset:offset+4])[0]
                                values.append(val)
                                offset += 4

                        if x_idx is not None and y_idx is not None and z_idx is not None:
                            points.append([values[x_idx], values[y_idx], values[z_idx]])

                        if has_color and r_idx is not None:
                            colors.append([values[r_idx], values[g_idx], values[b_idx]])
                else:
                    # Read ASCII data
                    for _ in range(vertex_count):
                        line = f.readline()
                        if isinstance(line, bytes):
                            line = line.decode('ascii')
                        values = list(map(float, line.strip().split()))

                        if x_idx is not None and y_idx is not None and z_idx is not None:
                            points.append([values[x_idx], values[y_idx], values[z_idx]])

                        if has_color and r_idx is not None:
                            colors.append([
                                int(values[r_idx]),
                                int(values[g_idx]),
                                int(values[b_idx])
                            ])

                points_array = np.array(points, dtype=np.float32)
                colors_array = np.array(colors, dtype=np.uint8) if colors else None

                return {
                    'points': points_array,
                    'colors': colors_array,
                    'vertex_count': vertex_count,
                    'has_color': has_color,
                    'format': 'binary' if binary_format else 'ascii'
                }

        except Exception as e:
            logger.error(f"Error parsing PLY file: {e}")
            raise


class BoundaryDetector:
    """Detects boundaries of point cloud with outlier removal."""

    @staticmethod
    def detect_boundaries(points: np.ndarray,
                         percentile_min: float = 2.0,
                         percentile_max: float = 98.0) -> Dict:
        """
        Detect point cloud boundaries using percentile-based outlier removal.

        Args:
            points: Nx3 array of XYZ coordinates
            percentile_min: Lower percentile for outlier removal
            percentile_max: Upper percentile for outlier removal

        Returns:
            Dict containing boundary information
        """
        if len(points) == 0:
            raise ValueError("Empty point cloud")

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
            'filter_percentage': (len(filtered_points) / len(points)) * 100
        }


class PointCloudRasterizer:
    """Rasterizes 3D point cloud to 2D top-down view."""

    @staticmethod
    def rasterize_point_cloud(points: np.ndarray,
                             boundaries: Dict,
                             rotation: float = 0.0,
                             resolution: Tuple[int, int] = (1920, 1080),
                             colors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create top-down rasterized view of point cloud.

        Args:
            points: Nx3 array of XYZ coordinates
            boundaries: Boundary information from BoundaryDetector
            rotation: Rotation angle in degrees
            resolution: Output image resolution (width, height)
            colors: Optional Nx3 array of RGB colors

        Returns:
            RGB image array
        """
        width, height = resolution

        # Create output image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply rotation if needed
        if rotation != 0:
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotate around center
            cx = boundaries['center_x']
            cz = boundaries['center_z']

            x_rot = cos_a * (points[:, 0] - cx) - sin_a * (points[:, 2] - cz) + cx
            z_rot = sin_a * (points[:, 0] - cx) + cos_a * (points[:, 2] - cz) + cz

            points_rotated = np.copy(points)
            points_rotated[:, 0] = x_rot
            points_rotated[:, 2] = z_rot
            points = points_rotated

        # Filter points within boundaries
        mask = (
            (points[:, 0] >= boundaries['x_min']) &
            (points[:, 0] <= boundaries['x_max']) &
            (points[:, 2] >= boundaries['z_min']) &
            (points[:, 2] <= boundaries['z_max'])
        )

        filtered_points = points[mask]
        if colors is not None:
            filtered_colors = colors[mask]
        else:
            filtered_colors = None

        # Map world coordinates to image coordinates
        x_range = boundaries['x_max'] - boundaries['x_min']
        z_range = boundaries['z_max'] - boundaries['z_min']

        # Maintain aspect ratio
        aspect_ratio = width / height  # 16:9 = 1.778
        world_aspect = x_range / z_range

        if world_aspect > aspect_ratio:
            # World is wider than image aspect
            scale = width / x_range
            y_offset = (height - z_range * scale) / 2
            x_offset = 0
        else:
            # World is taller than image aspect
            scale = height / z_range
            x_offset = (width - x_range * scale) / 2
            y_offset = 0

        # Create density map for coloring
        density_map = np.zeros((height, width), dtype=np.float32)
        height_map = np.zeros((height, width), dtype=np.float32)
        height_count = np.zeros((height, width), dtype=np.int32)

        for i, point in enumerate(filtered_points):
            # Convert world coordinates to pixel coordinates
            px = int((point[0] - boundaries['x_min']) * scale + x_offset)
            # Note: Flip Z axis for correct top-down view
            py = int(height - ((point[2] - boundaries['z_min']) * scale + y_offset) - 1)

            if 0 <= px < width and 0 <= py < height:
                density_map[py, px] += 1
                height_map[py, px] += point[1]  # Y is height
                height_count[py, px] += 1

                # If colors are available, use them
                if filtered_colors is not None:
                    image[py, px] = filtered_colors[i]

        # Normalize height map
        mask = height_count > 0
        height_map[mask] = height_map[mask] / height_count[mask]

        # If no colors provided, create visualization based on height or density
        if colors is None:
            # Normalize density for visualization
            max_density = np.max(density_map)
            if max_density > 0:
                density_normalized = (density_map / max_density * 255).astype(np.uint8)

                # Create color map based on height
                y_range = boundaries['y_max'] - boundaries['y_min']
                if y_range > 0:
                    for y in range(height):
                        for x in range(width):
                            if density_map[y, x] > 0:
                                # Color based on height (blue to red gradient)
                                h_normalized = (height_map[y, x] - boundaries['y_min']) / y_range
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

        # Apply Gaussian smoothing for better visualization
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            image[:, :, c] = gaussian_filter(image[:, :, c], sigma=1.0)

        return image


class YardDatabase:
    """Manages yard map storage and retrieval."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the yard maps database table."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS yard_maps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    image_data BLOB NOT NULL,
                    boundaries TEXT NOT NULL,
                    center_x REAL,
                    center_z REAL,
                    width REAL,
                    height REAL,
                    rotation REAL DEFAULT 0,
                    resolution_x INTEGER,
                    resolution_y INTEGER,
                    point_count INTEGER,
                    is_used BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Add is_used column if it doesn't exist (for existing databases)
            try:
                conn.execute('ALTER TABLE yard_maps ADD COLUMN is_used BOOLEAN DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add table for storing PLY files
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ply_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    file_data BLOB NOT NULL,
                    vertex_count INTEGER,
                    has_color BOOLEAN,
                    format TEXT,
                    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Add table for camera-to-map projections
            conn.execute('''
                CREATE TABLE IF NOT EXISTS camera_projections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT NOT NULL,
                    map_id INTEGER NOT NULL,
                    pixel_mappings BLOB NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (map_id) REFERENCES yard_maps(id),
                    UNIQUE(camera_name, map_id)
                )
            ''')

            conn.commit()
        finally:
            conn.close()

    def save_yard_map(self, name: str, image_array: np.ndarray,
                     boundaries: Dict, rotation: float,
                     resolution: Tuple[int, int], point_count: int) -> bool:
        """Save yard map to database."""
        try:
            # Convert image array to PNG bytes
            img = Image.fromarray(image_array)
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()

            conn = sqlite3.connect(self.db_path)

            boundaries_json = json.dumps(boundaries)

            conn.execute('''
                INSERT OR REPLACE INTO yard_maps
                (name, image_data, boundaries, center_x, center_z, width, height,
                 rotation, resolution_x, resolution_y, point_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, image_bytes, boundaries_json,
                boundaries['center_x'], boundaries['center_z'],
                boundaries['width'], boundaries['height'],
                rotation, resolution[0], resolution[1], point_count
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving yard map: {e}")
            return False
        finally:
            conn.close()

    def get_yard_map(self, name: str) -> Optional[Dict]:
        """Get yard map by name."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                'SELECT * FROM yard_maps WHERE name = ?', (name,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'image_data': row[2],
                    'boundaries': json.loads(row[3]),
                    'center_x': row[4],
                    'center_z': row[5],
                    'width': row[6],
                    'height': row[7],
                    'rotation': row[8],
                    'resolution_x': row[9],
                    'resolution_y': row[10],
                    'point_count': row[11],
                    'created_at': row[12]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting yard map: {e}")
            return None
        finally:
            conn.close()

    def get_all_yard_maps(self) -> List[Dict]:
        """Get all yard maps metadata (without image data)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, center_x, center_z, width, height,
                       rotation, resolution_x, resolution_y, point_count, is_used, created_at
                FROM yard_maps ORDER BY created_at DESC
            ''')
            rows = cursor.fetchall()

            maps = []
            for row in rows:
                maps.append({
                    'id': row[0],
                    'name': row[1],
                    'center_x': row[2],
                    'center_z': row[3],
                    'width': row[4],
                    'height': row[5],
                    'rotation': row[6],
                    'resolution_x': row[7],
                    'resolution_y': row[8],
                    'point_count': row[9],
                    'is_used': bool(row[10]),
                    'created_at': row[11]
                })
            return maps
        except Exception as e:
            logger.error(f"Error getting all yard maps: {e}")
            return []
        finally:
            conn.close()

    def get_yard_image(self, map_id: int) -> Optional[bytes]:
        """Get yard map image data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                'SELECT image_data FROM yard_maps WHERE id = ?', (map_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting yard image: {e}")
            return None
        finally:
            conn.close()

    def delete_yard_map(self, name: str) -> bool:
        """Delete a yard map."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM yard_maps WHERE name = ?', (name,))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting yard map: {e}")
            return False
        finally:
            conn.close()

    def set_used_yard_map(self, map_id: int) -> bool:
        """Set a yard map as 'used' for projection, clearing any previous used map."""
        try:
            conn = sqlite3.connect(self.db_path)
            # First, clear all other used flags
            conn.execute('UPDATE yard_maps SET is_used = 0')
            # Then set this one as used
            conn.execute('UPDATE yard_maps SET is_used = 1 WHERE id = ?', (map_id,))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting used yard map: {e}")
            return False
        finally:
            conn.close()

    def get_used_yard_map(self) -> Optional[Dict]:
        """Get the currently used yard map."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, image_data, boundaries, center_x, center_z, width, height,
                       rotation, resolution_x, resolution_y, point_count, created_at
                FROM yard_maps WHERE is_used = 1
                LIMIT 1
            ''')
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'image_data': row[2],
                    'boundaries': json.loads(row[3]),
                    'center_x': row[4],
                    'center_z': row[5],
                    'width': row[6],
                    'height': row[7],
                    'rotation': row[8],
                    'resolution_x': row[9],
                    'resolution_y': row[10],
                    'point_count': row[11],
                    'created_at': row[12]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting used yard map: {e}")
            return None
        finally:
            conn.close()

    def save_ply_file(self, name: str, file_data: bytes,
                     vertex_count: int = None, has_color: bool = None,
                     format: str = None) -> bool:
        """Save PLY file to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO ply_files
                (name, file_data, vertex_count, has_color, format)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, file_data, vertex_count, has_color, format))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving PLY file: {e}")
            return False
        finally:
            conn.close()

    def get_latest_ply_file(self) -> Optional[Dict]:
        """Get the most recently uploaded PLY file."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, file_data, vertex_count, has_color, format, uploaded_at
                FROM ply_files
                ORDER BY uploaded_at DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'file_data': row[2],
                    'vertex_count': row[3],
                    'has_color': row[4],
                    'format': row[5],
                    'uploaded_at': row[6]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting latest PLY file: {e}")
            return None
        finally:
            conn.close()

    def get_ply_file_by_id(self, file_id: int) -> Optional[Dict]:
        """Get PLY file by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, file_data, vertex_count, has_color, format, uploaded_at
                FROM ply_files
                WHERE id = ?
            ''', (file_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'file_data': row[2],
                    'vertex_count': row[3],
                    'has_color': row[4],
                    'format': row[5],
                    'uploaded_at': row[6]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting PLY file: {e}")
            return None
        finally:
            conn.close()

    def get_all_ply_files(self) -> List[Dict]:
        """Get metadata for all PLY files (without file data)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT id, name, vertex_count, has_color, format, uploaded_at
                FROM ply_files
                ORDER BY uploaded_at DESC
            ''')
            rows = cursor.fetchall()

            files = []
            for row in rows:
                files.append({
                    'id': row[0],
                    'name': row[1],
                    'vertex_count': row[2],
                    'has_color': row[3],
                    'format': row[4],
                    'uploaded_at': row[5]
                })
            return files
        except Exception as e:
            logger.error(f"Error getting all PLY files: {e}")
            return []
        finally:
            conn.close()

    def save_camera_projection(self, camera_name: str, map_id: int,
                               pixel_mappings: List, metadata: Dict) -> bool:
        """Save camera-to-map projection."""
        try:
            import pickle

            conn = sqlite3.connect(self.db_path)

            # Serialize pixel mappings
            mappings_blob = pickle.dumps(pixel_mappings)
            metadata_json = json.dumps(metadata)

            conn.execute('''
                INSERT OR REPLACE INTO camera_projections
                (camera_name, map_id, pixel_mappings, metadata)
                VALUES (?, ?, ?, ?)
            ''', (camera_name, map_id, mappings_blob, metadata_json))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving camera projection: {e}")
            return False
        finally:
            conn.close()

    def get_camera_projection(self, camera_name: str, map_id: int) -> Optional[Dict]:
        """Get camera projection for a specific camera and map."""
        try:
            import pickle

            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT pixel_mappings, metadata, created_at
                FROM camera_projections
                WHERE camera_name = ? AND map_id = ?
            ''', (camera_name, map_id))

            row = cursor.fetchone()

            if row:
                pixel_mappings = pickle.loads(row[0])
                metadata = json.loads(row[1]) if row[1] else {}

                return {
                    'camera_name': camera_name,
                    'map_id': map_id,
                    'pixel_mappings': pixel_mappings,
                    'metadata': metadata,
                    'created_at': row[2]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting camera projection: {e}")
            return None
        finally:
            conn.close()

    def get_all_projections_for_map(self, map_id: int) -> List[Dict]:
        """Get all camera projections for a specific map."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT camera_name, created_at
                FROM camera_projections
                WHERE map_id = ?
            ''', (map_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting projections for map {map_id}: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()


class YardManager:
    """Main class for managing yard point cloud processing."""

    def __init__(self, db_path: str = 'yard.db'):
        self.yard_db = YardDatabase(db_path)
        self.parser = PLYParser()
        self.detector = BoundaryDetector()
        self.rasterizer = PointCloudRasterizer()
        self.cuda_rasterizer = CudaPointCloudRasterizer()
        self.cuda_detector = CudaBoundaryDetector()

        # Import optimized database with file-based storage
        try:
            from yard_database_optimized import OptimizedYardDatabase
            self.optimized_db = OptimizedYardDatabase(db_path, 'ply_storage')
            logger.info("Optimized file-based database loaded")
        except ImportError:
            self.optimized_db = None
            logger.warning("Optimized database not available")

        # Import optimized detector with cube projection algorithms
        try:
            from cuda_boundary_detector_optimized import OptimizedCudaBoundaryDetector
            self.optimized_detector = OptimizedCudaBoundaryDetector()
            logger.info("Optimized CUDA boundary detector loaded (cube projection)")
        except ImportError:
            self.optimized_detector = None
            logger.warning("Optimized detector not available, using standard version")

        # Import optimized rasterizer with spatial hash grid
        try:
            from cuda_rasterizer_optimized import OptimizedCudaRasterizer
            self.optimized_rasterizer = OptimizedCudaRasterizer()
            logger.info("Optimized CUDA rasterizer loaded (spatial hash grid)")
        except ImportError:
            self.optimized_rasterizer = None
            logger.warning("Optimized rasterizer not available, using standard version")

        # Import ultra-fast NPY memory-mapped loader
        try:
            from npy_fast_loader import NPYFastLoader
            self.npy_loader = NPYFastLoader('npy_storage')
            logger.info("Ultra-fast NPY memory-mapped loader initialized")
        except ImportError:
            self.npy_loader = None
            logger.warning("NPY fast loader not available")

        # Import PLY to NPY converter for automatic conversion of uploads
        try:
            from ply_to_npy_converter import PLYToNPYConverter
            self.ply_converter = PLYToNPYConverter('npy_storage')
            logger.info("PLY to NPY converter initialized")
        except ImportError:
            self.ply_converter = None
            logger.warning("PLY to NPY converter not available")

    def _parse_ply_from_stream(self, stream):
        """Parse PLY data directly from a file-like stream (optimized for memory blobs)."""
        # Use the existing parser but with stream input
        # Create a temporary file method that works with streams
        import tempfile
        import os

        # For now, use a temporary approach until we can fully optimize the PLY parser
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
            stream.seek(0)  # Reset stream position
            tmp_file.write(stream.read())
            tmp_file.flush()
            tmp_path = tmp_file.name

        try:
            # Parse using existing parser
            result = self.parser.parse_ply_file(tmp_path)
            return result
        finally:
            # Clean up
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def process_ply_file(self, file_path: str) -> Dict:
        """Process PLY file and extract point cloud data."""
        try:
            result = self.parser.parse_ply_file(file_path)
            return {
                'status': 'success',
                'message': f'Successfully parsed {result["vertex_count"]} points',
                'data': {
                    'point_count': result['vertex_count'],
                    'has_color': result['has_color'],
                    'format': result['format']
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error parsing PLY file: {str(e)}'
            }

    def scan_boundaries(self, file_path: str,
                       percentile_min: float = 2.0,
                       percentile_max: float = 98.0) -> Dict:
        """Scan point cloud boundaries with CUDA acceleration."""
        try:
            # Parse PLY file
            ply_data = self.parser.parse_ply_file(file_path)
            points = ply_data['points']

            # Use optimized CUDA boundary detection with cube projection algorithms
            if self.optimized_detector:
                logger.info(f"Using optimized CUDA boundary detection with cube projection ({len(points):,} points)")
                boundaries = self.optimized_detector.detect_boundaries(
                    points, percentile_min, percentile_max
                )
            else:
                logger.info(f"Using standard CPU boundary detection ({len(points):,} points)")
                boundaries = self.detector.detect_boundaries(
                    points, percentile_min, percentile_max
                )

            # Add acceleration info to message
            accel_type = "CUDA-accelerated" if boundaries.get('cuda_accelerated', False) else "CPU"
            message = f'Boundaries detected successfully using {accel_type} processing'

            return {
                'status': 'success',
                'message': message,
                'boundaries': boundaries
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error scanning boundaries: {str(e)}'
            }

    def scan_boundaries_ultra_fast(self, dataset_name: str = None,
                                   percentile_min: float = 2.0,
                                   percentile_max: float = 98.0) -> Dict:
        """Ultra-fast boundary scanning using memory-mapped NPY files."""
        try:
            if not self.npy_loader:
                return {
                    'status': 'error',
                    'message': 'NPY fast loader not available'
                }

            logger.info("🚀 Ultra-fast NPY boundary scanning")

            # Load dataset with memory mapping (instant!)
            start_time = time.time()
            if dataset_name:
                points, colors, metadata = self.npy_loader.load_dataset_by_name(dataset_name)
            else:
                points, colors, metadata = self.npy_loader.load_latest_dataset()

            load_time = time.time() - start_time

            if points is None:
                return {
                    'status': 'error',
                    'message': 'No NPY dataset found'
                }

            logger.info(f"✅ Memory-mapped dataset loaded in {load_time:.6f} seconds")

            # Option 1: Use pre-computed boundaries (instant!)
            if metadata and 'percentile_bounds' in metadata:
                logger.info("⚡ Using pre-computed boundaries from metadata")
                boundaries = self.npy_loader.get_boundaries_from_metadata(metadata)
                boundaries['load_time_seconds'] = load_time

                return {
                    'status': 'success',
                    'message': f'Boundaries loaded instantly from metadata (load: {load_time:.6f}s)',
                    'boundaries': boundaries
                }

            # Option 2: Compute with CUDA (still very fast)
            else:
                logger.info("Computing boundaries with CUDA...")
                if self.optimized_detector:
                    boundaries = self.optimized_detector.detect_boundaries(
                        points, percentile_min, percentile_max
                    )
                else:
                    boundaries = self.detector.detect_boundaries(
                        points, percentile_min, percentile_max
                    )

                boundaries['load_time_seconds'] = load_time

                accel_type = "CUDA-accelerated" if boundaries.get('cuda_accelerated', False) else "CPU"
                return {
                    'status': 'success',
                    'message': f'Boundaries computed with {accel_type} processing (load: {load_time:.6f}s)',
                    'boundaries': boundaries
                }

        except Exception as e:
            logger.error(f"Error in ultra-fast boundary scanning: {e}")
            return {
                'status': 'error',
                'message': f'Error: {str(e)}'
            }

    def scan_boundaries_stored(self, ply_id: int = None,
                              percentile_min: float = 2.0,
                              percentile_max: float = 98.0) -> Dict:
        """Scan boundaries from stored PLY file with CUDA acceleration."""
        try:
            # Try optimized file-based storage first
            if self.optimized_db:
                logger.info("Using optimized file-based PLY storage")
                if ply_id:
                    ply_record = self.optimized_db.get_ply_file_by_id_optimized(ply_id)
                else:
                    ply_record = self.optimized_db.get_latest_ply_file_optimized()

                if ply_record and 'file_path' in ply_record:
                    # Direct file access - MUCH faster!
                    file_path = ply_record['file_path']
                    logger.info(f"Direct file access: {file_path}")

                    # Parse PLY directly from file
                    ply_data = self.parser.parse_ply_file(file_path)
                    points = ply_data['points']

                    # Continue with boundary detection below...
                else:
                    # Fallback to database blob storage
                    logger.info("Falling back to database blob storage")
                    if ply_id:
                        ply_record = self.yard_db.get_ply_file_by_id(ply_id)
                    else:
                        ply_record = self.yard_db.get_latest_ply_file()

                    if not ply_record:
                        return {
                            'status': 'error',
                            'message': 'No PLY file found'
                        }

                    # Parse from memory blob
                    import io
                    file_stream = io.BytesIO(ply_record['file_data'])
                    ply_data = self._parse_ply_from_stream(file_stream)
                    points = ply_data['points']
            else:
                # Original database blob approach
                if ply_id:
                    ply_record = self.yard_db.get_ply_file_by_id(ply_id)
                else:
                    ply_record = self.yard_db.get_latest_ply_file()

                if not ply_record:
                    return {
                        'status': 'error',
                        'message': 'No PLY file found in database'
                    }

                logger.info(f"Using in-memory PLY parsing for {ply_record['name']}")

                # Parse directly from memory blob
                import io
                file_stream = io.BytesIO(ply_record['file_data'])
                ply_data = self._parse_ply_from_stream(file_stream)
                points = ply_data['points']

            # Use optimized CUDA boundary detection with cube projection algorithms
            if self.optimized_detector:
                logger.info(f"Using optimized CUDA boundary detection with cube projection ({len(points):,} points)")
                boundaries = self.optimized_detector.detect_boundaries(
                    points, percentile_min, percentile_max
                )
            else:
                logger.info(f"Using standard CPU boundary detection ({len(points):,} points)")
                boundaries = self.detector.detect_boundaries(
                    points, percentile_min, percentile_max
                )

            # Add acceleration info to message
            accel_type = "CUDA-accelerated" if boundaries.get('cuda_accelerated', False) else "CPU"
            message = f'Boundaries detected successfully using {accel_type} processing'

            return {
                'status': 'success',
                'message': message,
                'boundaries': boundaries
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error scanning stored boundaries: {str(e)}'
            }

    def project_yard(self, file_path: str, boundaries: Dict,
                    rotation: float = 0.0,
                    resolution: str = '1080p') -> Dict:
        """Project point cloud to create yard map."""
        try:
            # Parse resolution
            resolution_map = {
                '720p': (1280, 720),
                '1080p': (1920, 1080),
                '4k': (3840, 2160)
            }
            res_tuple = resolution_map.get(resolution, (1920, 1080))

            # Parse PLY file
            ply_data = self.parser.parse_ply_file(file_path)
            points = ply_data['points']
            colors = ply_data.get('colors')

            # Rasterize point cloud using optimized CUDA acceleration if available
            if self.optimized_rasterizer:
                logger.info(f"Using optimized CUDA rasterizer with spatial hash grid ({len(points):,} points)")
                image = self.optimized_rasterizer.rasterize_points(
                    points, boundaries, colors, resolution, rotation
                )
            else:
                logger.info(f"Using standard rasterizer ({len(points):,} points)")
                image = self.rasterizer.rasterize_point_cloud(
                    points, boundaries, rotation, res_tuple, colors
                )

            # Convert to base64 for web display
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                'status': 'success',
                'message': 'Yard map created successfully',
                'image_base64': image_base64,
                'resolution': res_tuple,
                'point_count': len(points)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error projecting yard: {str(e)}'
            }

    def save_yard_map(self, name: str, file_path: str,
                     boundaries: Dict, rotation: float,
                     resolution: str) -> Dict:
        """Save projected yard map to database."""
        try:
            # Parse resolution
            resolution_map = {
                '720p': (1280, 720),
                '1080p': (1920, 1080),
                '4k': (3840, 2160)
            }
            res_tuple = resolution_map.get(resolution, (1920, 1080))

            # Parse PLY file and create image
            ply_data = self.parser.parse_ply_file(file_path)
            points = ply_data['points']
            colors = ply_data.get('colors')

            image = self.rasterizer.rasterize_point_cloud(
                points, boundaries, rotation, res_tuple, colors
            )

            # Save to database
            success = self.yard_db.save_yard_map(
                name, image, boundaries, rotation,
                res_tuple, len(points)
            )

            if success:
                return {
                    'status': 'success',
                    'message': f'Yard map "{name}" saved successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to save yard map'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error saving yard map: {str(e)}'
            }

    def save_yard_map_from_image(self, name: str, image_data: bytes,
                                  center_x: float, center_y: float,
                                  rotation: float, resolution_x: int, resolution_y: int,
                                  algorithm: str) -> Dict:
        """Save yard map directly from image data without re-processing."""
        import sqlite3
        conn = None

        try:
            # Create boundaries dict with required fields for database
            boundaries = {
                'center_x': center_x,
                'center_z': center_y,  # Note: uses center_z in database
                'width': 20.0,  # Placeholder, actual value depends on resolution
                'height': 20.0,  # Placeholder
                'min_x': center_x - 10,
                'max_x': center_x + 10,
                'min_y': center_y - 10,
                'max_y': center_y + 10
            }

            # Save directly to database (bypass the numpy array requirement)
            conn = sqlite3.connect(self.yard_db.db_path)
            cursor = conn.cursor()
            boundaries_json = json.dumps(boundaries)

            cursor.execute('''
                INSERT OR REPLACE INTO yard_maps
                (name, image_data, boundaries, center_x, center_z, width, height,
                 rotation, resolution_x, resolution_y, point_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, image_data, boundaries_json,
                boundaries['center_x'], boundaries['center_z'],
                boundaries['width'], boundaries['height'],
                rotation, resolution_x, resolution_y, 0  # point_count = 0 for saved images
            ))

            conn.commit()

            logger.info(f"Successfully saved yard map '{name}' from image data")

            return {
                'status': 'success',
                'message': f'Yard map "{name}" saved successfully'
            }

        except Exception as e:
            logger.error(f"Error saving yard map from image: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error saving yard map: {str(e)}'
            }
        finally:
            if conn:
                conn.close()

    def get_all_maps(self) -> List[Dict]:
        """Get all saved yard maps."""
        return self.yard_db.get_all_yard_maps()

    def get_map_image(self, map_id: int) -> Optional[bytes]:
        """Get yard map image."""
        return self.yard_db.get_yard_image(map_id)

    def delete_map(self, name: str) -> bool:
        """Delete a yard map."""
        return self.yard_db.delete_yard_map(name)

    def use_yard_map(self, map_id: int) -> bool:
        """Set a yard map as 'used' for projection."""
        return self.yard_db.set_used_yard_map(map_id)

    def get_used_map(self) -> Optional[Dict]:
        """Get the currently used yard map."""
        return self.yard_db.get_used_yard_map()

    def save_projection(self, camera_name: str, map_id: int,
                       pixel_mappings: List, metadata: Dict) -> bool:
        """Save camera-to-map projection."""
        return self.yard_db.save_camera_projection(camera_name, map_id, pixel_mappings, metadata)

    def get_projection(self, camera_name: str, map_id: int) -> Optional[Dict]:
        """Get camera projection."""
        return self.yard_db.get_camera_projection(camera_name, map_id)

    def get_all_projections_for_map(self, map_id: int) -> List[Dict]:
        """Get all camera projections for a specific map."""
        return self.yard_db.get_all_projections_for_map(map_id)

    def store_ply_file(self, file_data: bytes, filename: str) -> Dict:
        """Store PLY file permanently in database."""
        try:
            # Parse the PLY file first to get metadata
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                tmp_file.write(file_data)
                tmp_file.flush()
                tmp_path = tmp_file.name

            try:
                # Parse to get metadata
                ply_data = self.parser.parse_ply_file(tmp_path)
                vertex_count = ply_data['vertex_count']
                has_color = ply_data['has_color']
                format = ply_data['format']
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # Save to database
            success = self.yard_db.save_ply_file(
                filename, file_data, vertex_count, has_color, format
            )

            if success:
                # Also save to physical file location for direct access
                os.makedirs(os.path.dirname(PLY_FILE_PATH), exist_ok=True)
                with open(PLY_FILE_PATH, 'wb') as f:
                    f.write(file_data)
                logger.info(f"PLY file saved to physical location: {PLY_FILE_PATH}")

                # Automatically convert to NPY format for ultra-fast access
                npy_result = self._convert_uploaded_ply_to_npy(filename, file_data)

                return {
                    'status': 'success',
                    'message': f'PLY file "{filename}" stored successfully and converted to ultra-fast NPY format',
                    'vertex_count': vertex_count,
                    'has_color': has_color,
                    'format': format,
                    'npy_conversion': npy_result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to store PLY file'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error storing PLY file: {str(e)}'
            }

    def _convert_uploaded_ply_to_npy(self, filename: str, file_data: bytes) -> Dict:
        """Convert uploaded PLY file to NPY format for ultra-fast access."""
        try:
            if not self.ply_converter:
                return {
                    'status': 'warning',
                    'message': 'NPY converter not available'
                }

            # Create temporary file for conversion
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                tmp_file.write(file_data)
                tmp_file.flush()
                tmp_path = tmp_file.name

            try:
                # Use fixed dataset name for overwrite behavior (single file upload)
                base_name = os.path.splitext(filename)[0]
                dataset_name = base_name  # No timestamp - this will overwrite existing data

                logger.info(f"🔄 Converting uploaded PLY to ultra-fast NPY format: {dataset_name}")

                # Convert PLY to NPY
                result = self.ply_converter.convert_ply_to_npy(tmp_path, dataset_name)

                if result and 'dataset_dir' in result:
                    logger.info(f"✅ NPY conversion successful for {filename}")
                    metadata = result.get('metadata', {})
                    return {
                        'status': 'success',
                        'message': f'Converted to NPY dataset: {dataset_name}',
                        'dataset_name': dataset_name,
                        'dataset_dir': result['dataset_dir'],
                        'point_count': metadata.get('num_points', 0),
                        'conversion_time': result.get('parse_time', 0) + result.get('save_time', 0)
                    }
                else:
                    logger.warning(f"❌ NPY conversion failed for {filename}")
                    return {
                        'status': 'warning',
                        'message': 'NPY conversion failed: Unknown error'
                    }

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error in NPY conversion: {e}")
            return {
                'status': 'error',
                'message': f'NPY conversion error: {str(e)}'
            }

    def get_latest_ply_data(self) -> Optional[Dict]:
        """Get the latest PLY file data."""
        return self.yard_db.get_latest_ply_file()

    def process_stored_ply(self, ply_id: int = None) -> Dict:
        """Process a stored PLY file."""
        try:
            # Get PLY data from database
            if ply_id:
                ply_record = self.yard_db.get_ply_file_by_id(ply_id)
            else:
                ply_record = self.yard_db.get_latest_ply_file()

            if not ply_record:
                return {
                    'status': 'error',
                    'message': 'No PLY file found in database'
                }

            # Write to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                tmp_file.write(ply_record['file_data'])
                tmp_file.flush()
                return tmp_file.name, ply_record

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing stored PLY: {str(e)}'
            }

    def project_yard_interactive(self, file_path: str = None,
                                center_x: float = 0.0,
                                center_y: float = 0.0,
                                rotation: float = 0.0,
                                resolution: float = 0.01,
                                output_size: Tuple[int, int] = (800, 600),
                                algorithm: str = 'simple_average') -> Dict:
        """
        Create interactive yard projection with CUDA acceleration.

        Args:
            file_path: Path to PLY file (None to use stored PLY)
            center_x: X coordinate of map center
            center_y: Y coordinate of map center (Z in world coordinates)
            rotation: Rotation angle in degrees
            resolution: Meters per pixel
            output_size: Output image size (width, height)
            algorithm: Rasterization algorithm ('simple_average', 'ground_filter', 'cpu_fallback', 'simple_flip', 'simple_ply')

        Returns:
            Dict with status, image data, and metadata
        """
        try:
            # Get point cloud data
            if file_path:
                # Use traditional PLY parsing for uploaded files
                ply_data = self.parser.parse_ply_file(file_path)
                points = ply_data['points']
                colors = ply_data.get('colors')
            else:
                # Use ultra-fast NPY memory-mapped loading for stored data
                logger.info("🚀 Using ultra-fast NPY memory-mapped loading")
                points, colors, metadata = self.npy_loader.load_latest_dataset()

                # DEBUG: Check color statistics right after loading
                if colors is not None:
                    logger.info(f"🐛 DEBUG: Colors after NPY load: R={colors[:,0].mean():.1f}, G={colors[:,1].mean():.1f}, B={colors[:,2].mean():.1f}")

                if points is None:
                    # Fallback to PLY if NPY not available
                    logger.warning("NPY data not available, falling back to PLY parsing")
                    ply_record = self.yard_db.get_latest_ply_file()
                    if not ply_record:
                        return {
                            'status': 'error',
                            'message': 'No point cloud data found (neither NPY nor PLY)'
                        }

                    # Write to temporary file for processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                        tmp_file.write(ply_record['file_data'])
                        tmp_file.flush()
                        tmp_path = tmp_file.name

                    try:
                        ply_data = self.parser.parse_ply_file(tmp_path)
                        points = ply_data['points']
                        colors = ply_data.get('colors')
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    logger.info(f"✅ Memory-mapped dataset loaded instantly: {len(points):,} points")

            # Check if using simple PLY algorithm (direct from file)
            if algorithm == 'simple_ply':
                logger.info("Using simple PLY rasterization (loading from physical PLY file)")
                # Force load from physical PLY file
                import trimesh
                logger.info(f"Loading from {PLY_FILE_PATH}")
                mesh = trimesh.load(PLY_FILE_PATH)
                points = mesh.vertices
                colors = mesh.visual.vertex_colors[:, :3]
                logger.info(f"Loaded {len(points):,} points from physical PLY")

                image = self._simple_ply_rasterize(
                    points=points,
                    colors=colors,
                    center_x=center_x,
                    center_y=center_y,
                    resolution=resolution,
                    output_size=output_size
                )
            # Check if using Simple Flip algorithm (XZ projection with Z-flip)
            elif algorithm == 'simple_flip':
                logger.info("Using Simple Flip rasterization (XZ projection with Z-flip)")
                # Force load from physical PLY file to avoid NPY color corruption
                import trimesh
                logger.info(f"Loading from physical PLY: {PLY_FILE_PATH}")
                mesh = trimesh.load(PLY_FILE_PATH)
                points = mesh.vertices
                colors = mesh.visual.vertex_colors[:, :3]
                logger.info(f"Loaded {len(points):,} points")
                logger.info(f"Colors dtype: {colors.dtype}, shape: {colors.shape}")
                logger.info(f"Color range: R[{colors[:,0].min()}-{colors[:,0].max()}], G[{colors[:,1].min()}-{colors[:,1].max()}], B[{colors[:,2].min()}-{colors[:,2].max()}]")
                logger.info(f"Mean colors: R={colors[:,0].mean():.1f}, G={colors[:,1].mean():.1f}, B={colors[:,2].mean():.1f}")

                image = self._simple_flip_rasterize(
                    points=points,
                    colors=colors,
                    resolution=resolution,
                    output_size=output_size
                )
            # Use optimized CUDA rasterizer for maximum performance
            elif self.optimized_rasterizer:
                # Force load from physical PLY to avoid NPY corruption
                import trimesh
                logger.info(f"Loading from physical PLY: {PLY_FILE_PATH}")
                mesh = trimesh.load(PLY_FILE_PATH)
                points = mesh.vertices
                colors = mesh.visual.vertex_colors[:, :3]
                logger.info(f"Loaded {len(points):,} points for CUDA rasterization")

                logger.info(f"Using optimized CUDA rasterizer with spatial hash grid ({len(points):,} points)")
                image = self.optimized_rasterizer.rasterize_point_cloud(
                    points=points,
                    center_x=center_x,
                    center_y=center_y,
                    rotation=rotation,
                    resolution=resolution,
                    output_size=output_size,
                    colors=colors,
                    algorithm=algorithm
                )
            else:
                logger.info(f"Using standard CUDA rasterizer ({len(points):,} points)")
                image = self.cuda_rasterizer.rasterize_point_cloud(
                    points=points,
                    center_x=center_x,
                    center_y=center_y,
                    rotation=rotation,
                    resolution=resolution,
                    output_size=output_size,
                    colors=colors
                )

            # Convert to base64 for web display
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                'status': 'success',
                'message': 'Yard map generated successfully with CUDA acceleration',
                'image_base64': image_base64,
                'width': output_size[0],
                'height': output_size[1],
                'point_count': len(points),
                'center_x': center_x,
                'center_y': center_y,
                'rotation': rotation,
                'resolution': resolution,
                'cuda_accelerated': self.cuda_rasterizer.cuda_available
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating interactive yard map: {str(e)}'
            }

    def _simple_ply_rasterize(self, points: np.ndarray, colors: np.ndarray,
                             center_x: float, center_y: float,
                             resolution: float, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Simple PLY rasterization - exact reproduction of simple_Z_flipped.png algorithm.
        XZ projection (looking down Y-axis) with Z flipped.
        """
        width, height = output_size

        # XZ projection (looking down Y-axis) with Z flipped
        coords_2d = points[:, [0, 2]].copy()
        coords_2d[:, 1] = -coords_2d[:, 1]  # Flip Z axis

        # Calculate view bounds
        half_width = (width * resolution) / 2
        half_height = (height * resolution) / 2

        # Adjust center_y since Z is flipped
        view_x_min = center_x - half_width
        view_x_max = center_x + half_width
        view_z_min = -center_y - half_height  # Negate center_y due to Z flip
        view_z_max = -center_y + half_height

        logger.info(f"Simple PLY: XZ plane, view bounds X[{view_x_min:.2f}, {view_x_max:.2f}], Z[{view_z_min:.2f}, {view_z_max:.2f}]")

        # Create image (black background)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        pixel_counts = np.zeros((height, width), dtype=np.int32)
        color_accumulator = np.zeros((height, width, 3), dtype=np.float64)

        # Rasterize
        points_in_view = 0
        for i in range(len(points)):
            x, z = coords_2d[i]

            if not (view_x_min <= x <= view_x_max and view_z_min <= z <= view_z_max):
                continue

            points_in_view += 1

            px = int((x - view_x_min) / resolution)
            py = int((z - view_z_min) / resolution)

            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))

            color_accumulator[py, px] += colors[i]
            pixel_counts[py, px] += 1

        logger.info(f"Simple PLY: {points_in_view:,} points in view")

        # Average colors
        mask = pixel_counts > 0
        for c in range(3):
            image[mask, c] = (color_accumulator[mask, c] / pixel_counts[mask]).astype(np.uint8)

        data_pixels = np.sum(mask)
        logger.info(f"Simple PLY: {data_pixels:,} data pixels ({data_pixels/(width*height)*100:.1f}%)")

        return image

    def _simple_flip_rasterize(self, points: np.ndarray, colors: np.ndarray,
                               resolution: float, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Simple Flip rasterization - XZ projection with Z-flip for correct top-down view.
        This is the algorithm that produces algorithm1.png with proper orientation.
        Uses percentile-based boundary detection and centers the view on the data.
        """
        width, height = output_size

        # XZ projection (looking down Y-axis) with Z flipped
        coords_2d = points[:, [0, 2]].copy()
        coords_2d[:, 1] = -coords_2d[:, 1]  # Flip Z axis for correct orientation

        # Find bounds using percentiles to exclude outliers
        x_min, x_max = np.percentile(coords_2d[:, 0], [2, 98])
        z_min, z_max = np.percentile(coords_2d[:, 1], [2, 98])

        logger.info(f"Simple Flip: Bounds X[{x_min:.2f}, {x_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")

        # Calculate center
        center_x = (x_min + x_max) / 2
        center_z = (z_min + z_max) / 2

        # Calculate view bounds based on image size and resolution
        half_width = (width * resolution) / 2
        half_height = (height * resolution) / 2

        view_x_min = center_x - half_width
        view_x_max = center_x + half_width
        view_z_min = center_z - half_height
        view_z_max = center_z + half_height

        logger.info(f"Simple Flip: View bounds X[{view_x_min:.2f}, {view_x_max:.2f}], Z[{view_z_min:.2f}, {view_z_max:.2f}]")

        # Create image (black background)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        pixel_counts = np.zeros((height, width), dtype=np.int32)
        color_accumulator = np.zeros((height, width, 3), dtype=np.float64)

        # Rasterize: map each point to a pixel
        points_in_view = 0
        for i in range(len(points)):
            x, z = coords_2d[i]

            if not (view_x_min <= x <= view_x_max and view_z_min <= z <= view_z_max):
                continue

            points_in_view += 1

            # Map to pixel coordinates
            px = int((x - view_x_min) / resolution)
            py = int((z - view_z_min) / resolution)

            # Clamp to image bounds
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))

            # Accumulate color
            color_accumulator[py, px] += colors[i]
            pixel_counts[py, px] += 1

        logger.info(f"Simple Flip: {points_in_view:,} points in view ({points_in_view/len(points)*100:.1f}%)")

        # Average colors
        mask = pixel_counts > 0
        for c in range(3):
            image[mask, c] = (color_accumulator[mask, c] / pixel_counts[mask]).astype(np.uint8)

        data_pixels = np.sum(mask)
        logger.info(f"Simple Flip: {data_pixels:,} data pixels ({data_pixels/(width*height)*100:.1f}%)")

        # Log output image statistics
        if data_pixels > 0:
            data_colors = image[mask]
            logger.info(f"Output image colors: R={data_colors[:,0].mean():.1f}, G={data_colors[:,1].mean():.1f}, B={data_colors[:,2].mean():.1f}")
            logger.info(f"Output image dtype: {image.dtype}, shape: {image.shape}")

        return image

    def rasterize_ultra_fast(self, points: np.ndarray, colors: np.ndarray = None,
                            center_x: float = 0.0, center_y: float = 0.0,
                            rotation: float = 0.0, resolution: float = 0.01,
                            output_size: Tuple[int, int] = (800, 600)) -> Dict:
        """
        Ultra-fast rasterization using memory-mapped arrays directly.

        Args:
            points: Memory-mapped NumPy array of point coordinates
            colors: Memory-mapped NumPy array of point colors (optional)
            center_x: X coordinate of map center
            center_y: Y coordinate of map center (Z in world coordinates)
            rotation: Rotation angle in degrees
            resolution: Meters per pixel
            output_size: Output image size (width, height)

        Returns:
            Dict with status, image data, and metadata
        """
        try:
            start_time = time.time()

            logger.info(f"🚀 Ultra-fast rasterization with memory-mapped arrays ({len(points):,} points)")

            # Use optimized CUDA rasterizer with memory-mapped data
            if self.optimized_rasterizer:
                image = self.optimized_rasterizer.rasterize_point_cloud(
                    points=points,
                    center_x=center_x,
                    center_y=center_y,
                    rotation=rotation,
                    resolution=resolution,
                    output_size=output_size,
                    colors=colors
                )
            else:
                # Fallback to standard rasterizer
                image = self.cuda_rasterizer.rasterize_point_cloud(
                    points=points,
                    center_x=center_x,
                    center_y=center_y,
                    rotation=rotation,
                    resolution=resolution,
                    output_size=output_size,
                    colors=colors
                )

            # Convert to base64 for web display
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            raster_time = time.time() - start_time

            logger.info(f"✅ Ultra-fast rasterization completed in {raster_time:.6f} seconds")

            return {
                'status': 'success',
                'message': f'Ultra-fast yard map generated in {raster_time:.6f} seconds',
                'image_base64': image_base64,
                'width': output_size[0],
                'height': output_size[1],
                'point_count': len(points),
                'center_x': center_x,
                'center_y': center_y,
                'rotation': rotation,
                'resolution': resolution,
                'cuda_accelerated': True,
                'memory_mapped': True,
                'raster_time_seconds': raster_time
            }

        except Exception as e:
            logger.error(f"Error in ultra-fast rasterization: {e}")
            return {
                'status': 'error',
                'message': f'Error in ultra-fast rasterization: {str(e)}'
            }