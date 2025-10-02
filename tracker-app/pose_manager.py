"""
COLMAP pose extraction and camera calibration management.
Handles parsing of COLMAP binary files and extraction of camera poses.
"""

import struct
import numpy as np
import json
import os
import sqlite3
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ColmapBinaryReader:
    """Handles reading and parsing of COLMAP binary files."""

    @staticmethod
    def read_cameras_binary(path: str) -> Dict:
        """
        Parse COLMAP cameras.bin file.

        Returns:
            Dict: {camera_id: {model_id, width, height, params}}
        """
        cameras = {}

        if not os.path.exists(path):
            raise FileNotFoundError(f"Cameras file not found: {path}")

        with open(path, "rb") as file:
            num_cameras = struct.unpack("<Q", file.read(8))[0]

            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", file.read(4))[0]
                model_id = struct.unpack("<I", file.read(4))[0]
                width = struct.unpack("<Q", file.read(8))[0]
                height = struct.unpack("<Q", file.read(8))[0]

                # Read camera parameters (variable length based on model)
                params = []
                if model_id in [0, 1]:  # SIMPLE_PINHOLE, PINHOLE
                    num_params = 4 if model_id == 1 else 3
                elif model_id in [2, 3]:  # SIMPLE_RADIAL, RADIAL
                    num_params = 4 if model_id == 2 else 5
                elif model_id == 4:  # OPENCV
                    num_params = 8
                else:
                    num_params = 8  # Default fallback

                for _ in range(num_params):
                    params.append(struct.unpack("<d", file.read(8))[0])

                cameras[camera_id] = {
                    'model_id': model_id,
                    'width': width,
                    'height': height,
                    'params': params
                }

        return cameras

    @staticmethod
    def read_images_binary(path: str) -> Dict:
        """
        Parse COLMAP images.bin file.

        Returns:
            Dict: {image_id: {camera_id, name, qvec, tvec, points2D}}
        """
        images = {}

        if not os.path.exists(path):
            raise FileNotFoundError(f"Images file not found: {path}")

        with open(path, "rb") as file:
            num_reg_images = struct.unpack("<Q", file.read(8))[0]

            for _ in range(num_reg_images):
                image_id = struct.unpack("<I", file.read(4))[0]

                # Read quaternion (4 doubles)
                qvec = struct.unpack("<dddd", file.read(32))

                # Read translation (3 doubles)
                tvec = struct.unpack("<ddd", file.read(24))

                camera_id = struct.unpack("<I", file.read(4))[0]

                # Read image name
                name_length = 0
                name_bytes = b""
                while True:
                    char = file.read(1)
                    if char == b'\x00':
                        break
                    name_bytes += char
                    name_length += 1

                name = name_bytes.decode('utf-8')

                # Read 2D points
                num_points2D = struct.unpack("<Q", file.read(8))[0]
                points2D = []
                for _ in range(num_points2D):
                    x = struct.unpack("<d", file.read(8))[0]
                    y = struct.unpack("<d", file.read(8))[0]
                    point3D_id = struct.unpack("<Q", file.read(8))[0]
                    points2D.append((x, y, point3D_id))

                images[image_id] = {
                    'camera_id': camera_id,
                    'name': name,
                    'qvec': qvec,
                    'tvec': tvec,
                    'points2D': points2D
                }

        return images


class CameraPoseExtractor:
    """Extracts camera poses and calibration data from COLMAP results."""

    @staticmethod
    def quaternion_to_rotation_matrix(qvec: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        qw, qx, qy, qz = qvec

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    @staticmethod
    def extract_intrinsics(camera_data: Dict) -> Dict:
        """Extract camera intrinsic parameters."""
        model_id = camera_data['model_id']
        params = camera_data['params']
        width = camera_data['width']
        height = camera_data['height']

        intrinsics = {
            'width': width,
            'height': height,
            'model_id': model_id
        }

        if model_id == 0:  # SIMPLE_PINHOLE
            f, cx, cy = params[:3]
            intrinsics.update({
                'fx': f, 'fy': f, 'cx': cx, 'cy': cy,
                'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0
            })
        elif model_id == 1:  # PINHOLE
            fx, fy, cx, cy = params[:4]
            intrinsics.update({
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0
            })
        elif model_id == 2:  # SIMPLE_RADIAL
            f, cx, cy, k = params[:4]
            intrinsics.update({
                'fx': f, 'fy': f, 'cx': cx, 'cy': cy,
                'k1': k, 'k2': 0, 'p1': 0, 'p2': 0
            })
        elif model_id == 3:  # RADIAL
            f, cx, cy, k1, k2 = params[:5]
            intrinsics.update({
                'fx': f, 'fy': f, 'cx': cx, 'cy': cy,
                'k1': k1, 'k2': k2, 'p1': 0, 'p2': 0
            })
        elif model_id == 4:  # OPENCV
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            intrinsics.update({
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
            })

        # Calculate field of view
        if 'fx' in intrinsics and 'fy' in intrinsics:
            fov_x = 2 * np.arctan(width / (2 * intrinsics['fx'])) * 180 / np.pi
            fov_y = 2 * np.arctan(height / (2 * intrinsics['fy'])) * 180 / np.pi
            intrinsics.update({'fov_x': fov_x, 'fov_y': fov_y})

        return intrinsics

    @staticmethod
    def extract_extrinsics(image_data: Dict) -> Dict:
        """Extract camera extrinsic parameters (pose)."""
        qvec = image_data['qvec']
        tvec = image_data['tvec']

        # Convert quaternion to rotation matrix
        R = CameraPoseExtractor.quaternion_to_rotation_matrix(qvec)
        t = np.array(tvec)

        # Create camera-to-world transformation matrix
        T_cw = np.eye(4)
        T_cw[:3, :3] = R.T  # Transpose for camera-to-world
        T_cw[:3, 3] = -R.T @ t  # Camera center in world coordinates

        # Create world-to-camera transformation matrix
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t

        return {
            'rotation_matrix': R.tolist(),
            'translation_vector': t.tolist(),
            'quaternion': qvec,
            'camera_to_world': T_cw.tolist(),
            'world_to_camera': T_wc.tolist()
        }

    @staticmethod
    def match_camera_names(image_names: List[str], camera_configs: Dict) -> Dict[str, str]:
        """
        Match COLMAP image names to configured camera names.

        Returns:
            Dict: {image_name: camera_name}
        """
        matches = {}
        configured_names = list(camera_configs.keys())

        for image_name in image_names:
            # Remove file extension
            base_name = os.path.splitext(image_name)[0].lower()

            # Try exact match first
            for camera_name in configured_names:
                if base_name == camera_name.lower():
                    matches[image_name] = camera_name
                    break

            # Try partial match
            if image_name not in matches:
                for camera_name in configured_names:
                    if camera_name.lower() in base_name or base_name in camera_name.lower():
                        matches[image_name] = camera_name
                        break

        return matches


class PoseDatabase:
    """Manages camera pose data storage and retrieval."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the camera poses database table."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS camera_poses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT UNIQUE NOT NULL,
                    intrinsics TEXT NOT NULL,
                    extrinsics TEXT NOT NULL,
                    fov_x REAL,
                    fov_y REAL,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def save_camera_pose(self, camera_name: str, intrinsics: Dict, extrinsics: Dict) -> bool:
        """Save camera pose data to database."""
        try:
            conn = sqlite3.connect(self.db_path)

            intrinsics_json = json.dumps(intrinsics)
            extrinsics_json = json.dumps(extrinsics)
            fov_x = intrinsics.get('fov_x')
            fov_y = intrinsics.get('fov_y')
            width = intrinsics.get('width')
            height = intrinsics.get('height')

            conn.execute('''
                INSERT OR REPLACE INTO camera_poses
                (camera_name, intrinsics, extrinsics, fov_x, fov_y, image_width, image_height, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (camera_name, intrinsics_json, extrinsics_json, fov_x, fov_y, width, height, datetime.now().isoformat()))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving camera pose for {camera_name}: {e}")
            return False
        finally:
            conn.close()

    def get_camera_pose(self, camera_name: str) -> Optional[Dict]:
        """Get camera pose data for a specific camera."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                'SELECT * FROM camera_poses WHERE camera_name = ?', (camera_name,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'camera_name': row[1],
                    'intrinsics': json.loads(row[2]),
                    'extrinsics': json.loads(row[3]),
                    'fov_x': row[4],
                    'fov_y': row[5],
                    'image_width': row[6],
                    'image_height': row[7],
                    'created_at': row[8],
                    'updated_at': row[9]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting camera pose for {camera_name}: {e}")
            return None
        finally:
            conn.close()

    def get_all_poses(self) -> List[Dict]:
        """Get all saved camera poses."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT * FROM camera_poses ORDER BY camera_name')
            rows = cursor.fetchall()

            poses = []
            for row in rows:
                poses.append({
                    'id': row[0],
                    'camera_name': row[1],
                    'intrinsics': json.loads(row[2]),
                    'extrinsics': json.loads(row[3]),
                    'fov_x': row[4],
                    'fov_y': row[5],
                    'image_width': row[6],
                    'image_height': row[7],
                    'created_at': row[8],
                    'updated_at': row[9]
                })
            return poses
        except Exception as e:
            logger.error(f"Error getting all camera poses: {e}")
            return []
        finally:
            conn.close()

    def delete_camera_pose(self, camera_name: str) -> bool:
        """Delete camera pose data."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM camera_poses WHERE camera_name = ?', (camera_name,))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting camera pose for {camera_name}: {e}")
            return False
        finally:
            conn.close()


class PoseManager:
    """Main class for managing camera pose extraction and storage."""

    def __init__(self, db_path: str = 'poses.db'):
        self.pose_db = PoseDatabase(db_path)
        self.reader = ColmapBinaryReader()
        self.extractor = CameraPoseExtractor()

    def process_colmap_files(self, cameras_path: str, images_path: str, camera_configs: Dict) -> Dict:
        """
        Process COLMAP binary files and extract camera poses.

        Returns:
            Dict: {status, message, poses}
        """
        try:
            # Read COLMAP files
            cameras_data = self.reader.read_cameras_binary(cameras_path)
            images_data = self.reader.read_images_binary(images_path)

            # Extract image names for matching
            image_names = [img['name'] for img in images_data.values()]

            # Match image names to camera names
            name_matches = self.extractor.match_camera_names(image_names, camera_configs)

            if not name_matches:
                return {
                    'status': 'error',
                    'message': 'No matching camera names found in COLMAP data',
                    'poses': {}
                }

            # Extract poses for matched cameras
            extracted_poses = {}

            for image_data in images_data.values():
                image_name = image_data['name']
                if image_name in name_matches:
                    camera_name = name_matches[image_name]
                    camera_id = image_data['camera_id']

                    if camera_id in cameras_data:
                        # Extract intrinsics and extrinsics
                        intrinsics = self.extractor.extract_intrinsics(cameras_data[camera_id])
                        extrinsics = self.extractor.extract_extrinsics(image_data)

                        extracted_poses[camera_name] = {
                            'intrinsics': intrinsics,
                            'extrinsics': extrinsics,
                            'source_image': image_name
                        }

            return {
                'status': 'success',
                'message': f'Extracted poses for {len(extracted_poses)} cameras',
                'poses': extracted_poses,
                'matches': name_matches
            }

        except Exception as e:
            logger.error(f"Error processing COLMAP files: {e}")
            return {
                'status': 'error',
                'message': f'Error processing COLMAP files: {str(e)}',
                'poses': {}
            }

    def save_poses(self, poses_data: Dict) -> Dict:
        """Save extracted poses to database."""
        saved_count = 0
        failed_cameras = []

        for camera_name, pose_data in poses_data.items():
            success = self.pose_db.save_camera_pose(
                camera_name,
                pose_data['intrinsics'],
                pose_data['extrinsics']
            )

            if success:
                saved_count += 1
            else:
                failed_cameras.append(camera_name)

        if failed_cameras:
            return {
                'status': 'partial',
                'message': f'Saved {saved_count} poses, failed: {", ".join(failed_cameras)}',
                'saved_count': saved_count,
                'failed_cameras': failed_cameras
            }
        else:
            return {
                'status': 'success',
                'message': f'Successfully saved poses for {saved_count} cameras',
                'saved_count': saved_count
            }

    def get_camera_pose(self, camera_name: str) -> Optional[Dict]:
        """Get camera pose data for a specific camera."""
        return self.pose_db.get_camera_pose(camera_name)

    def get_all_poses(self) -> List[Dict]:
        """Get all saved camera poses."""
        return self.pose_db.get_all_poses()

    def delete_pose(self, camera_name: str) -> bool:
        """Delete a camera pose."""
        return self.pose_db.delete_camera_pose(camera_name)