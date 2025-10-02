"""
Camera-to-Map Projection Module

Ray-traces camera pixels to ground plane using pose information and point cloud data.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)


class CameraProjector:
    """Projects camera pixels onto yard map using ray tracing."""

    def __init__(self, point_cloud_path: str):
        """
        Initialize projector with point cloud data.

        Args:
            point_cloud_path: Path to PLY file
        """
        self.point_cloud_path = point_cloud_path
        self.points = None
        self.colors = None
        self._load_point_cloud()

    def _load_point_cloud(self):
        """Load point cloud from PLY file."""
        try:
            import trimesh
            mesh = trimesh.load(self.point_cloud_path)
            self.points = mesh.vertices
            if hasattr(mesh.visual, 'vertex_colors'):
                self.colors = mesh.visual.vertex_colors[:, :3]
            logger.info(f"Loaded {len(self.points)} points from point cloud")
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
        Project camera pixels onto map using ray tracing.

        Args:
            camera_pose: Dict with camera position and orientation
                        (position_x, position_y, position_z, rotation matrix)
            map_info: Dict with map boundaries, center, resolution
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels
            sample_rate: Sample every Nth pixel (for performance)

        Returns:
            Dict with projection results:
                - pixel_mappings: List of (camera_x, camera_y, map_x, map_y) tuples
                - projected_pixels: List of [map_x, map_y] for visualization
                - bounds: Dict with projection bounding box
                - pixel_count: Number of successful mappings
                - coverage_percent: Percentage of map covered
                - compute_time: Time taken
        """
        start_time = time.time()

        logger.info(f"Starting camera projection: {camera_width}x{camera_height}")
        logger.info(f"Map info: {map_info['resolution_x']}x{map_info['resolution_y']}")

        # Extract camera parameters
        camera_pos = np.array([
            camera_pose['position_x'],
            camera_pose['position_y'],
            camera_pose['position_z']
        ])

        # Get rotation matrix from camera pose
        # COLMAP stores as quaternion or rotation matrix
        rotation_matrix = self._get_rotation_matrix(camera_pose)

        # Get camera intrinsics from COLMAP calibration
        intrinsics = camera_pose.get('intrinsics', {})
        if intrinsics and 'fx' in intrinsics:
            # Use actual calibrated intrinsics
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']
            logger.info(f"Using calibrated intrinsics: fx={fx:.2f}, fy={fy:.2f}, fov_x={intrinsics.get('fov_x', 0):.1f}°, fov_y={intrinsics.get('fov_y', 0):.1f}°")
        else:
            # Fallback to estimation if intrinsics not available
            logger.warning("No intrinsics found, using approximation (less accurate)")
            fx = fy = camera_width * 1.2  # Approximate FOV
            cx = camera_width / 2
            cy = camera_height / 2

        # Build spatial grid for ground points
        logger.info("Building spatial grid for ground plane...")
        ground_grid = self._build_ground_grid(map_info)

        # Get boundaries for logging
        boundaries = map_info['boundaries']

        # Ray trace sampled pixels
        pixel_mappings = []
        projected_pixels = []

        # Debug counters
        rays_cast = 0
        rays_intersected = 0
        rays_out_of_bounds = 0

        logger.info(f"Ray tracing pixels (sampling every {sample_rate})...")
        logger.info(f"Map bounds: X=[{boundaries['min_x']:.2f}, {boundaries['max_x']:.2f}], Z=[{boundaries['min_y']:.2f}, {boundaries['max_y']:.2f}]")
        logger.info(f"Camera position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")

        for cam_y in range(0, camera_height, sample_rate):
            for cam_x in range(0, camera_width, sample_rate):
                rays_cast += 1

                # Compute ray direction in camera space using pinhole camera model
                # Ray direction = (x - cx)/fx, (y - cy)/fy, 1.0 (normalized)
                ray_dir_cam = np.array([
                    (cam_x - cx) / fx,
                    (cam_y - cy) / fy,
                    1.0
                ])
                ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

                # Transform ray to world space
                ray_dir_world = rotation_matrix @ ray_dir_cam

                # Find intersection with ground
                intersection = self._ray_ground_intersection(
                    camera_pos,
                    ray_dir_world,
                    ground_grid,
                    map_info
                )

                if intersection is not None:
                    rays_intersected += 1

                    # Convert world coordinates to map pixel coordinates
                    map_pixel = self._world_to_map_pixel(intersection, map_info)

                    if map_pixel is not None:
                        pixel_mappings.append((cam_x, cam_y, map_pixel[0], map_pixel[1]))
                        projected_pixels.append([int(map_pixel[0]), int(map_pixel[1])])
                    else:
                        rays_out_of_bounds += 1

        # Compute bounds
        total_rays = ((camera_height + sample_rate - 1) // sample_rate) * ((camera_width + sample_rate - 1) // sample_rate)

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

            # Calculate coverage
            map_total_pixels = map_info['resolution_x'] * map_info['resolution_y']
            coverage_percent = (unique_pixel_count / map_total_pixels) * 100

            # Log statistics
            logger.info(f"Ray tracing results:")
            logger.info(f"  Rays cast: {rays_cast:,}")
            logger.info(f"  Rays intersected ground: {rays_intersected:,} ({100*rays_intersected/rays_cast:.1f}%)")
            logger.info(f"  Rays out of bounds: {rays_out_of_bounds:,}")
            logger.info(f"  Final mapped pixels: {len(pixel_mappings):,} ({100*len(pixel_mappings)/rays_cast:.1f}%)")
            logger.info(f"Projection bounds on map:")
            logger.info(f"  X: [{bounds['min_x']}, {bounds['max_x']}] = {bounds['max_x']-bounds['min_x']+1} pixels wide")
            logger.info(f"  Y: [{bounds['min_y']}, {bounds['max_y']}] = {bounds['max_y']-bounds['min_y']+1} pixels tall")
            logger.info(f"Unique map pixels covered: {unique_pixel_count:,}")

        else:
            bounds = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
            coverage_percent = 0.0
            unique_map_pixels = []
            logger.info(f"Rays cast: {total_rays:,}, Hits: 0 (0.0%)")
            logger.warning("No rays hit the ground - check camera pose and map alignment")

        compute_time = time.time() - start_time

        logger.info(f"Projection complete in {compute_time:.2f}s")
        logger.info(f"Coverage: {coverage_percent:.2f}%")

        return {
            'pixel_mappings': pixel_mappings,
            'projected_pixels': unique_map_pixels,  # Return only unique pixels for visualization
            'bounds': bounds,
            'pixel_count': len(pixel_mappings),
            'coverage_percent': round(coverage_percent, 2),
            'compute_time': round(compute_time, 2)
        }

    def _get_rotation_matrix(self, camera_pose: Dict) -> np.ndarray:
        """
        Extract rotation matrix from camera pose.

        COLMAP stores camera orientation as rotation matrix or quaternion.
        """
        if 'rotation_matrix' in camera_pose:
            return np.array(camera_pose['rotation_matrix']).reshape(3, 3)
        elif 'quaternion' in camera_pose:
            # Convert quaternion to rotation matrix
            qw, qx, qy, qz = camera_pose['quaternion']
            return self._quaternion_to_matrix(qw, qx, qy, qz)
        else:
            # Default to identity
            logger.warning("No rotation information found, using identity matrix")
            return np.eye(3)

    def _quaternion_to_matrix(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def _build_ground_grid(self, map_info: Dict) -> Dict:
        """
        Build spatial grid of ground points from point cloud.

        Ground points are defined as the bottom 40% of points in each map pixel column.
        """
        # Get map boundaries
        boundaries = map_info['boundaries']
        min_x = boundaries['min_x']
        max_x = boundaries['max_x']
        min_z = boundaries['min_y']  # Z in world corresponds to Y in map
        max_z = boundaries['max_y']

        # Filter points within map boundaries
        mask = (
            (self.points[:, 0] >= min_x) &
            (self.points[:, 0] <= max_x) &
            (self.points[:, 2] >= min_z) &
            (self.points[:, 2] <= max_z)
        )
        map_points = self.points[mask]

        logger.info(f"Found {len(map_points)} points within map boundaries")

        # Build grid: divide map into cells and find ground points
        resolution_x = map_info['resolution_x']
        resolution_y = map_info['resolution_y']

        cell_size_x = (max_x - min_x) / resolution_x
        cell_size_z = (max_z - min_z) / resolution_y

        grid = {}

        for point in map_points:
            # Determine grid cell
            cell_x = int((point[0] - min_x) / cell_size_x)
            cell_z = int((point[2] - min_z) / cell_size_z)

            cell_x = max(0, min(cell_x, resolution_x - 1))
            cell_z = max(0, min(cell_z, resolution_y - 1))

            key = (cell_x, cell_z)

            if key not in grid:
                grid[key] = []

            grid[key].append(point)

        # For each cell, keep only bottom 40% of points (ground)
        ground_grid = {}
        for key, cell_points in grid.items():
            if len(cell_points) > 0:
                cell_points_arr = np.array(cell_points)
                y_values = cell_points_arr[:, 1]

                # Get 40th percentile (bottom 40%)
                y_threshold = np.percentile(y_values, 40)

                ground_points = cell_points_arr[y_values <= y_threshold]
                ground_grid[key] = ground_points

        logger.info(f"Built ground grid with {len(ground_grid)} cells")

        return ground_grid

    def _ray_ground_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        ground_grid: Dict,
        map_info: Dict
    ) -> Optional[np.ndarray]:
        """
        Find intersection of ray with ground points.

        Returns world coordinates of intersection point, or None if no intersection.
        """
        # March along ray and check for ground intersection
        max_distance = 50.0  # meters
        step_size = 0.5  # meters (increased for performance)

        boundaries = map_info['boundaries']
        min_x = boundaries['min_x']
        max_x = boundaries['max_x']
        min_z = boundaries['min_y']
        max_z = boundaries['max_y']

        resolution_x = map_info['resolution_x']
        resolution_y = map_info['resolution_y']
        cell_size_x = (max_x - min_x) / resolution_x
        cell_size_z = (max_z - min_z) / resolution_y

        for distance in np.arange(0, max_distance, step_size):
            point = ray_origin + ray_dir * distance

            # Check if point is within map bounds - skip if outside
            if not (min_x <= point[0] <= max_x and min_z <= point[2] <= max_z):
                continue

            # Determine grid cell (guaranteed to be in bounds due to check above)
            cell_x = int((point[0] - min_x) / cell_size_x)
            cell_z = int((point[2] - min_z) / cell_size_z)

            # Safety bounds check (should not be needed but prevents edge case crashes)
            if cell_x < 0 or cell_x >= resolution_x or cell_z < 0 or cell_z >= resolution_y:
                continue

            key = (cell_x, cell_z)

            # Check if ground points exist in this cell
            if key in ground_grid:
                ground_points = ground_grid[key]

                # Check if ray point is close to any ground point
                distances = np.linalg.norm(ground_points - point, axis=1)
                min_dist = distances.min()

                if min_dist < 0.2:  # Within 20cm of ground
                    return point

        # Ray did not intersect ground within map boundaries
        return None

    def _world_to_map_pixel(
        self,
        world_pos: np.ndarray,
        map_info: Dict
    ) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates to map pixel coordinates.

        Applies 90° clockwise rotation to match yard map orientation.
        """
        boundaries = map_info['boundaries']
        min_x = boundaries['min_x']
        max_x = boundaries['max_x']
        min_z = boundaries['min_y']
        max_z = boundaries['max_y']

        resolution_x = map_info['resolution_x']
        resolution_y = map_info['resolution_y']

        # Apply 90° clockwise rotation:
        # new_x = Z
        # new_y = -X
        rotated_x = world_pos[2]
        rotated_y = -world_pos[0]

        # Map rotated coordinates to pixel coordinates
        pixel_x = int((rotated_x - min_z) / (max_z - min_z) * resolution_x)
        pixel_y = int((rotated_y - (-max_x)) / ((-min_x) - (-max_x)) * resolution_y)

        # Validate pixel is within bounds - return None if outside
        # This prevents clustering at the edges
        if pixel_x < 0 or pixel_x >= resolution_x or pixel_y < 0 or pixel_y >= resolution_y:
            return None

        return (pixel_x, pixel_y)
