"""
Toddler Position Tracking System (TPTS)
Maps real-time toddler detections from camera pixels to yard map coordinates.
"""

import sqlite3
import numpy as np
import pickle
import logging
from scipy.spatial import cKDTree
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import requests
from database import MatchesDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks toddler positions by converting camera detections to yard map coordinates.

    Data Flow:
    1. Receive Frigate person detection (normalized bbox)
    2. Convert to pixel coordinates
    3. Extract feet position (bottom-center)
    4. Lookup map coordinates using pre-computed projection
    5. Store position in database
    """

    def __init__(self, yard_db_path: str = "yard.db", matches_db_path: str = "matches.db"):
        self.yard_db_path = yard_db_path
        self.matches_db_path = matches_db_path
        self.matches_db = MatchesDatabase(matches_db_path)
        self.projection_cache: Dict[str, Dict] = {}  # Cache KD-Trees per camera

    def convert_bbox_to_pixels(self, frigate_bbox: List[float], camera_width: int, camera_height: int) -> List[int]:
        """
        Convert Frigate normalized bounding box to pixel coordinates.

        Args:
            frigate_bbox: [x_center, y_center, width, height] normalized (0-1)
            camera_width: Camera resolution width
            camera_height: Camera resolution height

        Returns:
            [x, y, width, height] in pixels (top-left corner format)
        """
        x_center_norm, y_center_norm, width_norm, height_norm = frigate_bbox

        # Convert to pixel coordinates (top-left)
        x_px = int((x_center_norm - width_norm/2) * camera_width)
        y_px = int((y_center_norm - height_norm/2) * camera_height)
        w_px = int(width_norm * camera_width)
        h_px = int(height_norm * camera_height)

        return [x_px, y_px, w_px, h_px]

    def get_feet_position(self, frigate_bbox: List[float], camera_width: int, camera_height: int) -> Tuple[int, int]:
        """
        Extract feet position (where toddler is standing) from bounding box.
        Uses bottom-center of bbox as this maps to ground plane.

        Args:
            frigate_bbox: [x_center, y_center, width, height] normalized (0-1)
            camera_width: Camera resolution width
            camera_height: Camera resolution height

        Returns:
            (cam_x, cam_y) pixel coordinates of feet position
        """
        x_center_norm, y_center_norm, width_norm, height_norm = frigate_bbox

        # Bottom-center of bounding box (feet position)
        cam_x = int(x_center_norm * camera_width)
        cam_y = int((y_center_norm + height_norm/2) * camera_height)

        return cam_x, cam_y

    def _load_projection_cache(self, camera_name: str, map_id: int) -> Optional[Dict]:
        """
        Load camera projection data and build KD-Tree for fast lookups.
        Results are cached for performance.

        Args:
            camera_name: Name of camera
            map_id: ID of yard map

        Returns:
            Dict with 'tree' (KD-Tree) and 'map_coords' (numpy array)
        """
        cache_key = f"{camera_name}_{map_id}"

        # Return cached version if available
        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]

        # Load from database
        try:
            conn = sqlite3.connect(self.yard_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT pixel_mappings FROM camera_projections
                WHERE camera_name = ? AND map_id = ?
            """, (camera_name, map_id))

            row = cursor.fetchone()
            conn.close()

            if not row:
                logger.warning(f"No projection found for camera {camera_name}, map {map_id}")
                return None

            # Unpickle pixel mappings: [(cam_x, cam_y, map_x, map_y), ...]
            pixel_mappings = pickle.loads(row[0])

            if not pixel_mappings:
                logger.warning(f"Empty projection for camera {camera_name}, map {map_id}")
                return None

            # Convert to numpy arrays
            mappings_array = np.array(pixel_mappings)
            cam_coords = mappings_array[:, :2]  # (cam_x, cam_y)
            map_coords = mappings_array[:, 2:]  # (map_x, map_y)

            # Build KD-Tree for fast nearest neighbor search
            tree = cKDTree(cam_coords)

            # Cache the result
            self.projection_cache[cache_key] = {
                'tree': tree,
                'map_coords': map_coords,
                'cam_coords': cam_coords
            }

            logger.info(f"Loaded projection for {camera_name}: {len(pixel_mappings)} mappings")
            return self.projection_cache[cache_key]

        except Exception as e:
            logger.error(f"Error loading projection: {e}")
            return None

    def lookup_map_position(self, camera_name: str, map_id: int, cam_x: int, cam_y: int,
                           max_distance: int = 50) -> Optional[Tuple[int, int]]:
        """
        Find map coordinates for given camera pixel position using nearest neighbor search.

        Args:
            camera_name: Name of camera
            map_id: ID of yard map
            cam_x: Camera pixel x coordinate
            cam_y: Camera pixel y coordinate
            max_distance: Maximum pixel distance for valid match

        Returns:
            (map_x, map_y) or None if no valid mapping found
        """
        # Load projection data (uses cache)
        projection = self._load_projection_cache(camera_name, map_id)

        if not projection:
            return None

        tree = projection['tree']
        map_coords = projection['map_coords']

        # Find nearest camera pixel
        distance, idx = tree.query([cam_x, cam_y])

        # Threshold to avoid bad matches
        if distance > max_distance:
            logger.warning(f"Nearest mapping too far: {distance:.1f} pixels > {max_distance}")
            return None

        map_x, map_y = map_coords[idx]
        logger.debug(f"Mapped ({cam_x}, {cam_y}) -> ({map_x}, {map_y}), distance={distance:.1f}px")

        return int(map_x), int(map_y)

    def store_toddler_position(self, subject: str, camera: str, map_x: int, map_y: int,
                              confidence: float) -> bool:
        """
        Store toddler position in database.

        Args:
            subject: Name of subject (e.g., "Toddler")
            camera: Camera name
            map_x: X coordinate on yard map
            map_y: Y coordinate on yard map
            confidence: Detection confidence (0-1)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use MatchesDatabase to store position (it handles timestamp automatically)
            position_id = self.matches_db.add_toddler_position(
                subject=subject,
                camera=camera,
                map_x=map_x,
                map_y=map_y,
                confidence=confidence,
                timestamp=None  # Let database generate timestamp
            )

            if position_id:
                logger.info(f"Stored position: {subject} at ({map_x}, {map_y}) from {camera}, conf={confidence:.2f}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error storing position: {e}")
            return False

    def process_detection(self, camera_name: str, map_id: int, frigate_bbox: List[float],
                         camera_width: int, camera_height: int, subject: str,
                         confidence: float) -> Optional[Tuple[int, int]]:
        """
        Complete pipeline: convert detection to map position and store.

        Args:
            camera_name: Name of camera
            map_id: ID of yard map to use
            frigate_bbox: [x_center, y_center, width, height] normalized (0-1)
            camera_width: Camera resolution width
            camera_height: Camera resolution height
            subject: Name of detected subject
            confidence: Detection confidence

        Returns:
            (map_x, map_y) if successful, None otherwise
        """
        # Extract feet position
        cam_x, cam_y = self.get_feet_position(frigate_bbox, camera_width, camera_height)
        logger.debug(f"Feet position: ({cam_x}, {cam_y})")

        # Lookup map coordinates
        map_position = self.lookup_map_position(camera_name, map_id, cam_x, cam_y)

        if not map_position:
            logger.warning(f"Could not map position for {camera_name} at ({cam_x}, {cam_y})")
            return None

        map_x, map_y = map_position

        # Store in database
        success = self.store_toddler_position(subject, camera_name, map_x, map_y, confidence)

        if success:
            return map_position
        else:
            return None

    def get_frigate_events(self, frigate_url: str = "http://localhost:5000",
                          label: str = "person", camera: Optional[str] = None,
                          limit: int = 10) -> List[Dict]:
        """
        Fetch recent person detection events from Frigate.

        Args:
            frigate_url: Base URL of Frigate server
            label: Detection label (default: "person")
            camera: Optional camera name filter
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        try:
            params = {
                'label': label,
                'limit': limit
            }

            if camera:
                params['camera'] = camera

            response = requests.get(f"{frigate_url}/api/events", params=params)
            response.raise_for_status()

            events = response.json()
            logger.debug(f"Retrieved {len(events)} events from Frigate")
            return events

        except Exception as e:
            logger.error(f"Error fetching Frigate events: {e}")
            return []

    def clear_projection_cache(self):
        """Clear the KD-Tree cache (useful after updating projections)."""
        self.projection_cache.clear()
        logger.info("Projection cache cleared")


# Example usage
if __name__ == "__main__":
    tracker = PositionTracker()

    # Example: Process a Frigate detection
    frigate_bbox = [0.654, 0.507, 0.054, 0.161]  # Normalized bbox from Frigate
    camera_name = "side_yard"
    map_id = 1
    camera_width = 2560
    camera_height = 1920

    # Convert to pixel coords
    bbox_pixels = tracker.convert_bbox_to_pixels(frigate_bbox, camera_width, camera_height)
    print(f"Pixel bbox: {bbox_pixels}")

    # Get feet position
    cam_x, cam_y = tracker.get_feet_position(frigate_bbox, camera_width, camera_height)
    print(f"Feet position: ({cam_x}, {cam_y})")

    # Lookup map position
    map_position = tracker.lookup_map_position(camera_name, map_id, cam_x, cam_y)
    if map_position:
        map_x, map_y = map_position
        print(f"Map position: ({map_x}, {map_y})")

        # Store position
        tracker.store_toddler_position("Toddler", camera_name, map_x, map_y, 0.85)

    # Or use the complete pipeline
    result = tracker.process_detection(
        camera_name=camera_name,
        map_id=map_id,
        frigate_bbox=frigate_bbox,
        camera_width=camera_width,
        camera_height=camera_height,
        subject="Toddler",
        confidence=0.85
    )

    if result:
        print(f"Successfully tracked toddler at map position: {result}")
