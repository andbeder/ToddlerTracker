"""
Enhanced detection service with hybrid identification and position tracking.
Integrates OSNet, facial recognition, and color matching for robust person identification.
Tracks toddler positions on yard map using camera-to-map projections.
"""

import threading
import requests
import cv2
import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from database import MatchesDatabase
from config_manager import ConfigManager
from image_utils import ImageProcessor
from hybrid_identifier import HybridIdentifier, IdentificationResult
from position_tracker import PositionTracker


class HybridDetectionService:
    """Enhanced detection service with hybrid identification capabilities."""

    def __init__(self, db: MatchesDatabase, config: ConfigManager):
        self.db = db
        self.config = config
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_thread_running = False

        # Initialize hybrid identifier with config manager for live threshold updates
        try:
            self.hybrid_identifier = HybridIdentifier(config_manager=config)
            self.hybrid_enabled = True
            print("Hybrid identifier initialized successfully with live threshold support")
        except Exception as e:
            print(f"Warning: Could not initialize hybrid identifier: {e}")
            self.hybrid_identifier = None
            self.hybrid_enabled = False

        # Initialize position tracker
        try:
            self.position_tracker = PositionTracker()
            self.position_tracking_enabled = True
            print("Position tracker initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize position tracker: {e}")
            self.position_tracker = None
            self.position_tracking_enabled = False

        # Default map ID for position tracking (can be configured)
        self.active_map_id = 1

        # Track last seen event IDs per camera to avoid reprocessing
        self.last_event_ids = {}

        # Statistics tracking
        self.detection_stats = {
            'total_scans': 0,
            'successful_identifications': 0,
            'positions_tracked': 0,
            'method_usage': {
                'osnet': 0,
                'face': 0,
                'color': 0,
                'hybrid': 0
            },
            'last_scan_time': None
        }

    def start_background_detection(self) -> bool:
        """Start the background detection thread."""
        if self.detection_thread is not None and self.detection_thread.is_alive():
            return False  # Already running

        self.detection_thread_running = True
        self.detection_thread = threading.Thread(target=self._run_background_detection, daemon=True)
        self.detection_thread.start()
        return True

    def stop_background_detection(self):
        """Stop the background detection thread."""
        self.detection_thread_running = False

    def is_running(self) -> bool:
        """Check if detection thread is running."""
        return (self.detection_thread is not None and
                self.detection_thread.is_alive() and
                self.detection_thread_running)

    def _run_background_detection(self):
        """Background thread for continuous hybrid detection."""
        print("Enhanced background detection thread started")

        while self.detection_thread_running:
            interval = 10  # Default interval

            try:
                settings = self.config.load_detection_settings()

                if not settings.get("enabled", False):
                    # Sleep longer when disabled to reduce CPU usage
                    threading.Event().wait(5)
                    continue

                # Get interval from settings
                interval = settings.get("scan_interval", 10)

                # Perform enhanced detection
                new_matches = self.perform_hybrid_detection_scan()

                if new_matches > 0:
                    print(f"Background hybrid detection found {new_matches} new matches")

                # Update statistics
                self.detection_stats['total_scans'] += 1
                self.detection_stats['last_scan_time'] = time.time()

            except Exception as e:
                print(f"Error in background detection: {e}")

            # Wait for the configured interval
            threading.Event().wait(interval)

        print("Enhanced background detection thread stopped")

    def perform_hybrid_detection_scan(self) -> int:
        """Perform a hybrid detection scan on all cameras using Frigate events."""
        try:
            settings = self.config.load_detection_settings()
            new_matches = 0

            # Get list of cameras from Frigate config
            frigate_config = self.config.load_frigate_config()
            cameras = frigate_config.get('cameras', {})

            for camera_name in cameras.keys():
                try:
                    # Try events-based detection first (better for position tracking)
                    matches_from_events = self._process_frigate_events(camera_name, settings)
                    if matches_from_events > 0:
                        new_matches += matches_from_events
                    else:
                        # Fallback to snapshot-based detection
                        matches_from_camera = self._process_camera_hybrid(camera_name, settings)
                        new_matches += matches_from_camera

                except Exception as e:
                    print(f"Error processing camera {camera_name} in hybrid detection: {str(e)}")
                    continue

            return new_matches

        except Exception as e:
            print(f"Error in hybrid detection scan: {e}")
            return 0

    def _process_frigate_events(self, camera_name: str, settings: Dict) -> int:
        """
        Process Frigate events for a camera with bounding box data.
        This enables position tracking on yard map.
        """
        try:
            # Get recent person detection events from Frigate
            events = self.position_tracker.get_frigate_events(
                camera=camera_name,
                label="person",
                limit=5  # Process up to 5 recent events
            )

            if not events:
                return 0

            matches_count = 0

            for event in events:
                # Skip if we've already processed this event
                event_id = event.get('id')
                if event_id == self.last_event_ids.get(camera_name):
                    continue

                # Extract bounding box data
                event_data = event.get('data', {})
                bbox = event_data.get('box')  # [x_center, y_center, width, height] normalized

                if not bbox:
                    continue

                # Get the event thumbnail/snapshot
                snapshot_url = f"http://localhost:5000/api/events/{event_id}/thumbnail.jpg"
                try:
                    response = requests.get(snapshot_url, timeout=5)
                    if response.status_code != 200:
                        continue

                    image_data = response.content
                    image_array = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if image is None:
                        continue

                    # Get camera resolution from Frigate config
                    camera_config = self._get_camera_config(camera_name)
                    camera_width = camera_config.get('width', 2560)
                    camera_height = camera_config.get('height', 1920)

                    # Convert bbox to pixel coordinates for hybrid identifier
                    bbox_pixels = self.position_tracker.convert_bbox_to_pixels(
                        bbox, camera_width, camera_height
                    )

                    # Run hybrid identification with bbox
                    if self.hybrid_enabled and self.hybrid_identifier:
                        hybrid_result = self.hybrid_identifier.identify_person(image, bbox_pixels)

                        if hybrid_result and hybrid_result.confidence >= 0.3:
                            person_id = hybrid_result.person_id
                            confidence = hybrid_result.confidence * 100  # Convert to percentage

                            # Check thresholds
                            global_min = settings.get("global_min_confidence", 50)
                            subject_threshold = self.config.get_subject_threshold(person_id)

                            if confidence >= global_min and confidence >= subject_threshold:
                                # Track position on yard map
                                if self.position_tracking_enabled and self.position_tracker:
                                    map_position = self.position_tracker.process_detection(
                                        camera_name=camera_name,
                                        map_id=self.active_map_id,
                                        frigate_bbox=bbox,
                                        camera_width=camera_width,
                                        camera_height=camera_height,
                                        subject=person_id,
                                        confidence=confidence / 100.0  # Back to 0-1 range
                                    )

                                    if map_position:
                                        self.detection_stats['positions_tracked'] += 1
                                        print(f"Tracked {person_id} at map position {map_position}")

                                # Store detection match
                                thumbnail_data = ImageProcessor.create_detection_thumbnail(
                                    image_data,
                                    self._bbox_pixels_to_compreface_format(bbox_pixels),
                                    label=f"{person_id} ({confidence:.1f}%)"
                                )

                                self.db.add_match(
                                    subject=person_id,
                                    confidence=confidence,
                                    camera=camera_name,
                                    image_data=thumbnail_data
                                )

                                # Update statistics
                                self.detection_stats['successful_identifications'] += 1
                                self.detection_stats['method_usage']['hybrid'] += 1

                                matches_count += 1

                    # Update last processed event ID
                    self.last_event_ids[camera_name] = event_id

                except Exception as e:
                    print(f"Error processing event {event_id}: {e}")
                    continue

            return matches_count

        except Exception as e:
            print(f"Error processing Frigate events for {camera_name}: {e}")
            return 0

    def _get_camera_config(self, camera_name: str) -> Dict:
        """Get camera configuration from Frigate config."""
        try:
            frigate_config = self.config.load_frigate_config()
            camera_config = frigate_config.get('cameras', {}).get(camera_name, {})

            # Get resolution from detect config
            detect_config = camera_config.get('detect', {})
            width = detect_config.get('width', 2560)
            height = detect_config.get('height', 1920)

            return {
                'width': width,
                'height': height
            }

        except Exception as e:
            print(f"Error getting camera config: {e}")
            return {'width': 2560, 'height': 1920}  # Default resolution

    def _bbox_pixels_to_compreface_format(self, bbox_pixels: List[int]) -> Dict:
        """Convert pixel bbox [x, y, w, h] to CompreFace format."""
        x, y, w, h = bbox_pixels
        return {
            'x_min': x,
            'y_min': y,
            'x_max': x + w,
            'y_max': y + h
        }

    def _process_camera_hybrid(self, camera_name: str, settings: Dict) -> int:
        """Process a single camera with hybrid identification."""
        try:
            # Get latest snapshot from Frigate
            frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"
            response = requests.get(frigate_url, timeout=5)

            if response.status_code != 200:
                return 0

            image_data = response.content

            # Convert image data to numpy array for processing
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return 0

            matches_count = 0

            # Method 1: Try hybrid identification (if available)
            if self.hybrid_enabled and self.hybrid_identifier:
                hybrid_result = self.hybrid_identifier.identify_person(image)

                if hybrid_result and hybrid_result.confidence >= 0.3:
                    # Check if this person meets the threshold requirements
                    person_id = hybrid_result.person_id
                    confidence = hybrid_result.confidence * 100  # Convert to percentage

                    global_min = settings.get("global_min_confidence", 50)
                    subject_threshold = self.config.get_subject_threshold(person_id)

                    if confidence >= global_min and confidence >= subject_threshold:
                        # Create thumbnail
                        thumbnail_data = ImageProcessor.create_thumbnail(image_data)

                        # Store match with hybrid confidence and method info
                        self.db.add_match(
                            subject=person_id,
                            confidence=confidence,
                            camera=camera_name,
                            image_data=thumbnail_data
                        )

                        # Update statistics
                        self.detection_stats['successful_identifications'] += 1
                        self.detection_stats['method_usage']['hybrid'] += 1

                        # Log which methods contributed to this identification
                        active_methods = [method for method, score in hybrid_result.method_scores.items() if score > 0]
                        for method in active_methods:
                            self.detection_stats['method_usage'][method] += 1

                        print(f"Hybrid identification: {person_id} ({confidence:.1f}%) using {active_methods}")
                        matches_count += 1

            # Method 2: Fallback to facial recognition only (original method)
            else:
                face_matches = self._process_face_recognition_fallback(image_data, camera_name, settings)
                matches_count += face_matches

            return matches_count

        except Exception as e:
            print(f"Error in camera processing: {e}")
            return 0

    def _process_face_recognition_fallback(self, image_data: bytes, camera_name: str, settings: Dict) -> int:
        """Fallback to face recognition only when hybrid is not available."""
        try:
            # Get CompreFace client
            client = self.config.get_compreface_client()
            if not client:
                return 0

            # Run face recognition
            matches = client.recognize_face(image_data)
            matches_count = 0

            for match in matches:
                subject = match['subject']
                confidence = match['confidence']

                # Check thresholds
                global_min = settings.get("global_min_confidence", 50)
                subject_threshold = self.config.get_subject_threshold(subject)

                if confidence >= global_min and confidence >= subject_threshold:
                    # Create thumbnail
                    thumbnail_data = ImageProcessor.create_thumbnail(image_data)

                    self.db.add_match(
                        subject=subject,
                        confidence=confidence,
                        camera=camera_name,
                        image_data=thumbnail_data
                    )

                    # Update statistics
                    self.detection_stats['successful_identifications'] += 1
                    self.detection_stats['method_usage']['face'] += 1

                    matches_count += 1

            return matches_count

        except Exception as e:
            print(f"Face recognition fallback error: {e}")
            return 0

    def trigger_manual_detection(self) -> Tuple[bool, int]:
        """Trigger manual hybrid detection scan."""
        try:
            new_matches = self.perform_hybrid_detection_scan()
            self.detection_stats['total_scans'] += 1
            self.detection_stats['last_scan_time'] = time.time()

            return True, new_matches

        except Exception as e:
            print(f"Error in manual detection: {e}")
            return False, 0

    def train_person_hybrid(self, person_id: str, image_data: bytes) -> bool:
        """Train the hybrid identifier with a new person image."""
        try:
            if not self.hybrid_enabled or not self.hybrid_identifier:
                return False

            # Convert image data to numpy array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return False

            # Train with single image
            self.hybrid_identifier.train_person(person_id, [image])
            print(f"Trained hybrid identifier for {person_id}")

            return True

        except Exception as e:
            print(f"Error training person {person_id}: {e}")
            return False

    def get_detection_statistics(self) -> Dict:
        """Get detection statistics and performance metrics."""
        stats = self.detection_stats.copy()

        # Add hybrid identifier stats if available
        if self.hybrid_enabled and self.hybrid_identifier:
            stats['hybrid_stats'] = self.hybrid_identifier.get_identification_stats()

        # Add performance metrics
        total_scans = stats.get('total_scans', 0)
        if total_scans > 0:
            stats['success_rate'] = (stats.get('successful_identifications', 0) / total_scans) * 100

        stats['hybrid_enabled'] = self.hybrid_enabled
        stats['position_tracking_enabled'] = self.position_tracking_enabled
        stats['active_map_id'] = self.active_map_id

        return stats

    def set_active_map_id(self, map_id: int) -> bool:
        """
        Set the active yard map ID for position tracking.

        Args:
            map_id: ID of the yard map to use for position tracking

        Returns:
            True if successful
        """
        try:
            self.active_map_id = map_id
            print(f"Active map ID set to {map_id}")
            return True
        except Exception as e:
            print(f"Error setting active map ID: {e}")
            return False

    def clear_projection_cache(self) -> bool:
        """
        Clear the position tracker projection cache.
        Useful after updating camera projections.

        Returns:
            True if successful
        """
        try:
            if self.position_tracker:
                self.position_tracker.clear_projection_cache()
                return True
            return False
        except Exception as e:
            print(f"Error clearing projection cache: {e}")
            return False

    def update_hybrid_settings(self, settings: Dict) -> bool:
        """Update hybrid identification settings."""
        try:
            if not self.hybrid_enabled or not self.hybrid_identifier:
                return False

            # Update method weights if provided
            if 'method_weights' in settings:
                self.hybrid_identifier.update_method_weights(settings['method_weights'])

            # Update method thresholds if provided
            if 'method_thresholds' in settings:
                self.hybrid_identifier.update_method_thresholds(settings['method_thresholds'])

            print("Updated hybrid identification settings")
            return True

        except Exception as e:
            print(f"Error updating hybrid settings: {e}")
            return False

    def get_hybrid_settings(self) -> Optional[Dict]:
        """Get current hybrid identification settings."""
        if not self.hybrid_enabled or not self.hybrid_identifier:
            return None

        return {
            'method_weights': self.hybrid_identifier.method_weights,
            'method_thresholds': self.hybrid_identifier.method_thresholds,
            'enabled': self.hybrid_enabled
        }

    def test_hybrid_identification(self, image_data: bytes) -> Optional[Dict]:
        """Test hybrid identification on a single image for debugging."""
        try:
            if not self.hybrid_enabled or not self.hybrid_identifier:
                return None

            # Convert image data to numpy array
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return None

            # Run hybrid identification
            result = self.hybrid_identifier.identify_person(image)

            if result:
                return {
                    'person_id': result.person_id,
                    'confidence': result.confidence,
                    'method_scores': result.method_scores,
                    'timestamp': result.timestamp
                }

            return {'message': 'No identification result'}

        except Exception as e:
            return {'error': str(e)}