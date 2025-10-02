"""
Face detection service for continuous monitoring.
Handles background detection thread and camera processing.
"""

import threading
import requests
from typing import Optional
from database import MatchesDatabase
from config_manager import ConfigManager
from image_utils import ImageProcessor


class DetectionService:
    """Manages continuous face detection on camera feeds."""

    def __init__(self, db: MatchesDatabase, config: ConfigManager):
        self.db = db
        self.config = config
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_thread_running = False

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
        """Background thread for continuous face detection."""
        print("Background detection thread started")

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

                # Perform detection
                new_matches = self.perform_detection_scan()

                if new_matches > 0:
                    print(f"Background detection found {new_matches} new matches")

            except Exception as e:
                print(f"Error in background detection: {e}")

            # Wait for the configured interval
            threading.Event().wait(interval)

        print("Background detection thread stopped")

    def perform_detection_scan(self) -> int:
        """Perform a single detection scan on all cameras."""
        try:
            settings = self.config.load_detection_settings()
            new_matches = 0

            # Get CompreFace client
            client = self.config.get_compreface_client()
            if not client:
                return 0

            # Get list of cameras from Frigate config
            frigate_config = self.config.load_frigate_config()
            cameras = frigate_config.get('cameras', {})

            for camera_name in cameras.keys():
                try:
                    # Get latest snapshot from Frigate
                    frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"
                    response = requests.get(frigate_url, timeout=5)

                    if response.status_code == 200:
                        image_data = response.content

                        # Run face recognition
                        matches = client.recognize_face(image_data)

                        for match in matches:
                            subject = match['subject']
                            confidence = match['confidence']
                            box = match.get('box', {})

                            # Check thresholds
                            global_min = settings.get("global_min_confidence", 50)
                            subject_threshold = self.config.get_subject_threshold(subject)

                            if confidence >= global_min and confidence >= subject_threshold:
                                # Create thumbnail with detection bounding box
                                thumbnail_data = ImageProcessor.create_detection_thumbnail(
                                    image_data,
                                    box,
                                    label=f"{subject} ({confidence:.1f}%)"
                                )

                                self.db.add_match(
                                    subject=subject,
                                    confidence=confidence,
                                    camera=camera_name,
                                    image_data=thumbnail_data
                                )
                                new_matches += 1

                except Exception as e:
                    print(f"Error processing camera {camera_name} in background: {str(e)}")
                    continue

            return new_matches
        except Exception as e:
            print(f"Error in detection scan: {e}")
            return 0

    def trigger_manual_detection(self) -> tuple[bool, int]:
        """Trigger manual detection scan and return (success, new_matches)."""
        try:
            new_matches = 0
            settings = self.config.load_detection_settings()

            # Get CompreFace client
            client = self.config.get_compreface_client()
            if not client:
                return False, 0

            # Get list of cameras from Frigate config
            frigate_config = self.config.load_frigate_config()
            cameras = frigate_config.get('cameras', {})

            for camera_name in cameras.keys():
                try:
                    # Get latest snapshot from Frigate
                    frigate_url = f"http://localhost:5000/api/{camera_name}/latest.jpg"
                    response = requests.get(frigate_url, timeout=5)

                    if response.status_code == 200:
                        image_data = response.content

                        # Run face recognition
                        matches = client.recognize_face(image_data)

                        for match in matches:
                            subject = match['subject']
                            confidence = match['confidence']
                            box = match.get('box', {})

                            # Check both global minimum and subject-specific threshold
                            subject_threshold = self.config.get_subject_threshold(subject)
                            if confidence >= 50 and confidence >= subject_threshold:
                                # Create thumbnail with detection bounding box
                                thumbnail_data = ImageProcessor.create_detection_thumbnail(
                                    image_data,
                                    box,
                                    label=f"{subject} ({confidence:.1f}%)"
                                )

                                self.db.add_match(
                                    subject=subject,
                                    confidence=confidence,
                                    camera=camera_name,
                                    image_data=thumbnail_data
                                )
                                new_matches += 1

                except Exception as e:
                    print(f"Error processing camera {camera_name}: {str(e)}")
                    continue

            return True, new_matches
        except Exception as e:
            print(f"Error in manual detection: {e}")
            return False, 0