#!/usr/bin/env python3
"""
Standalone Hybrid Detection Service
Runs independently as a Docker container, polls Frigate for person detections,
performs hybrid identification (OSNet + Face + Color), and tracks positions.
"""

import time
import logging
import signal
import sys
from typing import Dict
from hybrid_detection_service import HybridDetectionService
from database import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectorService:
    """Standalone detection service that runs continuously."""

    def __init__(self):
        self.running = False
        self.config = ConfigManager()
        self.detection_service = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def start(self):
        """Start the detection service."""
        try:
            logger.info("Starting Hybrid Detection Service...")

            # Initialize hybrid detection service
            self.detection_service = HybridDetectionService()

            # Get enabled cameras from config
            cameras = self.config.get_all_cameras()
            enabled_cameras = {name: settings for name, settings in cameras.items()
                             if settings.get('enabled', False)}

            if not enabled_cameras:
                logger.warning("No cameras enabled in configuration!")
                logger.info("Add cameras via the web UI or edit config database")

            logger.info(f"Monitoring {len(enabled_cameras)} cameras: {list(enabled_cameras.keys())}")

            # Start background detection
            self.detection_service.start_background_detection()

            self.running = True
            logger.info("Detection service started successfully")

            # Main loop - just keep alive and log stats periodically
            while self.running:
                time.sleep(60)  # Log stats every minute
                stats = self.detection_service.get_detection_statistics()
                logger.info(
                    f"Stats: {stats.get('detections_processed', 0)} detections, "
                    f"{stats.get('matches_found', 0)} matches, "
                    f"{stats.get('positions_tracked', 0)} positions tracked"
                )

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"Error in detection service: {e}", exc_info=True)
            self.stop()
            sys.exit(1)

    def stop(self):
        """Stop the detection service."""
        logger.info("Stopping detection service...")
        self.running = False

        if self.detection_service:
            self.detection_service.stop_background_detection()

        logger.info("Detection service stopped")
        sys.exit(0)


def main():
    """Main entry point for standalone detector service."""
    logger.info("=" * 60)
    logger.info("Hybrid Detection Service - Standalone Mode")
    logger.info("=" * 60)

    service = DetectorService()
    service.start()


if __name__ == "__main__":
    main()
