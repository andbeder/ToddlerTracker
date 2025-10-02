"""
Multi-Camera Tracking System for Toddler Tracker
Handles tracking across multiple cameras with handoff detection and occlusion handling
"""

import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
import math

logger = logging.getLogger(__name__)


class Track:
    """Represents a single tracked object across cameras."""

    def __init__(self, track_id: str, camera: str, bbox: List[float],
                 confidence: float, features: Optional[np.ndarray] = None):
        self.track_id = track_id
        self.global_id = None  # Global ID across all cameras
        self.camera_history = [camera]
        self.current_camera = camera
        self.bbox = bbox  # [x, y, width, height]
        self.confidence = confidence
        self.features = features  # Feature vector for re-identification
        self.last_seen = datetime.now()
        self.first_seen = datetime.now()
        self.disappeared_count = 0
        self.positions = deque(maxlen=30)  # Last 30 positions
        self.positions.append({'camera': camera, 'bbox': bbox, 'time': datetime.now()})
        self.velocity = [0.0, 0.0]  # Estimated velocity
        self.is_toddler = False
        self.toddler_confidence = 0.0

    def update(self, camera: str, bbox: List[float], confidence: float,
               features: Optional[np.ndarray] = None):
        """Update track with new detection."""
        # Calculate velocity if same camera
        if camera == self.current_camera and len(self.positions) > 0:
            last_pos = self.positions[-1]
            time_diff = (datetime.now() - last_pos['time']).total_seconds()
            if time_diff > 0:
                self.velocity[0] = (bbox[0] - last_pos['bbox'][0]) / time_diff
                self.velocity[1] = (bbox[1] - last_pos['bbox'][1]) / time_diff

        # Update track state
        self.current_camera = camera
        if camera not in self.camera_history:
            self.camera_history.append(camera)

        self.bbox = bbox
        self.confidence = confidence
        if features is not None:
            self.features = features

        self.last_seen = datetime.now()
        self.disappeared_count = 0
        self.positions.append({'camera': camera, 'bbox': bbox, 'time': datetime.now()})

    def predict_position(self, dt: float = 0.1) -> List[float]:
        """Predict next position based on velocity."""
        return [
            self.bbox[0] + self.velocity[0] * dt,
            self.bbox[1] + self.velocity[1] * dt,
            self.bbox[2],
            self.bbox[3]
        ]

    @property
    def age_seconds(self) -> float:
        """Time since track was created."""
        return (datetime.now() - self.first_seen).total_seconds()

    @property
    def time_since_seen(self) -> float:
        """Time since last update."""
        return (datetime.now() - self.last_seen).total_seconds()


class MultiCameraTracker:
    """Manages tracking across multiple cameras with consistent IDs."""

    def __init__(self, max_disappeared: int = 50, max_distance: float = 100.0,
                 feature_threshold: float = 0.7):
        """
        Initialize multi-camera tracker.

        Args:
            max_disappeared: Max frames before track is removed
            max_distance: Max pixel distance for association
            feature_threshold: Min similarity for feature matching
        """
        self.tracks: Dict[str, Track] = {}  # track_id -> Track
        self.global_tracks: Dict[str, List[str]] = defaultdict(list)  # global_id -> [track_ids]
        self.camera_tracks: Dict[str, List[str]] = defaultdict(list)  # camera -> [track_ids]

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_threshold = feature_threshold

        self.next_global_id = 1
        self.camera_overlaps = {}  # Store camera overlap regions

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize tracking database."""
        self.conn = sqlite3.connect('tracking.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                global_id TEXT,
                camera TEXT NOT NULL,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                is_toddler BOOLEAN,
                toddler_confidence REAL,
                camera_history TEXT,
                total_detections INTEGER DEFAULT 1
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS track_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                camera TEXT,
                bbox TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id)
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS camera_handoffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_camera TEXT,
                to_camera TEXT,
                track_id TEXT,
                global_id TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    def update(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update tracker with new detections from multiple cameras.

        Args:
            detections: List of detections with format:
                {
                    'camera': str,
                    'track_id': str,  # Camera-specific track ID
                    'bbox': [x, y, w, h],
                    'confidence': float,
                    'features': np.ndarray (optional)
                }

        Returns:
            Tracking results with global IDs and associations
        """
        results = {
            'tracks': [],
            'handoffs': [],
            'disappeared': []
        }

        # Group detections by camera
        camera_detections = defaultdict(list)
        for det in detections:
            camera_detections[det['camera']].append(det)

        # Process each camera's detections
        for camera, cam_detections in camera_detections.items():
            self._process_camera_detections(camera, cam_detections, results)

        # Check for camera handoffs
        self._detect_handoffs(results)

        # Handle disappeared tracks
        self._handle_disappeared(results)

        # Clean up old tracks
        self._cleanup_tracks()

        # Save to database
        self._save_tracking_state()

        return results

    def _process_camera_detections(self, camera: str, detections: List[Dict],
                                  results: Dict):
        """Process detections from a single camera."""

        # Get existing tracks for this camera
        camera_track_ids = self.camera_tracks.get(camera, [])
        existing_tracks = [self.tracks[tid] for tid in camera_track_ids
                          if tid in self.tracks]

        # Associate detections with existing tracks
        associations = self._associate_detections(existing_tracks, detections)

        used_detections = set()
        used_tracks = set()

        # Update matched tracks
        for track_idx, det_idx in associations:
            track = existing_tracks[track_idx]
            detection = detections[det_idx]

            track.update(
                camera=camera,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                features=detection.get('features')
            )

            used_detections.add(det_idx)
            used_tracks.add(track.track_id)

            results['tracks'].append({
                'track_id': track.track_id,
                'global_id': track.global_id,
                'camera': camera,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'is_toddler': track.is_toddler
            })

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                track_id = f"{camera}_{detection['track_id']}"

                # Check if this might be a re-appearance from another camera
                global_id = self._check_reappearance(detection, camera)

                if track_id not in self.tracks:
                    track = Track(
                        track_id=track_id,
                        camera=camera,
                        bbox=detection['bbox'],
                        confidence=detection['confidence'],
                        features=detection.get('features')
                    )

                    if global_id is None:
                        # Assign new global ID
                        global_id = f"global_{self.next_global_id}"
                        self.next_global_id += 1

                    track.global_id = global_id

                    self.tracks[track_id] = track
                    self.global_tracks[global_id].append(track_id)
                    self.camera_tracks[camera].append(track_id)

                    results['tracks'].append({
                        'track_id': track.track_id,
                        'global_id': track.global_id,
                        'camera': camera,
                        'bbox': track.bbox,
                        'confidence': track.confidence,
                        'is_toddler': track.is_toddler
                    })

        # Mark unmatched tracks as disappeared
        for track_id in camera_track_ids:
            if track_id not in used_tracks and track_id in self.tracks:
                self.tracks[track_id].disappeared_count += 1

    def _associate_detections(self, tracks: List[Track],
                             detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.

        Returns list of (track_idx, detection_idx) pairs.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return []

        # Build cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            # Predict track position
            predicted_pos = track.predict_position()

            for j, detection in enumerate(detections):
                # Calculate distance cost
                dist = self._calculate_bbox_distance(predicted_pos, detection['bbox'])
                dist_cost = dist / self.max_distance if dist < self.max_distance else 1.0

                # Calculate feature similarity if available
                feat_cost = 0.0
                if track.features is not None and 'features' in detection:
                    similarity = self._calculate_feature_similarity(
                        track.features, detection['features']
                    )
                    feat_cost = 1.0 - similarity

                # Combined cost (weighted)
                cost_matrix[i, j] = 0.7 * dist_cost + 0.3 * feat_cost

        # Solve assignment problem
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Filter associations by threshold
            associations = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.5:  # Threshold for valid association
                    associations.append((row, col))

            return associations

        except ImportError:
            logger.warning("scipy not available, using greedy association")
            return self._greedy_association(cost_matrix)

    def _greedy_association(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback greedy association if scipy is not available."""
        associations = []
        used_cols = set()

        for i in range(cost_matrix.shape[0]):
            best_j = -1
            best_cost = float('inf')

            for j in range(cost_matrix.shape[1]):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j

            if best_j >= 0 and best_cost < 0.5:
                associations.append((i, best_j))
                used_cols.add(best_j)

        return associations

    def _detect_handoffs(self, results: Dict):
        """Detect and handle camera handoffs."""

        # Check for tracks appearing in overlap regions
        for track_id, track in self.tracks.items():
            if len(track.camera_history) > 1:
                # Check if this is a recent handoff
                if track.time_since_seen < 1.0:  # Within 1 second
                    prev_camera = track.camera_history[-2]
                    curr_camera = track.current_camera

                    if prev_camera != curr_camera:
                        # Record handoff
                        handoff = {
                            'from_camera': prev_camera,
                            'to_camera': curr_camera,
                            'track_id': track_id,
                            'global_id': track.global_id,
                            'confidence': track.confidence,
                            'timestamp': datetime.now()
                        }

                        results['handoffs'].append(handoff)
                        self._save_handoff(handoff)

    def _check_reappearance(self, detection: Dict, camera: str) -> Optional[str]:
        """
        Check if a new detection is a reappearance of a disappeared track.

        Returns global_id if matched, None otherwise.
        """
        if 'features' not in detection or detection['features'] is None:
            return None

        best_match_id = None
        best_similarity = 0.0

        # Check recently disappeared tracks
        for track_id, track in self.tracks.items():
            if (track.disappeared_count > 0 and
                track.disappeared_count < self.max_disappeared and
                track.features is not None):

                similarity = self._calculate_feature_similarity(
                    track.features, detection['features']
                )

                if similarity > self.feature_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track.global_id

        return best_match_id

    def _handle_disappeared(self, results: Dict):
        """Handle tracks that have disappeared."""

        for track_id, track in list(self.tracks.items()):
            if track.disappeared_count > 0:
                if track.disappeared_count >= self.max_disappeared:
                    # Remove track
                    results['disappeared'].append({
                        'track_id': track_id,
                        'global_id': track.global_id,
                        'last_camera': track.current_camera,
                        'last_seen': track.last_seen
                    })

                    # Clean up references
                    if track.global_id in self.global_tracks:
                        self.global_tracks[track.global_id].remove(track_id)
                        if len(self.global_tracks[track.global_id]) == 0:
                            del self.global_tracks[track.global_id]

                    if track.current_camera in self.camera_tracks:
                        if track_id in self.camera_tracks[track.current_camera]:
                            self.camera_tracks[track.current_camera].remove(track_id)

                    del self.tracks[track_id]

    def _cleanup_tracks(self):
        """Remove old tracks to prevent memory buildup."""
        current_time = datetime.now()
        max_age = timedelta(minutes=5)

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if current_time - track.last_seen > max_age:
                # Clean up old track
                if track.global_id in self.global_tracks:
                    self.global_tracks[track.global_id].remove(track_id)

                if track.current_camera in self.camera_tracks:
                    if track_id in self.camera_tracks[track.current_camera]:
                        self.camera_tracks[track.current_camera].remove(track_id)

                del self.tracks[track_id]

    def _calculate_bbox_distance(self, bbox1: List[float],
                                 bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes (center points)."""
        center1 = [bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2]
        center2 = [bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2]

        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _calculate_feature_similarity(self, feat1: np.ndarray,
                                     feat2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        if feat1 is None or feat2 is None:
            return 0.0

        # Normalize features
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-6)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-6)

        # Cosine similarity
        similarity = np.dot(feat1_norm, feat2_norm)

        return max(0.0, min(1.0, similarity))

    def _save_tracking_state(self):
        """Save current tracking state to database."""
        for track_id, track in self.tracks.items():
            # Save or update track
            self.conn.execute('''
                INSERT OR REPLACE INTO tracks
                (track_id, global_id, camera, first_seen, last_seen,
                 is_toddler, toddler_confidence, camera_history, total_detections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_id, track.global_id, track.current_camera,
                track.first_seen, track.last_seen,
                track.is_toddler, track.toddler_confidence,
                json.dumps(track.camera_history),
                len(track.positions)
            ))

            # Save latest position
            if len(track.positions) > 0:
                latest = track.positions[-1]
                self.conn.execute('''
                    INSERT INTO track_positions (track_id, camera, bbox, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    track_id, latest['camera'],
                    json.dumps(latest['bbox']), latest['time']
                ))

        self.conn.commit()

    def _save_handoff(self, handoff: Dict):
        """Save camera handoff event."""
        self.conn.execute('''
            INSERT INTO camera_handoffs
            (from_camera, to_camera, track_id, global_id, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            handoff['from_camera'], handoff['to_camera'],
            handoff['track_id'], handoff['global_id'],
            handoff['confidence'], handoff['timestamp']
        ))
        self.conn.commit()

    def get_active_tracks(self) -> List[Dict]:
        """Get all currently active tracks."""
        active = []
        for track_id, track in self.tracks.items():
            if track.disappeared_count == 0:
                active.append({
                    'track_id': track_id,
                    'global_id': track.global_id,
                    'camera': track.current_camera,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'velocity': track.velocity,
                    'age': track.age_seconds,
                    'is_toddler': track.is_toddler,
                    'camera_history': track.camera_history
                })
        return active

    def get_track_history(self, global_id: str, limit: int = 100) -> List[Dict]:
        """Get position history for a global track."""
        cursor = self.conn.execute('''
            SELECT tp.camera, tp.bbox, tp.timestamp, t.track_id
            FROM track_positions tp
            JOIN tracks t ON tp.track_id = t.track_id
            WHERE t.global_id = ?
            ORDER BY tp.timestamp DESC
            LIMIT ?
        ''', (global_id, limit))

        history = []
        for row in cursor.fetchall():
            history.append({
                'camera': row[0],
                'bbox': json.loads(row[1]),
                'timestamp': row[2],
                'track_id': row[3]
            })

        return history

    def get_camera_handoffs(self, hours: int = 1) -> List[Dict]:
        """Get recent camera handoff events."""
        since = datetime.now() - timedelta(hours=hours)

        cursor = self.conn.execute('''
            SELECT from_camera, to_camera, track_id, global_id,
                   confidence, timestamp
            FROM camera_handoffs
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (since,))

        handoffs = []
        for row in cursor.fetchall():
            handoffs.append({
                'from_camera': row[0],
                'to_camera': row[1],
                'track_id': row[2],
                'global_id': row[3],
                'confidence': row[4],
                'timestamp': row[5]
            })

        return handoffs

    def set_camera_overlap(self, camera1: str, camera2: str,
                          overlap_region: Dict):
        """Define overlapping regions between cameras for better handoff."""
        key = tuple(sorted([camera1, camera2]))
        self.camera_overlaps[key] = overlap_region

    def mark_as_toddler(self, track_id: str, confidence: float):
        """Mark a track as belonging to the toddler."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.is_toddler = True
            track.toddler_confidence = confidence

            # Update all tracks with same global ID
            if track.global_id in self.global_tracks:
                for tid in self.global_tracks[track.global_id]:
                    if tid in self.tracks:
                        self.tracks[tid].is_toddler = True
                        self.tracks[tid].toddler_confidence = confidence

    def clear_all_data(self):
        """Clear all tracking data including tracks, handoffs, and database records."""
        try:
            # Clear in-memory data structures
            self.tracks.clear()
            self.global_tracks.clear()
            self.camera_tracks.clear()
            self.next_global_id = 1

            # Clear database tables
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM tracks')
            cursor.execute('DELETE FROM track_positions')
            cursor.execute('DELETE FROM camera_handoffs')
            self.conn.commit()

            logger.info("All tracking data cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing tracking data: {e}")
            return False


class TrackingService:
    """Service for managing multi-camera tracking."""

    def __init__(self, tracker: Optional[MultiCameraTracker] = None):
        self.tracker = tracker or MultiCameraTracker()
        self.last_update = datetime.now()

    def process_detections(self, detections: List[Dict]) -> Dict:
        """
        Process new detections and update tracking.

        Expected detection format:
        {
            'camera': 'camera_name',
            'track_id': 'local_track_id',
            'bbox': [x, y, width, height],
            'confidence': 0.95,
            'features': np.array(...) optional
        }
        """
        results = self.tracker.update(detections)
        self.last_update = datetime.now()
        return results

    def get_toddler_tracks(self) -> List[Dict]:
        """Get all tracks identified as the toddler."""
        all_tracks = self.tracker.get_active_tracks()
        return [t for t in all_tracks if t['is_toddler']]

    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = self.tracker.get_active_tracks()
        handoffs = self.tracker.get_camera_handoffs(hours=1)

        return {
            'total_tracks': len(self.tracker.tracks),
            'active_tracks': len(active_tracks),
            'toddler_tracks': len([t for t in active_tracks if t['is_toddler']]),
            'cameras_active': len(set(t['camera'] for t in active_tracks)),
            'recent_handoffs': len(handoffs),
            'last_update': self.last_update.isoformat()
        }