"""
Hybrid Person Identification System
Combines OSNet person re-identification, facial recognition, and color matching
for robust toddler identification across multiple cameras
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

from osnet_extractor import OSNetExtractor, OSNetDatabase
from compreface_client import CompreFaceClient
from database import MatchesDatabase

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result from hybrid identification."""
    person_id: str
    confidence: float
    method_scores: Dict[str, float]  # Scores from each identification method
    features: Dict[str, Any]  # Features extracted from each method
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ColorMatcher:
    """Color-based person identification using clothing dominant colors."""

    def __init__(self):
        """Initialize color matcher with daily adaptive learning."""
        # person_id -> dict with daily color data
        # Structure: {
        #   'features': np.ndarray,      # Today's color histogram
        #   'timestamp': datetime,        # When extracted
        #   'confidence': float,          # Confidence of source match
        #   'source_methods': list        # Which methods contributed to identification
        # }
        self.daily_color_cache = {}
        self.color_expiration_hours = 12  # Colors expire after 12 hours
        logger.info("Color matcher initialized with daily adaptive learning")

    def extract_color_features(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> np.ndarray:
        """
        Extract dominant color features from person image.

        Args:
            image: Input image (H, W, 3) in BGR format
            bbox: Optional bounding box [x, y, width, height]

        Returns:
            Color histogram features (normalized)
        """
        try:
            # Crop person region if bbox provided
            if bbox is not None:
                x, y, w, h = [int(coord) for coord in bbox]
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)

                if w > 0 and h > 0:
                    image = image[y:y+h, x:x+w]

            # Focus on upper body (clothing region) - top 60% of person
            height = image.shape[0]
            clothing_region = image[:int(height * 0.6), :]

            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)

            # Create mask to exclude skin colors (approximate range)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            clothing_mask = cv2.bitwise_not(skin_mask)

            # Calculate color histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], clothing_mask, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], clothing_mask, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], clothing_mask, [32], [0, 256])

            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            # Combine histograms
            color_features = np.concatenate([hist_h, hist_s, hist_v])

            return color_features

        except Exception as e:
            logger.error(f"Error extracting color features: {e}")
            return np.zeros(96, dtype=np.float32)  # 32 * 3 channels

    def update_daily_colors(self, person_id: str, color_features: np.ndarray,
                           confidence: float, source_methods: List[str]):
        """
        Update daily color profile for a person.
        Should only be called after high-confidence identification from OSNet/Face.

        Args:
            person_id: Person identifier
            color_features: Extracted color histogram
            confidence: Confidence of the identification that triggered this update
            source_methods: Methods that contributed to the identification (e.g., ['osnet', 'face'])
        """
        self.daily_color_cache[person_id] = {
            'features': color_features,
            'timestamp': datetime.now(),
            'confidence': confidence,
            'source_methods': source_methods
        }
        logger.info(f"Updated daily colors for {person_id} (confidence: {confidence:.2f}, sources: {source_methods})")

    def compute_color_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute color similarity using histogram correlation."""
        try:
            # Use correlation coefficient for histogram comparison
            correlation = cv2.compareHist(features1.astype(np.float32),
                                        features2.astype(np.float32),
                                        cv2.HISTCMP_CORREL)

            # Convert correlation (-1 to 1) to similarity (0 to 1)
            similarity = (correlation + 1.0) / 2.0
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Error computing color similarity: {e}")
            return 0.0

    def has_fresh_colors(self, person_id: str) -> bool:
        """
        Check if person has fresh daily color data.

        Args:
            person_id: Person identifier

        Returns:
            True if fresh color data exists (< expiration_hours old)
        """
        if person_id not in self.daily_color_cache:
            return False

        cache_entry = self.daily_color_cache[person_id]
        age_hours = (datetime.now() - cache_entry['timestamp']).total_seconds() / 3600

        if age_hours > self.color_expiration_hours:
            logger.debug(f"Color cache for {person_id} expired ({age_hours:.1f}h old)")
            return False

        return True

    def identify_person_by_color(self, color_features: np.ndarray, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Identify person based on TODAY's color features only.
        Only uses daily color cache, not training photos.

        Args:
            color_features: Color features to match
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (person_id, confidence) or None if no match
        """
        best_match = None
        best_similarity = 0.0

        # Only check fresh daily color cache
        for person_id, cache_entry in self.daily_color_cache.items():
            # Skip expired entries
            if not self.has_fresh_colors(person_id):
                continue

            stored_colors = cache_entry['features']
            similarity = self.compute_color_similarity(color_features, stored_colors)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        if best_similarity >= threshold:
            logger.debug(f"Color match: {best_match} with {best_similarity:.2f} similarity")
            return best_match, best_similarity

        return None

    def clear_expired_colors(self):
        """Remove expired color cache entries."""
        expired_persons = [
            person_id for person_id in self.daily_color_cache.keys()
            if not self.has_fresh_colors(person_id)
        ]

        for person_id in expired_persons:
            del self.daily_color_cache[person_id]
            logger.info(f"Cleared expired color cache for {person_id}")

    def clear_all_colors(self):
        """Clear all daily color cache (e.g., at midnight or manual reset)."""
        self.daily_color_cache.clear()
        logger.info("Cleared all daily color cache")


class HybridIdentifier:
    """Hybrid person identification combining multiple methods."""

    def __init__(self, config_path: str = "config.yaml", config_manager=None):
        """Initialize hybrid identifier."""
        self.osnet_extractor = OSNetExtractor()
        self.osnet_db = OSNetDatabase()
        self.color_matcher = ColorMatcher()

        # Store config manager for dynamic threshold lookups
        if config_manager is None:
            try:
                from config_manager import ConfigManager
                self.config_manager = ConfigManager(config_path)
            except Exception as e:
                logger.warning(f"Could not initialize ConfigManager: {e}")
                self.config_manager = None
        else:
            self.config_manager = config_manager

        # Initialize CompreFace client
        try:
            from config_manager import ConfigManager
            if self.config_manager:
                self.compreface_client = CompreFaceClient(self.config_manager)
            else:
                config_manager_temp = ConfigManager(config_path)
                self.compreface_client = CompreFaceClient(config_manager_temp)
        except Exception as e:
            logger.warning(f"Could not initialize CompreFace client: {e}")
            self.compreface_client = None

        # Method weights for confidence fusion (dynamic based on color freshness)
        # When NO fresh colors available:
        self.method_weights_no_color = {
            'osnet': 0.5,      # OSNet person re-identification
            'face': 0.5,       # Facial recognition
            'color': 0.0       # Color matching disabled
        }

        # When fresh colors ARE available:
        self.method_weights_with_color = {
            'osnet': 0.3,      # OSNet person re-identification
            'face': 0.3,       # Facial recognition
            'color': 0.4       # Color matching (higher weight - today's clothes!)
        }

        # Minimum thresholds for each method (defaults)
        # NOTE: Face threshold is now per-subject and read from thresholds.json
        # These are fallback values only
        self.method_thresholds = {
            'osnet': 0.5,
            'face': 0.6,  # Fallback only - actual threshold from config per subject
            'color': 0.5
        }

        # Minimum confidence threshold for updating daily colors
        # Only update colors when we have high confidence from OSNet/Face
        self.color_learning_threshold = 0.7

        logger.info("Hybrid identifier initialized")

    def extract_all_features(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Extract features using all identification methods.

        Args:
            image: Input image (H, W, 3) in BGR format
            bbox: Optional bounding box [x, y, width, height]

        Returns:
            Dictionary of features from each method
        """
        features = {}

        try:
            # Extract OSNet features
            features['osnet'] = self.osnet_extractor.extract_features(image, bbox)

            # Extract color features
            features['color'] = self.color_matcher.extract_color_features(image, bbox)

            # Extract facial features (if CompreFace is available)
            if self.compreface_client:
                try:
                    # Convert image to bytes for CompreFace
                    _, img_encoded = cv2.imencode('.jpg', image)
                    img_bytes = img_encoded.tobytes()

                    # Get face recognition results
                    face_results = self.compreface_client.recognize_faces(img_bytes)
                    if face_results and len(face_results) > 0:
                        # Use the first face found
                        features['face'] = face_results[0]  # Store full result
                    else:
                        features['face'] = None

                except Exception as e:
                    logger.debug(f"Face recognition failed: {e}")
                    features['face'] = None
            else:
                features['face'] = None

        except Exception as e:
            logger.error(f"Error extracting features: {e}")

        return features

    def identify_person(self, image: np.ndarray, bbox: Optional[List[float]] = None) -> Optional[IdentificationResult]:
        """
        Identify person using hybrid approach with adaptive color learning.

        Args:
            image: Input image (H, W, 3) in BGR format
            bbox: Optional bounding box [x, y, width, height]

        Returns:
            IdentificationResult or None if no confident match
        """
        try:
            # Clean up expired color cache entries
            self.color_matcher.clear_expired_colors()

            # Extract features from all methods
            features = self.extract_all_features(image, bbox)

            # Get identification scores from each method
            method_scores = {}
            candidates = {}
            primary_methods = []  # Track which primary methods (osnet/face) contributed

            # OSNet identification
            if 'osnet' in features:
                osnet_result = self._identify_by_osnet(features['osnet'])
                if osnet_result:
                    person_id, confidence = osnet_result
                    method_scores['osnet'] = confidence
                    primary_methods.append('osnet')
                    # Weight will be applied later based on color availability

            # Facial recognition with per-subject threshold
            if features.get('face'):
                face_result = features['face']
                if 'subject' in face_result:
                    person_id = face_result['subject']
                    face_similarity = face_result.get('similarity', 0)

                    # Get subject-specific threshold from config (live lookup for immediate updates)
                    if self.config_manager:
                        subject_threshold = self.config_manager.get_subject_threshold(person_id)
                    else:
                        # Fallback to default if no config manager
                        subject_threshold = self.method_thresholds['face'] * 100  # Convert to percentage

                    logger.debug(f"Face recognition: {person_id} with {face_similarity}% "
                               f"(threshold: {subject_threshold}%)")

                    # Check against subject-specific threshold
                    if face_similarity >= subject_threshold:
                        confidence = face_similarity / 100.0  # Convert percentage to 0-1
                        method_scores['face'] = confidence
                        primary_methods.append('face')
                        # Weight will be applied later based on color availability
                    else:
                        logger.debug(f"Face recognition below threshold: {face_similarity}% < {subject_threshold}%")

            # Determine which weights to use based on whether we have fresh colors
            # Check if ANY candidate has fresh colors
            has_any_fresh_colors = False
            if candidates:
                for person_id in set([pid for pid in candidates.keys()]):
                    if self.color_matcher.has_fresh_colors(person_id):
                        has_any_fresh_colors = True
                        break

            # If we don't have candidates yet, check if we have fresh colors for any person
            if not has_any_fresh_colors and method_scores:
                # Check the person IDs from method scores
                for method, score in method_scores.items():
                    # Get person_id from the identification result
                    if method == 'osnet':
                        osnet_result = self._identify_by_osnet(features['osnet'])
                        if osnet_result and self.color_matcher.has_fresh_colors(osnet_result[0]):
                            has_any_fresh_colors = True
                            break
                    elif method == 'face' and features.get('face'):
                        face_person_id = features['face'].get('subject')
                        if face_person_id and self.color_matcher.has_fresh_colors(face_person_id):
                            has_any_fresh_colors = True
                            break

            # Select appropriate weights
            if has_any_fresh_colors:
                active_weights = self.method_weights_with_color
                logger.debug("Using weights WITH fresh color data")
            else:
                active_weights = self.method_weights_no_color
                logger.debug("Using weights WITHOUT color data (first detection of day)")

            # Apply weights to primary methods (OSNet and Face)
            if 'osnet' in method_scores:
                osnet_result = self._identify_by_osnet(features['osnet'])
                if osnet_result:
                    person_id, confidence = osnet_result
                    candidates[person_id] = candidates.get(person_id, 0) + confidence * active_weights['osnet']

            if 'face' in method_scores:
                face_result = features['face']
                person_id = face_result['subject']
                confidence = method_scores['face']
                candidates[person_id] = candidates.get(person_id, 0) + confidence * active_weights['face']

            # Color matching (only if we have fresh colors)
            if active_weights['color'] > 0 and 'color' in features:
                color_result = self.color_matcher.identify_person_by_color(
                    features['color'], self.method_thresholds['color']
                )
                if color_result:
                    person_id, confidence = color_result
                    method_scores['color'] = confidence
                    candidates[person_id] = candidates.get(person_id, 0) + confidence * active_weights['color']

            # Find best candidate
            if candidates:
                best_person_id = max(candidates, key=candidates.get)
                best_confidence = candidates[best_person_id]

                # Check if confidence is above minimum threshold
                if best_confidence >= 0.3:  # Overall confidence threshold
                    # ADAPTIVE COLOR LEARNING:
                    # If this is a high-confidence match from primary methods (OSNet/Face)
                    # and we DON'T have fresh colors yet, learn today's colors
                    primary_confidence = sum([method_scores.get(m, 0) * active_weights[m]
                                            for m in primary_methods])

                    if (primary_confidence >= self.color_learning_threshold and
                        not self.color_matcher.has_fresh_colors(best_person_id) and
                        'color' in features):
                        # Extract and store today's colors
                        self.color_matcher.update_daily_colors(
                            person_id=best_person_id,
                            color_features=features['color'],
                            confidence=primary_confidence,
                            source_methods=primary_methods
                        )
                        logger.info(f"Learned daily colors for {best_person_id} "
                                  f"(primary confidence: {primary_confidence:.2f})")

                    return IdentificationResult(
                        person_id=best_person_id,
                        confidence=best_confidence,
                        method_scores=method_scores,
                        features=features
                    )

            return None

        except Exception as e:
            logger.error(f"Error in hybrid identification: {e}")
            return None

    def _identify_by_osnet(self, osnet_features: np.ndarray) -> Optional[Tuple[str, float]]:
        """Identify person using OSNet features."""
        try:
            # Get all known persons
            # This would need to be implemented based on your person database
            # For now, return None (to be implemented with actual person database)
            return None

        except Exception as e:
            logger.error(f"Error in OSNet identification: {e}")
            return None

    def train_person(self, person_id: str, images: List[np.ndarray], bboxes: Optional[List[List[float]]] = None):
        """
        Train the hybrid identifier with images of a person.
        NOTE: Colors from training images are NOT used for matching.
        Only OSNet features are stored. Daily colors are learned adaptively during runtime.

        Args:
            person_id: Unique identifier for the person
            images: List of training images
            bboxes: Optional list of bounding boxes for each image
        """
        try:
            if bboxes is None:
                bboxes = [None] * len(images)

            for i, image in enumerate(images):
                bbox = bboxes[i] if i < len(bboxes) else None

                # Extract and store features
                features = self.extract_all_features(image, bbox)

                # Store OSNet features for person re-identification
                if 'osnet' in features:
                    self.osnet_db.add_person_features(person_id, features['osnet'])

                # DO NOT store color features from training images
                # Colors will be learned adaptively when person is first identified each day

                # Facial recognition training would be handled by CompreFace separately

            logger.info(f"Trained hybrid identifier for person {person_id} with {len(images)} images (OSNet only)")

        except Exception as e:
            logger.error(f"Error training person {person_id}: {e}")

    def get_identification_stats(self) -> Dict[str, Any]:
        """Get statistics about identification methods."""
        # Get color cache details
        color_cache_info = {}
        for person_id, cache_entry in self.color_matcher.daily_color_cache.items():
            age_hours = (datetime.now() - cache_entry['timestamp']).total_seconds() / 3600
            color_cache_info[person_id] = {
                'age_hours': round(age_hours, 2),
                'is_fresh': self.color_matcher.has_fresh_colors(person_id),
                'confidence': cache_entry['confidence'],
                'source_methods': cache_entry['source_methods']
            }

        return {
            'method_weights_no_color': self.method_weights_no_color,
            'method_weights_with_color': self.method_weights_with_color,
            'method_thresholds': self.method_thresholds,
            'color_learning_threshold': self.color_learning_threshold,
            'osnet_model': self.osnet_extractor.get_model_info(),
            'daily_color_cache': color_cache_info,
            'color_expiration_hours': self.color_matcher.color_expiration_hours,
            'compreface_available': self.compreface_client is not None
        }

    def update_method_weights(self, new_weights: Dict[str, float]):
        """
        Update weights for method fusion.
        NOTE: This updates BOTH weight sets (with/without color) proportionally.
        """
        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            normalized = {k: v/total for k, v in new_weights.items()}
            # Update both weight sets
            self.method_weights_no_color.update({k: v for k, v in normalized.items() if k != 'color'})
            self.method_weights_with_color.update(normalized)
            logger.info(f"Updated method weights (with color): {self.method_weights_with_color}")
            logger.info(f"Updated method weights (no color): {self.method_weights_no_color}")

    def update_method_thresholds(self, new_thresholds: Dict[str, float]):
        """Update minimum thresholds for each method."""
        self.method_thresholds.update(new_thresholds)
        logger.info(f"Updated method thresholds: {self.method_thresholds}")

    def clear_daily_colors(self, person_id: Optional[str] = None):
        """
        Clear daily color cache.

        Args:
            person_id: If specified, clear only this person's colors. If None, clear all.
        """
        if person_id:
            if person_id in self.color_matcher.daily_color_cache:
                del self.color_matcher.daily_color_cache[person_id]
                logger.info(f"Cleared daily colors for {person_id}")
        else:
            self.color_matcher.clear_all_colors()

    def get_daily_color_status(self, person_id: str) -> Optional[Dict]:
        """
        Get the status of daily color cache for a person.

        Args:
            person_id: Person identifier

        Returns:
            Dict with color cache status or None if no cache exists
        """
        if person_id not in self.color_matcher.daily_color_cache:
            return None

        cache_entry = self.color_matcher.daily_color_cache[person_id]
        age_hours = (datetime.now() - cache_entry['timestamp']).total_seconds() / 3600

        return {
            'has_colors': True,
            'is_fresh': self.color_matcher.has_fresh_colors(person_id),
            'age_hours': round(age_hours, 2),
            'confidence': cache_entry['confidence'],
            'source_methods': cache_entry['source_methods'],
            'timestamp': cache_entry['timestamp'].isoformat()
        }

    def set_color_expiration_hours(self, hours: float):
        """
        Set the color cache expiration time.

        Args:
            hours: Number of hours before colors expire
        """
        self.color_matcher.color_expiration_hours = hours
        logger.info(f"Color expiration set to {hours} hours")


# Global hybrid identifier instance
_hybrid_identifier = None

def get_hybrid_identifier() -> HybridIdentifier:
    """Get global hybrid identifier instance."""
    global _hybrid_identifier
    if _hybrid_identifier is None:
        _hybrid_identifier = HybridIdentifier()
    return _hybrid_identifier

def identify_person_hybrid(image: np.ndarray, bbox: Optional[List[float]] = None) -> Optional[IdentificationResult]:
    """Convenience function for hybrid identification."""
    identifier = get_hybrid_identifier()
    return identifier.identify_person(image, bbox)