"""
Toddler-Focused Safety System
Optimized for identifying Erik (toddler) while safely handling other family members
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for toddler-focused safety system."""
    target_child: str = "Erik"
    max_false_positive_rate: float = 0.01  # 1% max false positives
    min_confidence_toddler: float = 0.95   # Very high for toddler
    enable_unknown_person_tracking: bool = True
    alert_on_unknown_adult: bool = True


class ToddlerFocusedSafety:
    """Safety system focused on accurate toddler identification."""

    def __init__(self, config: SafetyConfig = None):
        """Initialize toddler-focused safety system."""
        self.config = config or SafetyConfig()
        self.toddler_characteristics = self._define_toddler_characteristics()
        self.identification_history = []

        logger.info(f"Toddler-focused safety initialized for {self.config.target_child}")

    def _define_toddler_characteristics(self) -> Dict[str, any]:
        """Define what makes someone identifiable as the target toddler."""
        return {
            # Physical characteristics
            'height_range': (60, 200),  # pixels in camera view - adjust for your setup
            'width_range': (40, 120),   # pixels - toddler body width
            'head_body_ratio': (0.25, 0.45),  # toddlers have larger heads proportionally

            # Movement characteristics
            'movement_patterns': {
                'unsteady_gait': True,
                'frequent_direction_changes': True,
                'low_center_of_gravity': True
            },

            # Context characteristics
            'typical_locations': ['backyard', 'play_area'],
            'typical_times': ['morning', 'afternoon'],
            'supervised_activity': True,

            # Clothing characteristics (you'll calibrate these)
            'typical_colors': ['bright_primary_colors'],
            'clothing_style': 'child_casual'
        }

    def validate_toddler_identification(self,
                                      person_id: str,
                                      confidence: float,
                                      image: np.ndarray,
                                      bbox: List[float],
                                      method_scores: Dict[str, float] = None,
                                      camera: str = None) -> Dict[str, any]:
        """
        Validate if this identification is truly the target toddler.

        Returns:
            {
                'is_valid': bool,
                'confidence_adjusted': float,
                'risk_level': str,  # 'low', 'medium', 'high'
                'warnings': List[str],
                'recommendation': str
            }
        """
        result = {
            'is_valid': True,
            'confidence_adjusted': confidence,
            'risk_level': 'low',
            'warnings': [],
            'recommendation': 'accept'
        }

        # Only validate if claiming to be our target toddler
        if person_id != self.config.target_child:
            return result

        # Check 1: Confidence threshold
        if confidence < self.config.min_confidence_toddler:
            result['is_valid'] = False
            result['risk_level'] = 'high'
            result['warnings'].append(f"Confidence {confidence:.1%} below toddler threshold {self.config.min_confidence_toddler:.1%}")
            result['recommendation'] = 'reject_low_confidence'

        # Check 2: Physical size validation
        size_check = self._validate_toddler_size(bbox)
        if not size_check['valid']:
            result['is_valid'] = False
            result['risk_level'] = 'high'
            result['warnings'].extend(size_check['reasons'])
            result['recommendation'] = 'reject_wrong_size'

        # Check 3: Multi-method agreement (hybrid validation)
        if method_scores:
            agreement_check = self._validate_method_agreement(method_scores)
            if not agreement_check['valid']:
                result['confidence_adjusted'] *= 0.7  # Reduce confidence
                result['risk_level'] = 'medium'
                result['warnings'].extend(agreement_check['warnings'])

        # Check 4: Temporal consistency
        temporal_check = self._check_temporal_consistency(person_id, camera)
        if not temporal_check['valid']:
            result['confidence_adjusted'] *= 0.8
            result['warnings'].extend(temporal_check['warnings'])

        # Final decision logic
        if result['risk_level'] == 'high':
            result['is_valid'] = False
            result['recommendation'] = 'reject_high_risk'
        elif result['risk_level'] == 'medium' and result['confidence_adjusted'] < 0.85:
            result['is_valid'] = False
            result['recommendation'] = 'reject_medium_risk_low_confidence'

        return result

    def _validate_toddler_size(self, bbox: List[float]) -> Dict[str, any]:
        """Validate that the detected person size matches toddler characteristics."""
        if not bbox or len(bbox) < 4:
            return {'valid': False, 'reasons': ['No bounding box provided']}

        x, y, width, height = bbox

        # Check height
        height_range = self.toddler_characteristics['height_range']
        if height < height_range[0] or height > height_range[1]:
            return {
                'valid': False,
                'reasons': [f"Height {height}px outside toddler range {height_range}"]
            }

        # Check width
        width_range = self.toddler_characteristics['width_range']
        if width < width_range[0] or width > width_range[1]:
            return {
                'valid': False,
                'reasons': [f"Width {width}px outside toddler range {width_range}"]
            }

        # Check aspect ratio (toddlers are more square, adults more rectangular)
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 2.5:  # Toddler proportions
            return {
                'valid': False,
                'reasons': [f"Aspect ratio {aspect_ratio:.2f} not typical for toddler"]
            }

        return {'valid': True, 'reasons': []}

    def _validate_method_agreement(self, method_scores: Dict[str, float]) -> Dict[str, any]:
        """Validate that multiple identification methods agree."""
        result = {'valid': True, 'warnings': []}

        face_score = method_scores.get('face', 0)
        osnet_score = method_scores.get('osnet', 0)
        color_score = method_scores.get('color', 0)

        # Count high-confidence methods
        high_confidence_methods = sum(1 for score in method_scores.values() if score > 0.7)

        # For toddler safety, require at least 2 methods to agree
        if high_confidence_methods < 2:
            result['valid'] = False
            result['warnings'].append("Need at least 2 identification methods to agree for toddler safety")

        # Be suspicious if only face recognition is high (sibling confusion)
        if face_score > 0.8 and osnet_score < 0.4 and color_score < 0.4:
            result['valid'] = False
            result['warnings'].append("Face-only identification - possible sibling confusion")

        # OSNet should be primary for toddlers (body characteristics are more reliable)
        if osnet_score < 0.6 and face_score > 0.8:
            result['warnings'].append("Body characteristics don't strongly match - verify identification")

        return result

    def _check_temporal_consistency(self, person_id: str, camera: str) -> Dict[str, any]:
        """Check for impossible rapid changes that might indicate false positives."""
        result = {'valid': True, 'warnings': []}

        # Get recent identifications (last 60 seconds)
        recent_ids = self._get_recent_identifications(60)

        # Check for rapid switching between different people
        if len(recent_ids) > 1:
            unique_ids = set(recent_ids)
            if len(unique_ids) > 1 and person_id in unique_ids:
                result['warnings'].append("Rapid identity switching detected - verify accuracy")

        return result

    def _get_recent_identifications(self, seconds: int) -> List[str]:
        """Get recent identifications within time window."""
        # This would query the actual database
        # For now, return empty list
        return []

    def process_unknown_person(self, confidence: float, image: np.ndarray,
                             bbox: List[float], camera: str) -> Dict[str, any]:
        """Handle detection of unknown persons for safety context."""
        result = {
            'person_type': 'unknown',
            'estimated_age_category': 'unknown',
            'safety_alert': False,
            'recommended_action': 'monitor'
        }

        if not bbox or len(bbox) < 4:
            return result

        # Estimate age category based on size
        height = bbox[3]

        if height < 150:
            result['estimated_age_category'] = 'child'
            result['recommended_action'] = 'verify_if_family_member'
        elif height < 250:
            result['estimated_age_category'] = 'teen_or_adult'
            result['recommended_action'] = 'identify_person'
        else:
            result['estimated_age_category'] = 'adult'
            if self.config.alert_on_unknown_adult:
                result['safety_alert'] = True
                result['recommended_action'] = 'security_alert'

        return result

    def get_safety_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for improving toddler identification safety."""
        return {
            'training_tips': [
                f"Add more photos of {self.config.target_child} in different lighting",
                f"Include full-body shots of {self.config.target_child} for OSNet training",
                f"Train with {self.config.target_child}'s typical daily outfits",
                "Avoid photos with siblings in the same frame",
                "Include photos from multiple camera angles"
            ],
            'system_settings': [
                f"Keep {self.config.target_child}'s confidence threshold at 95%+",
                "Enable hybrid identification (OSNet + face + color)",
                "Set up size-based validation rules",
                "Monitor matches daily for false positives"
            ],
            'operational_tips': [
                "Mark any false positives immediately",
                "Review matches when other family members are present",
                "Calibrate height ranges based on your camera setup",
                "Test system during different times of day"
            ]
        }

    def log_identification_result(self, person_id: str, confidence: float,
                                 validation_result: Dict, camera: str):
        """Log identification result for analysis and improvement."""
        self.identification_history.append({
            'timestamp': datetime.now().isoformat(),
            'person_id': person_id,
            'confidence': confidence,
            'validation_result': validation_result,
            'camera': camera
        })

        # Keep only recent history (last 1000 entries)
        if len(self.identification_history) > 1000:
            self.identification_history = self.identification_history[-1000:]


# Global instance
_toddler_safety = None

def get_toddler_safety_system() -> ToddlerFocusedSafety:
    """Get global toddler safety system."""
    global _toddler_safety
    if _toddler_safety is None:
        config = SafetyConfig(
            target_child="Erik",
            min_confidence_toddler=0.95
        )
        _toddler_safety = ToddlerFocusedSafety(config)
    return _toddler_safety

def validate_erik_identification(confidence: float, image: np.ndarray,
                               bbox: List[float], method_scores: Dict[str, float] = None,
                               camera: str = None) -> Dict[str, any]:
    """Convenience function to validate Erik identification."""
    safety_system = get_toddler_safety_system()
    return safety_system.validate_toddler_identification(
        "Erik", confidence, image, bbox, method_scores, camera
    )