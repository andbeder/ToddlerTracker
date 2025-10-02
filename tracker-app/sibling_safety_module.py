"""
Sibling Safety Module for Toddler Tracker
Prevents false identification between siblings through enhanced validation
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SiblingProfile:
    """Profile for a family member to prevent false identification."""
    name: str
    age_category: str  # 'toddler', 'child', 'teen', 'adult'
    typical_height_range: Tuple[int, int]  # in pixels for camera view
    distinctive_features: Dict[str, any]
    exclusion_rules: List[str]


class SiblingSafetyValidator:
    """Validates identifications to prevent sibling false positives."""

    def __init__(self):
        """Initialize sibling safety validator."""
        self.family_profiles = {}
        self.false_positive_history = []
        self.validation_rules = self._init_validation_rules()

        logger.info("Sibling safety validator initialized")

    def _init_validation_rules(self) -> Dict[str, any]:
        """Initialize validation rules for different age groups."""
        return {
            'toddler_vs_child': {
                'min_height_ratio': 0.5,  # Toddler should be significantly shorter
                'max_height_ratio': 0.8,
                'required_confidence_boost': 0.15,  # Need 15% higher confidence
                'body_proportion_check': True,
                'gait_analysis': True
            },
            'confidence_thresholds': {
                'toddler': 0.92,  # Very high threshold for toddlers
                'child': 0.85,
                'teen': 0.80,
                'adult': 0.75
            },
            'temporal_rules': {
                'rapid_switch_detection': True,  # Detect impossible quick switches
                'min_switch_time': 30,  # Minimum seconds between different person IDs
                'location_consistency': True
            }
        }

    def add_family_member(self, name: str, age_category: str,
                         typical_height_range: Tuple[int, int] = None,
                         distinctive_features: Dict = None):
        """Add a family member profile for validation."""
        self.family_profiles[name] = SiblingProfile(
            name=name,
            age_category=age_category,
            typical_height_range=typical_height_range or (0, 1000),
            distinctive_features=distinctive_features or {},
            exclusion_rules=[]
        )

        logger.info(f"Added family profile for {name} ({age_category})")

    def validate_identification(self, person_id: str, confidence: float,
                               image: np.ndarray, bbox: List[float],
                               method_scores: Dict[str, float] = None) -> Dict[str, any]:
        """
        Validate an identification result against sibling safety rules.

        Returns:
            Dict with 'valid', 'adjusted_confidence', 'warnings', 'reasons'
        """
        result = {
            'valid': True,
            'adjusted_confidence': confidence,
            'warnings': [],
            'reasons': [],
            'suggested_action': None
        }

        if person_id not in self.family_profiles:
            # Unknown person - apply general rules
            return result

        profile = self.family_profiles[person_id]

        # Apply age-specific confidence threshold
        required_threshold = self.validation_rules['confidence_thresholds'].get(
            profile.age_category, 0.80
        )

        if confidence < required_threshold:
            result['valid'] = False
            result['reasons'].append(f"Confidence {confidence:.3f} below required {required_threshold:.3f} for {profile.age_category}")

        # Check body proportions if bbox provided
        if bbox and len(bbox) >= 4:
            height_check = self._validate_body_proportions(person_id, bbox, image)
            if not height_check['valid']:
                result['valid'] = False
                result['reasons'].extend(height_check['reasons'])
                result['warnings'].append("Body proportions don't match expected profile")

        # Check for rapid identity switching (impossible scenarios)
        temporal_check = self._check_temporal_consistency(person_id)
        if not temporal_check['valid']:
            result['valid'] = False
            result['reasons'].extend(temporal_check['reasons'])
            result['suggested_action'] = "Review recent identifications for accuracy"

        # Apply sibling-specific rules
        sibling_check = self._apply_sibling_rules(person_id, confidence, method_scores)
        if not sibling_check['valid']:
            result['adjusted_confidence'] *= 0.7  # Reduce confidence
            result['warnings'].extend(sibling_check['warnings'])

        # Special handling for toddlers
        if profile.age_category == 'toddler':
            toddler_check = self._validate_toddler_identification(person_id, confidence, bbox, method_scores)
            if not toddler_check['valid']:
                result['valid'] = False
                result['reasons'].extend(toddler_check['reasons'])

        return result

    def _validate_body_proportions(self, person_id: str, bbox: List[float],
                                  image: np.ndarray) -> Dict[str, any]:
        """Validate body proportions against expected profile."""
        if person_id not in self.family_profiles:
            return {'valid': True, 'reasons': []}

        profile = self.family_profiles[person_id]

        # Calculate person height in image
        person_height = bbox[3]  # height from bbox

        # Check against expected height range
        min_height, max_height = profile.typical_height_range

        if min_height > 0 and max_height > 0:
            if person_height < min_height or person_height > max_height:
                return {
                    'valid': False,
                    'reasons': [f"Height {person_height}px outside expected range {min_height}-{max_height}px"]
                }

        # Additional body proportion analysis for toddlers
        if profile.age_category == 'toddler':
            head_body_ratio = self._calculate_head_body_ratio(bbox, image)
            if head_body_ratio < 0.2 or head_body_ratio > 0.5:  # Toddlers have larger head-to-body ratio
                return {
                    'valid': False,
                    'reasons': [f"Head-body ratio {head_body_ratio:.3f} not typical for toddler"]
                }

        return {'valid': True, 'reasons': []}

    def _calculate_head_body_ratio(self, bbox: List[float], image: np.ndarray) -> float:
        """Calculate head-to-body ratio for age verification."""
        try:
            x, y, w, h = [int(coord) for coord in bbox]

            # Estimate head region (top 30% of person)
            head_height = int(h * 0.3)
            body_height = h - head_height

            if body_height > 0:
                return head_height / body_height
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating head-body ratio: {e}")
            return 0.25  # Default reasonable ratio

    def _check_temporal_consistency(self, person_id: str) -> Dict[str, any]:
        """Check for impossible rapid identity switches."""
        # This would check recent identification history
        # For now, return valid - implement with actual history tracking
        return {'valid': True, 'reasons': []}

    def _apply_sibling_rules(self, person_id: str, confidence: float,
                           method_scores: Dict[str, float] = None) -> Dict[str, any]:
        """Apply specific rules to prevent sibling confusion."""
        result = {'valid': True, 'warnings': []}

        if not method_scores:
            return result

        profile = self.family_profiles.get(person_id)
        if not profile:
            return result

        # Check if OSNet and facial recognition agree
        osnet_score = method_scores.get('osnet', 0)
        face_score = method_scores.get('face', 0)

        # If face recognition is high but OSNet is low, be suspicious
        if face_score > 0.8 and osnet_score < 0.4:
            result['warnings'].append("Face match high but body characteristics don't match - possible sibling confusion")

        # If only face recognition is contributing, be more careful
        if face_score > 0.7 and osnet_score < 0.3 and method_scores.get('color', 0) < 0.3:
            result['warnings'].append("Identification primarily based on face only - verify clothing/body match")

        return result

    def _validate_toddler_identification(self, person_id: str, confidence: float,
                                       bbox: List[float] = None,
                                       method_scores: Dict[str, float] = None) -> Dict[str, any]:
        """Special validation for toddler identifications."""
        result = {'valid': True, 'reasons': []}

        # Toddlers require very high confidence
        if confidence < 0.92:
            result['valid'] = False
            result['reasons'].append(f"Toddler identification requires >92% confidence, got {confidence:.1%}")

        # Require at least 2 methods to agree for toddlers
        if method_scores:
            high_confidence_methods = sum(1 for score in method_scores.values() if score > 0.6)
            if high_confidence_methods < 2:
                result['valid'] = False
                result['reasons'].append("Toddler identification requires agreement from at least 2 methods")

        return result

    def log_false_positive(self, incorrect_id: str, correct_id: str,
                          confidence: float, method_scores: Dict[str, float] = None):
        """Log a false positive for system learning."""
        self.false_positive_history.append({
            'timestamp': datetime.now().isoformat(),
            'incorrect_id': incorrect_id,
            'correct_id': correct_id,
            'confidence': confidence,
            'method_scores': method_scores or {}
        })

        logger.warning(f"False positive logged: {incorrect_id} misidentified as {correct_id}")

    def get_safety_recommendations(self, person_id: str) -> List[str]:
        """Get safety recommendations for improving identification."""
        if person_id not in self.family_profiles:
            return ["Add family member profile for better validation"]

        profile = self.family_profiles[person_id]
        recommendations = []

        if profile.age_category == 'toddler':
            recommendations.extend([
                "Increase confidence threshold to 95%+",
                "Train with multiple distinct outfits",
                "Include full-body shots for OSNet training",
                "Ensure good lighting in training photos",
                "Add photos from multiple angles"
            ])

        return recommendations

    def create_erik_matthew_profile(self):
        """Create specific profiles for Erik (toddler) and Matthew (8-year-old)."""
        # Add Erik's profile
        self.add_family_member(
            name="Erik",
            age_category="toddler",
            typical_height_range=(80, 200),  # Adjust based on your camera setup
            distinctive_features={
                'typical_clothing_colors': ['blue', 'red', 'yellow'],
                'height_category': 'short',
                'body_type': 'toddler_proportions'
            }
        )

        # Add Matthew's profile
        self.add_family_member(
            name="Matthew",
            age_category="child",
            typical_height_range=(200, 400),  # Taller than Erik
            distinctive_features={
                'typical_clothing_colors': ['green', 'black', 'white'],
                'height_category': 'medium',
                'body_type': 'child_proportions'
            }
        )

        logger.info("Created Erik and Matthew safety profiles")


# Global instance
_sibling_validator = None

def get_sibling_validator() -> SiblingSafetyValidator:
    """Get global sibling safety validator."""
    global _sibling_validator
    if _sibling_validator is None:
        _sibling_validator = SiblingSafetyValidator()
        _sibling_validator.create_erik_matthew_profile()
    return _sibling_validator

def validate_identification_safety(person_id: str, confidence: float,
                                  image: np.ndarray, bbox: List[float],
                                  method_scores: Dict[str, float] = None) -> Dict[str, any]:
    """Convenience function for identification validation."""
    validator = get_sibling_validator()
    return validator.validate_identification(person_id, confidence, image, bbox, method_scores)