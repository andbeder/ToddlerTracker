"""
Quick size-based validation to distinguish Erik (toddler) from Matthew (8-year-old)
"""

def validate_erik_size(bbox, confidence, person_id):
    """
    Quick validation based on person size in camera view.

    Args:
        bbox: [x, y, width, height] - person bounding box
        confidence: detection confidence
        person_id: identified person name

    Returns:
        (is_valid, adjusted_confidence, reason)
    """
    if person_id != "Erik":
        return True, confidence, "Not Erik"

    # Erik should be notably smaller than Matthew
    person_height = bbox[3] if bbox and len(bbox) > 3 else 0

    # Adjust these thresholds based on your camera setup
    # These are example values - you'll need to calibrate
    ERIK_MAX_HEIGHT = 250  # pixels - adjust based on your cameras
    MATTHEW_MIN_HEIGHT = 300  # pixels - adjust based on your cameras

    if person_height > ERIK_MAX_HEIGHT:
        # Too tall to be Erik - likely Matthew
        return False, confidence * 0.3, f"Too tall for Erik: {person_height}px > {ERIK_MAX_HEIGHT}px"

    if person_height < 80:
        # Too small to be a person
        return False, confidence * 0.1, f"Too small: {person_height}px"

    # Size looks appropriate for Erik
    return True, confidence, "Size appropriate for toddler"

# Quick integration function
def apply_size_validation(matches):
    """Apply size validation to a list of matches."""
    validated_matches = []

    for match in matches:
        if 'bbox' in match and match.get('subject') == 'Erik':
            is_valid, adj_confidence, reason = validate_erik_size(
                match['bbox'],
                match.get('confidence', 0),
                match.get('subject', '')
            )

            if is_valid:
                match['confidence'] = adj_confidence
                validated_matches.append(match)
            else:
                print(f"Rejected Erik match: {reason}")
        else:
            validated_matches.append(match)

    return validated_matches