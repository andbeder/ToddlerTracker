# Toddler Position Tracking System (TPTS)

## Overview
System to map real-time toddler detections from camera pixels to yard map coordinates using pre-computed camera-to-map projections.

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Frigate Person Detection                                     │
│    - Detects people in camera frames                            │
│    - Provides bounding box: [x_center, y_center, width, height] │
│    - Normalized coordinates (0-1 range)                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Hybrid Tracker Identification                                │
│    - OSNet features (clothing/appearance)                        │
│    - Color histogram matching                                    │
│    - CompreFace (face recognition when visible)                  │
│    - Identifies toddler even when back is turned                 │
│    - Receives bbox from Frigate                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Coordinate Conversion                                         │
│    - Convert normalized bbox → pixel coordinates                 │
│    - Extract feet position (bottom-center of bbox)               │
│    - cam_x = x_center * camera_width                            │
│    - cam_y = (y_center + height/2) * camera_height              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Pixel Mapping Lookup                                          │
│    - Query camera_projections table                              │
│    - Find nearest (cam_x, cam_y) → (map_x, map_y) mapping       │
│    - Uses pre-computed CUDA projection data                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Position Storage                                              │
│    - Store in toddler_positions table                            │
│    - Fields: subject, camera, map_x, map_y, confidence, timestamp│
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Live Map Display                                              │
│    - Real-time position marker on yard map                       │
│    - Auto-update every 2 seconds                                 │
│    - Pulsing red marker at (map_x, map_y)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Frigate Events API
**Endpoint**: `http://localhost:5000/api/events?label=person&camera=<camera_name>`

**Response Structure**:
```json
{
  "id": "1759362729.850332-e0fqyb",
  "camera": "side_yard",
  "label": "person",
  "data": {
    "box": [0.654, 0.507, 0.054, 0.161],  // [x_center, y_center, width, height]
    "score": 0.586,
    "top_score": 0.580
  }
}
```

### 2. Hybrid Tracker
**Location**: `hybrid_identifier.py`

**Key Methods**:
- `identify_person(image, bbox)` - Main identification function
- `extract_all_features(image, bbox)` - Extract OSNet + Color + Face features
- Accepts bbox format: `[x, y, width, height]` in pixels

**Database**: `hybrid_matches` table
- Stores: subject, confidence, camera, bbox, timestamp
- Already has bbox column (TEXT format)

### 3. Camera Projections
**Database**: `camera_projections` table in `yard.db`

**Structure**:
```sql
CREATE TABLE camera_projections (
    id INTEGER PRIMARY KEY,
    camera_name TEXT NOT NULL,
    map_id INTEGER NOT NULL,
    pixel_mappings BLOB NOT NULL,  -- Pickled: [(cam_x, cam_y, map_x, map_y), ...]
    metadata TEXT,
    created_at TEXT,
    UNIQUE(camera_name, map_id)
);
```

**Pixel Mappings**:
- ~4.9M entries per camera (for 2560×1920 resolution with sample_rate=1)
- Format: `(cam_x, cam_y, map_x, map_y)` tuples
- Pre-computed using CUDA ray tracing (~21 seconds per camera)

### 4. Position Storage
**Database**: `toddler_positions` table in `matches.db`

**Structure**:
```sql
CREATE TABLE toddler_positions (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    camera TEXT NOT NULL,
    map_x INTEGER NOT NULL,
    map_y INTEGER NOT NULL,
    confidence REAL NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Implementation Details

### Bounding Box Formats

**Frigate Format** (normalized 0-1):
```python
[x_center, y_center, width, height]
# Example: [0.654, 0.507, 0.054, 0.161]
```

**Hybrid Tracker Format** (pixels):
```python
[x, y, width, height]  # Top-left corner
# Example: [1600, 900, 150, 300]
```

**Conversion**:
```python
# Frigate (normalized) → Pixel coords
camera_width = 2560
camera_height = 1920

x_center_norm, y_center_norm, width_norm, height_norm = frigate_bbox

# Convert to pixel coordinates (top-left)
x_px = int((x_center_norm - width_norm/2) * camera_width)
y_px = int((y_center_norm - height_norm/2) * camera_height)
w_px = int(width_norm * camera_width)
h_px = int(height_norm * camera_height)

bbox_pixels = [x_px, y_px, w_px, h_px]
```

### Feet Position Calculation

Since camera projection maps to ground plane, we want the position where the toddler is standing (feet), not the center of the bounding box.

**Option 1: Bottom-Center (Recommended)**
```python
# Best for ground-plane projection
cam_x = int(x_center_norm * camera_width)
cam_y = int((y_center_norm + height_norm/2) * camera_height)
```

**Option 2: Bottom-Third**
```python
# More robust to occlusion
cam_x = int(x_center_norm * camera_width)
cam_y = int((y_center_norm + height_norm/3) * camera_height)
```

### Pixel Mapping Lookup

**Challenge**: Exact pixel match unlikely due to:
- Quantization (multiple camera pixels → same map pixel)
- Toddler movement between frames
- Bounding box jitter

**Solution: Nearest Neighbor Search**
```python
def lookup_map_position(camera_name, map_id, cam_x, cam_y):
    """
    Find nearest pixel mapping for given camera position.

    Returns:
        (map_x, map_y) or None if not found
    """
    # Load pixel_mappings from database
    projection = get_camera_projection(camera_name, map_id)
    pixel_mappings = projection['pixel_mappings']

    # Convert to numpy for fast search
    mappings_array = np.array(pixel_mappings)
    cam_coords = mappings_array[:, :2]  # (cam_x, cam_y)
    map_coords = mappings_array[:, 2:]  # (map_x, map_y)

    # Find nearest camera pixel
    distances = np.sqrt((cam_coords[:, 0] - cam_x)**2 +
                       (cam_coords[:, 1] - cam_y)**2)

    # Threshold to avoid bad matches
    min_idx = np.argmin(distances)
    if distances[min_idx] > 50:  # Max 50 pixels away
        return None

    map_x, map_y = map_coords[min_idx]
    return int(map_x), int(map_y)
```

**Optimization**: Use KD-Tree for faster lookups
```python
from scipy.spatial import cKDTree

# Build tree once per camera (cache it)
tree = cKDTree(cam_coords)

# Fast lookup
distance, idx = tree.query([cam_x, cam_y])
if distance <= 50:
    map_x, map_y = map_coords[idx]
```

## Service Architecture

### New Service: Position Tracker

**File**: `position_tracker.py`

**Responsibilities**:
1. Monitor Frigate events for person detections
2. Extract bounding boxes
3. Call Hybrid Tracker for identification
4. Lookup map coordinates
5. Store positions in database

**Key Methods**:
```python
class PositionTracker:
    def __init__(self):
        self.projection_cache = {}  # Cache KD-Trees per camera
        self.hybrid_identifier = get_hybrid_identifier()

    def process_frigate_event(self, event):
        """Process a Frigate person detection event."""

    def convert_bbox_to_pixels(self, frigate_bbox, camera_width, camera_height):
        """Convert Frigate normalized bbox to pixel coords."""

    def get_feet_position(self, frigate_bbox, camera_width, camera_height):
        """Extract feet position from bbox."""

    def lookup_map_position(self, camera_name, map_id, cam_x, cam_y):
        """Find map coordinates for camera position."""

    def store_toddler_position(self, subject, camera, map_x, map_y, confidence):
        """Store position in database."""
```

### Integration with Detection Service

**Current**: Detection service polls cameras every 5 seconds for face recognition

**New Approach**: React to Frigate events
- Subscribe to Frigate MQTT events (real-time)
- Or poll Frigate events API every 2 seconds
- Process only new events since last check

## Implementation Plan

### Phase 1: Core Position Lookup (MVP)
- [ ] Create `position_tracker.py` with PositionTracker class
- [ ] Implement bbox conversion (Frigate → pixels)
- [ ] Implement feet position extraction
- [ ] Implement nearest-neighbor mapping lookup
- [ ] Add position storage to database

### Phase 2: Frigate Integration
- [ ] Add Frigate events API polling to detection service
- [ ] Extract bbox from events
- [ ] Pass bbox to Hybrid Tracker
- [ ] Trigger position lookup on toddler identification

### Phase 3: Optimization
- [ ] Cache KD-Trees for each camera (avoid rebuilding)
- [ ] Add position filtering (remove jitter/outliers)
- [ ] Implement trajectory smoothing
- [ ] Add confidence thresholds

### Phase 4: Live Map Enhancement
- [ ] Update live map to show historical path
- [ ] Add velocity/direction indicators
- [ ] Show dwell time in areas
- [ ] Add zone alerts (e.g., "toddler near pool")

## Configuration

### Camera Resolution
Update camera configs with actual resolution:
```yaml
cameras:
  backyard:
    detect:
      width: 2560
      height: 1920
```

### Projection Requirements
Each camera must have:
1. Camera pose (from COLMAP)
2. Saved projection mapping for active map
3. Resolution must match detection resolution

### Performance Targets
- Position update latency: < 3 seconds
- Map lookup time: < 10ms (with KD-Tree)
- Database storage: ~100 positions/hour per camera

## Testing Strategy

### Test Cases
1. **Bbox Conversion**: Verify Frigate bbox → pixel coords
2. **Feet Position**: Validate ground position extraction
3. **Map Lookup**: Test nearest neighbor search accuracy
4. **Edge Cases**:
   - Toddler at edge of camera FOV
   - Partial occlusion
   - Multiple people in frame
   - Fast movement

### Validation
- Compare map positions with ground truth
- Visual inspection on live map
- Check for position jumps/outliers
- Verify historical paths make sense

## Future Enhancements

### Multi-Camera Fusion
- Combine positions from multiple cameras
- Weighted average based on confidence
- Hand-off tracking between cameras

### Predictive Tracking
- Kalman filter for smooth trajectories
- Predict next position
- Detect unusual behavior

### Analytics
- Heatmaps of frequent locations
- Time spent in different zones
- Activity patterns throughout day

## References

### Related Files
- `camera_projection_cupy.py` - CUDA projection computation
- `hybrid_identifier.py` - Toddler identification
- `detection_service.py` - Current detection service
- `yard_manager.py` - Projection storage/retrieval
- `templates/map.html` - Live map display

### Database Tables
- `camera_projections` (yard.db) - Pixel mappings
- `hybrid_matches` (matches.db) - Identification results
- `toddler_positions` (matches.db) - Map positions
- `camera_poses` (poses.db) - Camera calibration

### API Endpoints
- `POST /add_toddler_position` - Store position
- `GET /get_last_toddler_position` - Latest position
- Frigate: `GET /api/events?label=person&camera=X` - Detection events

---

**Last Updated**: 2025-10-02
**Status**: Design Complete, Implementation Pending
