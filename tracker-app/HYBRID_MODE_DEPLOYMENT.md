# Hybrid Mode Deployment

Run the **detector in Docker** while keeping the **web UI on the host** for easier debugging.

## Architecture

```
┌─────────────────────────┐
│  Host Machine           │
│                         │
│  ┌───────────────────┐  │
│  │ Tracker Web UI    │  │  ← Running locally (python3 app.py)
│  │ Port 9000         │  │
│  └─────────┬─────────┘  │
│            │             │
│            │ Shared DBs  │
│            ▼             │
│  ┌───────────────────┐  │
│  │ *.db files        │  │  ← Shared via volume mounts
│  └─────────┬─────────┘  │
│            │             │
│  ┌─────────▼─────────┐  │
│  │ Docker Container  │  │
│  │ Hybrid Detector   │  │  ← Running in Docker
│  │ (GPU enabled)     │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

## Setup

### 1. Disable Background Detection in Web UI

Already done! Background detection is disabled in `app.py`:

```python
# Background detection disabled - running in Docker container
# detection_service.start_background_detection()
```

### 2. Build Detector Image

```bash
cd /home/andrew/toddler-tracker/tracker-app

# Build the detector image
docker build -f Dockerfile.detector -t toddler-tracker-detector .
```

### 3. Start Detector Container

```bash
# Start detector using detector-only compose file
docker-compose -f docker-compose.detector-only.yml up -d

# View logs
docker-compose -f docker-compose.detector-only.yml logs -f
```

### 4. Start Web UI on Host

```bash
# In another terminal, start the local web app
python3 app.py
```

## How It Works

### Shared Databases

Both the detector container and host web UI access the same database files:

- **`yard.db`** - Yard maps and camera projections
- **`matches.db`** - Detection matches and positions
- **`config.db`** - Configuration settings

The detector container mounts these as volumes:

```yaml
volumes:
  - ./yard.db:/app/yard.db
  - ./matches.db:/app/matches.db
  - ./config.db:/app/config.db
```

### Network Mode

The detector uses `network_mode: host` so it can:
- Access Frigate at `http://localhost:5000`
- Share files with the host seamlessly
- No port mapping needed

## Managing the Detector

### Start Detector

```bash
docker-compose -f docker-compose.detector-only.yml up -d
```

### Stop Detector

```bash
docker-compose -f docker-compose.detector-only.yml down
```

### View Logs

```bash
# Follow logs in real-time
docker-compose -f docker-compose.detector-only.yml logs -f

# Last 100 lines
docker-compose -f docker-compose.detector-only.yml logs --tail=100
```

### Restart Detector

```bash
docker-compose -f docker-compose.detector-only.yml restart
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker-compose -f docker-compose.detector-only.yml up -d --build
```

### Check Status

```bash
# Container status
docker ps | grep toddler-tracker-detector

# GPU usage
docker exec toddler-tracker-detector nvidia-smi

# Health check
docker inspect toddler-tracker-detector --format='{{.State.Health.Status}}'
```

## Debugging

### Shell Into Container

```bash
docker exec -it toddler-tracker-detector /bin/bash
```

### Check Database Access

```bash
# From inside container
docker exec toddler-tracker-detector ls -lh /app/*.db

# Check write permissions
docker exec toddler-tracker-detector sqlite3 /app/matches.db "SELECT COUNT(*) FROM toddler_positions"
```

### Monitor Detection Activity

```bash
# Watch matches database grow
watch -n 1 'sqlite3 matches.db "SELECT COUNT(*) FROM toddler_positions"'

# View recent detections
sqlite3 matches.db "SELECT * FROM toddler_positions ORDER BY timestamp DESC LIMIT 10"
```

### Check Frigate Connectivity

```bash
# Test from inside container
docker exec toddler-tracker-detector curl http://localhost:5000/api/events
```

## Troubleshooting

### Detector Not Starting

```bash
# Check logs for errors
docker-compose -f docker-compose.detector-only.yml logs

# Common issues:
# 1. GPU not available - check nvidia-docker runtime
# 2. Database locked - make sure no other detector is running
# 3. Frigate not accessible - check FRIGATE_URL
```

### Database Locked Errors

SQLite doesn't handle concurrent writes well. If you see locked errors:

```bash
# Stop detector
docker-compose -f docker-compose.detector-only.yml down

# Check for lingering processes
ps aux | grep detector_service

# Restart detector
docker-compose -f docker-compose.detector-only.yml up -d
```

### GPU Not Detected

```bash
# Verify GPU access in container
docker exec toddler-tracker-detector nvidia-smi

# If this fails, check nvidia-docker runtime:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Web UI Can't See Detections

The web UI and detector share databases via the filesystem. Check:

```bash
# Verify databases exist and are being updated
ls -lh *.db
sqlite3 matches.db "SELECT COUNT(*) FROM toddler_positions WHERE timestamp > datetime('now', '-1 hour')"

# Check web UI is reading from same database
# In web UI logs, you should see matches being loaded
```

## Performance Tips

### Monitor Resource Usage

```bash
# Container stats
docker stats toddler-tracker-detector

# GPU memory
watch -n 1 'docker exec toddler-tracker-detector nvidia-smi'
```

### Reduce Detection Load

Edit `config.db` to reduce detection frequency:

```sql
sqlite3 config.db "UPDATE camera_settings SET enabled = 0 WHERE camera_name = 'unused_camera'"
```

### Clear Old Positions

```bash
# Keep only last 24 hours
sqlite3 matches.db "DELETE FROM toddler_positions WHERE timestamp < datetime('now', '-24 hours')"
```

## Development Workflow

### Typical Development Session

1. **Start detector in Docker**
   ```bash
   docker-compose -f docker-compose.detector-only.yml up -d
   ```

2. **Start web UI locally**
   ```bash
   python3 app.py
   ```

3. **Make changes to web UI code** - Flask auto-reloads

4. **Make changes to detector code**:
   ```bash
   # Rebuild and restart detector
   docker-compose -f docker-compose.detector-only.yml up -d --build
   ```

5. **Stop everything**:
   ```bash
   # Stop detector
   docker-compose -f docker-compose.detector-only.yml down

   # Stop web UI (Ctrl+C in terminal)
   ```

## Switching Back to Local Mode

To run everything locally again (no Docker):

1. **Stop detector container**:
   ```bash
   docker-compose -f docker-compose.detector-only.yml down
   ```

2. **Re-enable background detection in `app.py`**:
   ```python
   # Uncomment these lines:
   detection_service.start_background_detection()
   ```

3. **Restart web UI**:
   ```bash
   python3 app.py
   ```

## Production Migration

When ready to move web UI to Docker too:

1. Use the full `docker-compose.yml` instead:
   ```bash
   docker-compose up -d
   ```

2. Both services will run in Docker with shared volumes

## Summary

**Start detector:** `docker-compose -f docker-compose.detector-only.yml up -d`
**Start web UI:** `python3 app.py`
**View logs:** `docker-compose -f docker-compose.detector-only.yml logs -f`
**Stop detector:** `docker-compose -f docker-compose.detector-only.yml down`

This setup gives you the best of both worlds:
- ✅ Easy debugging of web UI (local Python, auto-reload)
- ✅ Isolated detector with GPU (Docker, no conflicts)
- ✅ Shared data through database files
