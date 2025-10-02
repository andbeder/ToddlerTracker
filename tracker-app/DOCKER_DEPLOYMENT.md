# Docker Deployment Guide

## Architecture

The toddler tracker system is split into two Docker containers:

1. **Web UI** (`toddler-tracker-web`) - Flask application for visualization and configuration
2. **Detector Service** (`toddler-tracker-detector`) - Hybrid detection with CUDA acceleration

Both containers share databases via volume mounts and communicate through the shared filesystem.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.1+ support

### Install NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Build and Start Services

```bash
cd /home/andrew/toddler-tracker/tracker-app

# Build and start both services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f web
docker-compose logs -f detector
```

### 2. Access Web UI

Open browser to: `http://localhost:9000`

### 3. Stop Services

```bash
# Stop containers (keeps data)
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Configuration

### Environment Variables

Edit `docker-compose.yml` to configure:

**Web Service:**
- `FLASK_ENV` - Set to `production` or `development`

**Detector Service:**
- `CUDA_VISIBLE_DEVICES` - Which GPU to use (default: 0)
- `FRIGATE_URL` - Frigate NVR URL (default: http://frigate:5000)

### Frigate Integration

If running Frigate in Docker, add to the same network:

```yaml
# In your Frigate docker-compose.yml
services:
  frigate:
    networks:
      - toddler-tracker-network

networks:
  toddler-tracker-network:
    external: true
```

Or use host network mode in `docker-compose.yml`:

```yaml
detector:
  network_mode: host
  environment:
    - FRIGATE_URL=http://localhost:5000
```

## Volume Mounts

### Shared Data

Both containers access:
- `./yard.db` - Yard maps and camera projections
- `./matches.db` - Detection matches and positions
- `./config.db` - Configuration settings

### Read-Only Mounts (Web UI)

- `./ply_storage` - Point cloud files
- `./npy_storage` - Optimized binary point clouds

### Model Cache (Detector)

- `~/.cache/torch` - Pre-trained model weights (persisted)

## Building Individual Images

### Build Web UI Only

```bash
docker build -f Dockerfile.web -t toddler-tracker-web .
docker run -p 9000:9000 -v $(pwd)/yard.db:/app/yard.db toddler-tracker-web
```

### Build Detector Only

```bash
docker build -f Dockerfile.detector -t toddler-tracker-detector .
docker run --gpus all -v $(pwd)/matches.db:/app/matches.db toddler-tracker-detector
```

## Health Checks

Both containers include health checks:

**Web UI:**
```bash
docker exec toddler-tracker-web python3 -c "import requests; requests.get('http://localhost:9000/health')"
```

**Detector:**
```bash
docker exec toddler-tracker-detector python3 -c "import os; assert os.path.exists('matches.db')"
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
# Should include: "default-runtime": "nvidia"
```

### Database Locked Errors

SQLite doesn't handle concurrent writes well. If you see database locked errors:

1. Ensure only one detector container is writing
2. Consider using PostgreSQL for production
3. Reduce detection frequency in config

### Out of Memory

```bash
# Check GPU memory
docker exec toddler-tracker-detector nvidia-smi

# Reduce batch size or disable features in config.db
```

### Container Exits Immediately

```bash
# Check logs
docker-compose logs detector

# Run interactively to debug
docker-compose run --rm detector /bin/bash
```

## Production Deployment

### Use External Database

For production, replace SQLite with PostgreSQL:

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: toddler_tracker
      POSTGRES_USER: tracker
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  web:
    environment:
      - DATABASE_URL=postgresql://tracker:secure_password@postgres:5432/toddler_tracker
```

### Scaling Detector Service

Run multiple detector instances for different cameras:

```bash
# Start 3 detector instances
docker-compose up -d --scale detector=3
```

Configure each to monitor different cameras via environment variables.

### Resource Limits

```yaml
detector:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Updating

### Pull Latest Code

```bash
cd /home/andrew/toddler-tracker/tracker-app
git pull

# Rebuild and restart
docker-compose up -d --build
```

### Update Base Images

```bash
# Pull latest CUDA base image
docker pull nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

## Backup

### Backup Databases

```bash
# Stop services
docker-compose down

# Backup databases
tar -czf tracker-backup-$(date +%Y%m%d).tar.gz *.db ply_storage/ npy_storage/

# Restart
docker-compose up -d
```

### Restore Databases

```bash
docker-compose down
tar -xzf tracker-backup-YYYYMMDD.tar.gz
docker-compose up -d
```

## Monitoring

### View Resource Usage

```bash
# CPU/Memory usage
docker stats

# GPU usage
docker exec toddler-tracker-detector nvidia-smi -l 5
```

### Logs

```bash
# Follow all logs
docker-compose logs -f

# Last 100 lines from detector
docker-compose logs --tail=100 detector

# Export logs
docker-compose logs > tracker-logs.txt
```

## Networking

### Expose to Network

To access from other devices on your network:

```yaml
web:
  ports:
    - "0.0.0.0:9000:9000"  # Listen on all interfaces
```

Then access via: `http://<host-ip>:9000`

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name tracker.example.com;

    location / {
        proxy_pass http://localhost:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Development Mode

For development with live code reload:

```yaml
web:
  volumes:
    - .:/app  # Mount entire directory
  environment:
    - FLASK_ENV=development
  command: python3 app.py --reload
```

## Summary

**Start:** `docker-compose up -d`
**Stop:** `docker-compose down`
**Logs:** `docker-compose logs -f`
**Rebuild:** `docker-compose up -d --build`
**Clean:** `docker-compose down -v` (WARNING: deletes data)

For issues, check logs first: `docker-compose logs detector`
