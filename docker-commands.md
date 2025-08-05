# GROT-NET Docker Commands Guide

## Quick Start

### 1. Build and Start the Experiment Environment
```bash
# Build and start the main experiment container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 2. Access the Container
```bash
# Enter the running container
docker-compose exec grot-net-experiment bash

# Or start a new interactive session
docker-compose run --rm grot-net-experiment bash
```

## Available Services

### Main Experiment Service
```bash
# Start the main experiment environment
docker-compose up grot-net-experiment

# Run with development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up grot-net-experiment
```

### Jupyter Notebook Service
```bash
# Start Jupyter Lab for interactive development
docker-compose --profile jupyter up jupyter

# Access Jupyter at http://localhost:8889
```

### Development Service
```bash
# Start development environment with hot reload
docker-compose --profile dev up dev

# Access development server at http://localhost:5000
```

## Common Commands

### Inside the Container
```bash
# Run the YouTube crawler
python youtube_engine_crawler.py --max-results 10

# Setup experiment structure
python setup_experiment.py

# Test dependencies
python -c "import librosa, yt_dlp, torch; print('All dependencies working!')"

# Check FFmpeg
ffmpeg -version

# Start Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Development Tools (in dev mode)
```bash
# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Data Management

### Volume Mounts
- `./data` → `/app/data` (raw, processed, features)
- `./results` → `/app/results` (experiment results)
- `./logs` → `/app/logs` (log files)
- `./models` → `/app/models` (trained models)
- `./notebooks` → `/app/notebooks` (Jupyter notebooks)

### Backup Data
```bash
# Create a backup of your data
docker-compose exec grot-net-experiment tar -czf /app/backup_$(date +%Y%m%d).tar.gz /app/data /app/results /app/models
```

## Production Deployment

### Production Mode
```bash
# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View production logs
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f
```

### Scaling
```bash
# Scale the experiment service
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale grot-net-experiment=3
```

## Troubleshooting

### Check Container Status
```bash
# List running containers
docker-compose ps

# View logs
docker-compose logs grot-net-experiment

# Check container resources
docker stats grot-net-nvh
```

### Debug Issues
```bash
# Enter container with debugging tools
docker-compose run --rm grot-net-experiment bash

# Check Python environment
python -c "import sys; print(sys.path)"

# Check FFmpeg installation
which ffmpeg && ffmpeg -version

# Test audio processing
python -c "import librosa; print('librosa version:', librosa.__version__)"
```

### Rebuild Everything
```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up --build
```

## Environment Variables

### Available Environment Variables
- `PYTHONPATH=/app` - Python path configuration
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `FFMPEG_BINARY=/usr/bin/ffmpeg` - FFmpeg binary path
- `FFPROBE_BINARY=/usr/bin/ffprobe` - FFprobe binary path
- `DEBUG=1` - Enable debug mode (dev only)
- `PRODUCTION=1` - Enable production mode

### Custom Environment
```bash
# Create .env file for custom environment variables
echo "CUSTOM_VAR=value" > .env
docker-compose up
```

## Performance Optimization

### Resource Limits
```bash
# Add to docker-compose.yml for resource limits
services:
  grot-net-experiment:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Multi-stage Build (for smaller images)
```bash
# Use the existing Dockerfile which already uses multi-stage build
docker-compose build --no-cache
```

## Monitoring and Logging

### View Real-time Logs
```bash
# Follow logs in real-time
docker-compose logs -f grot-net-experiment

# View logs for all services
docker-compose logs -f
```

### Monitor Resource Usage
```bash
# Check container resource usage
docker stats

# Check disk usage
docker system df
```

## Cleanup

### Remove Everything
```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

### Clean Docker System
```bash
# Remove unused containers, networks, images
docker system prune

# Remove everything including volumes (WARNING: deletes all data)
docker system prune -a --volumes
```

## Advanced Usage

### Custom Dockerfile
```bash
# Build with custom Dockerfile
docker-compose build --build-arg BUILD_ENV=production
```

### Multiple Environments
```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up

# Staging
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up
```

### Network Configuration
```bash
# Create custom network
docker network create grot-net-network

# Use in docker-compose.yml
networks:
  default:
    external:
      name: grot-net-network
```

This Docker setup provides a complete, isolated environment for the GROT-NET experiment with all dependencies pre-installed and properly configured! 