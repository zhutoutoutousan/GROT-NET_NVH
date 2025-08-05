#!/bin/bash
# Quick start script for GROT-NET experiment

echo "Starting GROT-NET YouTube Crawler..."
python youtube_engine_crawler.py --output-dir ./data/engine_sounds

echo "Crawling completed! Check the results in ./data/engine_sounds/"
