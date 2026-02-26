#!/bin/bash
# Run the 15-minute market collector
cd "$(dirname "$0")"
export $(cat ../.env | xargs)
python3 collector.py
