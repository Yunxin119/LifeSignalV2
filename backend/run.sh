#!/bin/bash

# Configuration
PORT="${PORT:-5100}"
ENV="${ENV:-development}"
DEBUG="${DEBUG:-1}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup script...${NC}"
    ./setup.sh
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Start Flask server
echo -e "${GREEN}Starting Flask server...${NC}"
export FLASK_APP=app.py
export FLASK_ENV=$ENV
export FLASK_DEBUG=$DEBUG
export FLASK_RUN_PORT=$PORT

# Run with the Python from the virtual environment
python app.py

# Deactivate virtual environment when done
deactivate 