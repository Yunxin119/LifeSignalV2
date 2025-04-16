#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up LifeSignal backend...${NC}"

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf venv
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python -m venv venv

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements
echo -e "${GREEN}Installing requirements...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Setup complete! Run ./run.sh to start the server.${NC}" 