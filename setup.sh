#!/bin/bash

# The location to save the model file, adjust as necessary
MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_SAVE_PATH="./sam_vit_h_4b8939.pth"

if ! command -v wget &> /dev/null; then
    echo "wget could not be found, please install wget"
    exit
fi

if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found, please install Python 3"
    exit
fi

# Clone the repository
git clone $GIT_REPO_URL
cd $(basename $GIT_REPO_URL .git)

# Checkout the specific branch, if not the default branch
if [ "$BRANCH_NAME" != "main" ]; then
    git checkout $BRANCH_NAME
fi

# Install requirements using pip
python3 -m pip install -r requirements.txt

# Fix for blinker installation, if needed
python3 -m pip install --ignore-installed blinker

# Download the model file using wget
wget $MODEL_URL -O $MODEL_SAVE_PATH

# Run the application
python3 app.py
