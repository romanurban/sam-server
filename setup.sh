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

# Fix for blinker installation, if needed
python3 -m pip install --ignore-installed blinker

# Install requirements using pip
python3 -m pip install -r ./requirements.txt


# Download the model file using wget
wget $MODEL_URL -O $MODEL_SAVE_PATH

# Run the application
python3 ./server.py
