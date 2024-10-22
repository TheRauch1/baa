#!/bin/bash

# This script installs a conda environment with necessary packages
conda create -n gpuhub python=3.10 -y
conda init
source ~/.bashrc
conda activate gpuhub
pip install -r requirements.txt