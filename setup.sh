#!/bin/bash

# Install miniconda (this particular version installs htmd) - July 2020
curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh
bash Miniconda3-4.5.11-Linux-x86_64.sh
conda install dlscore python=3.6
source activate dlscore

# install htmd
conda install -c acellera -c psi4 -c conda-forge htmd
