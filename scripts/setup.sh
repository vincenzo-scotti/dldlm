#!/bin/bash
# NOTE this script must be run from the repository root
# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM environment setup script."
  echo "This script will install the environment, set up environment related variables "
  echo "and download data (without preprocessing) and models."
  echo
  echo "Syntax: setup.sh [-h]"
  echo "Options:"
  echo "-h     Displays help."
  echo
}

# Install environment
bash ./scripts/install_environment.sh -a
# Download data
bash ./scripts/prepare_data.sh -d
