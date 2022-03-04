#!/bin/bash
# NOTE this script must be run from the repository root
# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM tools installation script."
  echo
  echo "Syntax: install_environment.sh [-h|p|a]"
  echo "Options:"
  echo "-h     Displays help."
  echo "-p     Uses Pip3 virtual environment, if not specified uses Anaconda virtual environment."
  echo "-a     Adds automatic virtual environment activation."
  echo
}
# Check options
use_pip=false
automatic_activation=false
while getopts :hpa: option;
do
  case "${option}" in
    h) help
       exit;;
    p) use_pip=true;;
    a) automatic_activation=true;;
  esac
done

# Create environment
if $use_pip
then # Crete virtualenv environment (assumes already installed GPU driver if needed)
  python3 -m venv ./dldlm
  source ./dldlm/bin/activate
  pip install -r ./requirements.txt
else  # Create anaconda environment
  conda env create -f ./environment.yml
fi
# Mark start of extensions to ~/.bashrc
echo "# Begin of extension for DLDLM" >> ~/.bashrc
# Create variable for project directory
echo "DLDLM=$PWD" >> ~/.bashrc
# Extend python path
echo "export PYTHONPATH=\$PYTHONPATH:\$DLDLM/src" >> ~/.bashrc
# Activate environment (if required)
if $automatic_activation  # Add automatic environment activation if required
then
  if $use_pip
  then
    echo "source \$DLDLM/dldlm/bin/activate" >> ~/.bashrc
  else
    echo "conda activate dldlm" >> ~/.bashrc
  fi
fi
# Move to project directory
echo "cd \$DLDLM" >> ~/.bashrc
# Mark end of extensions to ~/.bashrc
echo "# End of extension for DLDLM" >> ~/.bashrc
# Reload srcfile
source ~/.bashrc
