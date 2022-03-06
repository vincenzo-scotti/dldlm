#!/bin/bash
# NOTE this script must be run from the repository root
# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM tools uninstallation script."
  echo
  echo "Syntax: uninstall_environment.sh [-h|p]"
  echo "Options:"
  echo "-h     Displays help."
  echo "-p     Removes Pip3 virtual environment, if not specified removes Anaconda virtual environment."
  echo
  echo "Note: this script requires root privileges"
  echo
}
# Check options
use_pip=false
while getopts :hp: option;
do
  case "${option}" in
    h) help
       exit;;
    p) use_pip=true;;
  esac
done

# Delete environment
if $use_pip
then
  deactivate
  sudo rm -rf $DLDLM/dldlm/
else
  conda deactivate
  conda remove --name dldlm --all
fi
# Delete the added content to srcfile
# Create temporary .bashrc to host updates
tmpfile=$(mktemp /tmp/.bashrc)
# Init flag
reading=true
# Loop over file lines
while read line;
do
  # If the current line is the start of the excluded sequence stop reading
  if [ "$line" = "# Begin of extension for DLDLM" ];
  then
    reading=false
  fi
  # If reading copy the current line
  if $reading
  then
    echo $line >> $tmpfile
  fi
  # If the current line is the end of the excluded sequence start reading back
  if [ "$line" = "# End of extension for DLDLM" ];
  then
    reading=true
  fi
done < ~/.bashrc
# Update .bashrc
mv $tmpfile ~/
# Reload srcfile
source ~/.bashrc
