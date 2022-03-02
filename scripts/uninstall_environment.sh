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
while getopts :hpa: option;
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
# TODO complete by removing the remaining part in .bashrc
# Reload srcfile
source ~/.bashrc
