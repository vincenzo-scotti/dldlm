# NOTE this script must be run from the repository root
# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM environment export script."
  echo
  echo "Syntax: export_environment.sh [-h|p]"
  echo "Options:"
  echo "-h     Displays help."
  echo "-p     Exports a Pip environment, if not specified uses Anaconda virtual environment."
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

# Export environment
if $use_pip
  then
    source $DLDLM/dldlm/bin/activate
    pip freeze > $DLDLM/requirements.txt
  else
    conda activate dldlm
    conda env export | grep -v "^prefix: " > $DLDLM/environment.yml
fi
