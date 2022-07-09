#!/bin/bash
# NOTE this script must be run from the repository root
# NOTE to run in background detached:
# $ nohup ./scripts/hparam_search.sh > hparam.out &

# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM hyper-parameters search script."
  echo
  echo "Syntax: hparam_search.sh [-h]"
  echo "Options:"
  echo "-h     Displays help."
  echo
}
# Check options
while getopts :h: option;
do
  case "${option}" in
    h) help
       exit;;
  esac
done

# Activate Anaconda environment
conda activate dldlm
# Extend Python path
export PYTHONPATH=$PYTHONPATH:$PWD
# Train model with different hyper-parameters
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/hparam_search/.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out

# NOTE to run single training in background detached:
# nohup python /path/to/script.py --config_file_path /path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
# NOTE to access Tensorboard remotely:
# ssh  -L 16006:127.0.0.1:6006 <user>@<host>
# Then connect to http://127.0.0.1:16006 on local machine