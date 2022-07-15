#!/bin/sh
# NOTE this script must be run from the repository root
# NOTE to run in background detached:
# $ nohup ./scripts/ablations.sh > hparam.out &

# Train model with different hyper-parameters
python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/ablations/ > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out

# NOTE to run single training in background detached:
# nohup python /path/to/script.py --config_file_path /path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
# NOTE to access Tensorboard remotely:
# ssh  -L 16006:127.0.0.1:6006 <user>@<host>
# Then connect to http://127.0.0.1:16006 on local machine