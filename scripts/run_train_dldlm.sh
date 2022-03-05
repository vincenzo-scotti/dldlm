#!/bin/bash
# Train small model
python $DLDLM/src/bin/train_rl_dldlm.py --config_file_path $DLDLM/resources/configs/train_dldlm_small_emp_config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
# Train medium model
python $DLDLM/src/bin/train_rl_dldlm.py --config_file_path $DLDLM/resources/configs/train_dldlm_medium_emp_config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out

# NOTE to run in background detached:
# nohup python $DLDLM/path/to/script.py --config_file_path $DLDLM/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &

# NOTE to access Tensorboard remotely:
# ssh  -L 16006:127.0.0.1:6006 <USER>@<ADDRESS>
# Then connect to http://127.0.0.1:16006 on local machine