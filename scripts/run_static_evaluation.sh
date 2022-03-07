#!/bin/bash
# Move to repository directory to ensure paths work correctly
cd $DLDLM
# Evaluate small model
python $DLDLM/src/bin/evaluate_static.py --config_file_path $DLDLM/resources/configs/static_evaluation_dldlm_small.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
# Evaluate medium model
python $DLDLM/src/bin/evaluate_static.py --config_file_path $DLDLM/resources/configs/static_evaluation_dldlm_medium.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
# Evaluate small empathetic model
python $DLDLM/src/bin/evaluate_static.py --config_file_path $DLDLM/resources/configs/static_evaluation_dldlm_small_emp.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out
# Evaluate medium empathetic model
python $DLDLM/src/bin/evaluate_static.py --config_file_path $DLDLM/resources/configs/static_evaluation_dldlm_medium_emp.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out

# NOTE to run in background detached:
# nohup python $DLDLM/path/to/script.py --config_file_path $DLDLM/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
