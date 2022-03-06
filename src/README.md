# Source code

This directory is used to host the project source code.
There is a single directory with all the executable scripts (`bin`) and three packages (`data`, `model` and `training`).

## Scripts

- `bin` contains the main scripts to repeat trainings and evaluations:
    - `bin/train_dldlm.py` is used to train the hierarchical dialogue language model with discrete latent;
    - `bin/train_rl_dldlm.py` is used to refine the dialogue language model on the desired task with the hybrid reinforcement learning objective;
    - `bin/evaluate_static.py` is used to evaluate any of the trained model on the empathetic task selected for the paper;
    - `bin/evaluate_interactive.py` is used to chat directly with any of the trained models.
- `bin/utils` contains the utility scripts to standardise and prepare the corpora
    - `bin/utils/data/standardization/` contains one script per data set to standardise. 
      All scripts are independent and can be run separately.
      Standardised corpora are serialised into CSV files;
    - `bin/utils/data/preparation/prepare_dialogue_corpus.py` is used to combine all the standardised corpora into a single one and serialise it into a CSV file.

## Packages

- `data` contains the classes to interface with the conversational corpora as set of turns and as set of episodes (entire dialogues);
- `model` contains the classes for model configuration, model tokeniser and model implementations (there are various depending on the heads on top). 
  All classes have been realised extending the *GPT-2* ones provided in the *Transformers* library by *HuggingFace*;
- `training` contains additional utilitiess to carry on the training process, in this case a linear learning rate scheduler with warmup option.
