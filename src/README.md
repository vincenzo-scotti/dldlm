# Source code

This directory is used to host the project source code.
There is a single directory with all the executable scripts (`bin`) and three packages (`data`, `model` and `training`).

## Scripts

- `bin` contains the main scripts to repeat trainings and evaluations:
    - `train_dldlm.py` is used to train the hierarchical dialogue language model with discrete latent;
    - `refine_dldlm.py` is used to refine the dialogue language model on the desired task;
    - `evaluate_static.py` is used to evaluate any of the trained model on the empathetic task selected for the paper;
    - `evaluate_interactive.py` is used to chat directly with any of the trained models.

## Packages

- `data` contains the classes to interface with the conversational corpora as set of context-response pairs;
- `model` contains the classes for model configuration, model tokeniser and model implementations (there are various depending on the heads on top). 
  All classes have been realised extending the *GPT-2* ones provided in the *Transformers* library by *HuggingFace*;
- `training` contains additional utilitiess to carry on the training process, in this case a linear learning rate scheduler with warmup option, a beta parameter scheduler for KL annealing, metrics computation functions, and some visualisation functions.
