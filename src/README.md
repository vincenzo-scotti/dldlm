# Source code

This directory is used to host the project source code.
There is a single directory with all the executable scripts (`bin`) and three packages (`data`, `model` and `training`).

## Scripts

- `bin` contains the main scripts to repeat trainings and evaluations:
    - `train_dldlm.py` is used to train the hierarchical dialogue language model with discrete latent;
    - `train_rl_dldlm.py` is used to refine the dialogue language model on the desired task with the hybrid reinforcement learning objective;
    - `evaluate_static.py` is used to evaluate any of the trained model on the empathetic task selected for the paper;
    - `evaluate_static_gen.py` is used to generate using any of the trained model responses on the samples task selected for human evaluation;
    - `evaluate_static_hlvl.py` is used to evaluate any of the trained model high-level dialogue structures;
    - `evaluate_interactive.py` is used to chat directly with any of the trained models.
- `bin/utils` contains the utility scripts to standardise and prepare the corpora, use the baseline models and prepare samples for evaluation
    - `data` contains the scripts to prepare data sets and samples for training and evaluation:
        - `standardization` contains one script per data set to standardise. 
        All scripts are independent and can be run separately.
        Standardised corpora are serialised into CSV files;
        - `preparation`
            - `prepare_dialogue_corpus.py` is used to combine all the standardised corpora into a single one and serialise it into a CSV file;
            - `prepare_human_evaluation_dialogue_corpus.py` is used to prepare the corpus for the empathetic evaluation and serialise it into a CSV file;
            - `sample_human_evaluation_dialogue_corpus.py` is used to select randomly the samples for the empathetic evaluation and serialise them into a CSV file.
            - `prepare_human_evaluation_samples.py` is used to combine the generated samples for the empathetic evaluation from the different systems to carry out the evaluation. 
              Samples and ground truth are printed in random order for each of the considered contexts. 
              The merged samples are also serialised into a CSV file.
    - `external_models`
        - `evaluate_baseline_static_gen.py` is used to extract the generative evaluation results from the baseline taken from the [ParlAI](https://parl.ai/docs/index.html) library. 

## Packages

- `data` contains the classes to interface with the conversational corpora as set of turns and as set of episodes (entire dialogues);
- `model` contains the classes for model configuration, model tokeniser and model implementations (there are various depending on the heads on top). 
  All classes have been realised extending the *GPT-2* ones provided in the *Transformers* library by *HuggingFace*;
- `training` contains additional utilitiess to carry on the training process, in this case a linear learning rate scheduler with warmup option.
