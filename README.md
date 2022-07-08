# DLDLM

Codebase for the paper "[Learning high-level structures in open-domain conversations for task oriented chatbots]()". 
This repository contains the implementation of the Discrete Latent Dialogue Language Model (DLDLM) described in the paper.

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments 
    - checkpoints generated during the experiments;
    - experiment configuration dumps;
    - experiment logs.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
    - directory to host the pre-trained models, and the references to download them.
- `scripts/` contains the scripts to:
    - automate environment installation and uninstallation;
    - automate corpora download and unpacking;
    - automate models download and unpacking;
    - automate experiments execution.
- `src/` contains modules and scripts to: 
    - run training steps;
    - run evaluation steps;
    - interact with the trained models;
    - preprocess corpora.

For further details, refer to the `README.md` within each directory.

## Installation

To ease the installation you can leverage the setup script, run it using the following command (from the project root directory):

```bash
./scripts/setup.sh
```

It will take care of installing all the packages 

To manually install all the required packages, instead, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n dldlm python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate dldlm
# Install packages
conda install pytorch=1.11.0 torchvision=0.12.0 -c pytorch
conda install -c conda-forge transformers=4.18.0
conda install -c conda-forge tensorboard=2.9.1 pandas scikit-learn matplotlib seaborn spacy jupyterlab
python -m spacy download en_core_web_sm
pip install spacytextblob
conda install -c plotly plotly
conda install -c conda-forge python-kaleido
```

## Chatting

There is a script available to chat directly with any of the models, it can be run using the following command:

```bash
python ./src/bin/evaluate_interactive.py
```

## References

If you are willing to use our code or our models, please cite our work through the following BibTeX entry:

```bibtex

```