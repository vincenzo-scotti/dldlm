# DLDLM

Codebase for the paper "[Taming Pre-Trained Transformers into Discrete Variational AutoEncoders for Dialogue](https://www.overleaf.com/read/nnvywbkzvgjn)". 
This repository contains the implementation of the Discrete Latent Dialogue Language Model (DLDLM) described in the paper.

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments;
    - checkpoints generated during the experiments;
    - experiment configuration dumps;
    - experiment logs.
- `notebooks/` contains the directories to host:  
    - data exploration notebooks.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
    - directory to host the pre-trained models, and the references to download them.
- `src/` contains modules and scripts to: 
    - run training and evaluation steps;
    - interact with the trained models;
    - load and preprocess corpora.

For further details, refer to the `README.md` within each directory.

## Environment

To install all the required packages within an anaconda environment, run the following commands:

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

## Training

### Run

There is a script to train or fine-tune the model, it expects to have the `./src` in the Python path and all data sets to be downloaded and placed in the `./resources/data/raw/` directory. 

to add the directory to the Python, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/dldlm/src
```

To run the training in background execute:

```bash
nohup python ./src/bin/train_dldlm.py --config_file_path ./resources/configs/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

### Monitor

To connect to a remote server and monitor the training process via [Tensorboard](https://www.tensorflow.org/tensorboard) connect via ssh to your machine using a tunnel

```bash
ssh  -L 16006:127.0.0.1:6006 user@adderess
```

Start the Tensorboard server on the remote machine

```bash
tensorboard --logdir ./expertiments/path/to/tensorboard/
```

Finally connect to http://127.0.0.1:16006 on your local machine

## Evaluation

There is a script to run the final evaluation of the model, the requirements to run it are the same of the training script.

To run the evaluation in background execute:

```bash
nohup python ./src/bin/evaluate_static.py --config_file_path ./resources/configs/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
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