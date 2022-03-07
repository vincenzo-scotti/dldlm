# Scripts

This is the directory to host the scripts. 
Note that every script must be run from the repository root directory.
Every script has a help option to show how it works, run 
```bash
path/to/scprit.sh -h
```
to see the available options.

For a fast setup use the `setup.sh` script, it will install the anaconda environment and download the data sets.

## Environment installation (and uninstallation)

To install the environment, use the `install_environment.sh` script in the following way:
```bash
bash $DLDLM/scripts/install_environment.sh -dsp
```
This will install an Anaconda environment and add the automatic environment activation.

To uninstall the environment can use the `uninstall_environment.sh` script in the following way:
```bash
bash $DLDLM/scripts/uninstall_environment.sh
```

## Resources download

To download the data sets, standardise them and prepare the final corpus, 
use the `prepare_data.sh` script in the following way:
```bash
bash $DLDLM/scripts/prepare_data.sh -dsp
```

To download the pre-trained models use the `download_models.sh` script in the following way:
```bash
bash $DLDLM/scripts/download_models.sh
```

## Run experiments

To train the DLDLM models (small and medium) use the `run_train_dldlm.sh` script in the following way:
```bash
nohup bash $DLDLM/scripts/run_train_dldlm.sh &
```

To train the empathetic DLDLM models (small and medium) use the `run_train_rl_dldlm.sh` script in the following way:
```bash
nohup bash $DLDLM/scripts/run_train_rl_dldlm.sh &
```

To evaluate all the trained models (small and medium) use the `run_static_evaluation.sh` script in the following way:
```bash
nohup bash $DLDLM/scripts/run_static_evaluation.sh &
```

all scripts are though to executed in background (possibly on a remote server you want to detach from)
