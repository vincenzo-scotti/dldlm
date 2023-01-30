# Models

This directory is used to host the pre-trained model directories.
These directories contain both models and tokenisers trained during the experiments.
It is possible to download a zip archive with all the trained models and tokenisers at this [link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/Ecx_Q7u2fYBApI198ukuCUwBLY8vgZUjUVvSItZqV-z93g?e=vlZuxO).

The model in the `therapy_dldlm` directory is the final version.

Directory structure:
```
 |- models/
    |- dldlm_pretraining/
      |- added_tokens.json
      |- config.json
      |- merges.txt
      |- pytorch_model.bin
      |- special_tokens_map.json
      |- tokenizer_config.json
      |- vocab.json
    |- dldlm_finetuning/
      |- added_tokens.json
      |- config.json
      |- merges.txt
      |- pytorch_model.bin
      |- special_tokens_map.json
      |- tokenizer_config.json
      |- vocab.json
    |- therapy_dldlm/
      |- added_tokens.json
      |- config.json
      |- merges.txt
      |- pytorch_model.bin
      |- special_tokens_map.json
      |- tokenizer_config.json
      |- vocab.json
```