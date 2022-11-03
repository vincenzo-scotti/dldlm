# Models

This directory is used to host the pre-trained model directories.
These directories contain both models and tokenisers trained during the experiments.
It is possible to download a zip archive with all the trained models and tokenisers at [this link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/ESuBW6NcQZtLuvxiQVc9PhwBXn-H2K22oi2Z4F1Q_a9ajA?e=72vhXB).

Pre-trained models are also available through the *HuggingFace*'s [*Model Hub*](https://huggingface.co/models?filter=dldlm). 
They are loaded as *GPT-2* model to easy the re-use outside this project.
Refer to the chatbot API to use the models.

Directory structure:
```
 |- models/
    |- dldlm/
      |- added_tokens.json
      |- config.json
      |- merges.txt
      |- pytorch_model.bin
      |- special_tokens_map.json
      |- tokenizer_config.json
      |- vocab.json
    |- dldlm_therapy/
      |- added_tokens.json
      |- config.json
      |- merges.txt
      |- pytorch_model.bin
      |- special_tokens_map.json
      |- tokenizer_config.json
      |- vocab.json
```