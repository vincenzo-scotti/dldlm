# Experiments

This directory is used to host the output files, checkpoints and logs generated during the experiments (both training and evaluation).

Pretraining experiments:
- [x] NLLt latent objective and LDA guidance (softening)
  - [x] with lower LR 
- [x] Weighted NLLt latent objective and LDA guidance
- [x] KL from LDA prediction
- [x] Multi-objective with BoW and without prior  ([PLATO]( https://aclanthology.org/2021.findings-acl.222.pdf )-like, for comparison)
- [x] Beta cyclical annealing and KL-Divergence latent objective with BoW ([βVAE]( https://aclanthology.org/N19-1021.pdf )-like, for comparison)

Adaptation experiments:
- [ ] NLLt latent objective
