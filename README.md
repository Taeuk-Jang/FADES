# FADES: Fair Disentanglement with Sensitive Relevance

This is code of FADES: Fair Disentanglement with Sensitive Relevance for CVPR 2024 submission.

## Description of the repo
This repository contains the following files:
- train_FADES.py: This python file is to train FADES on CelebA dataset
- FADES-Colored MNIST.ipynb: This notebook contains training of FADES on C-MNIST dataset.
- module.py: This contains modules for various domains and methods.
- dataloader.py: This is dataloader to load dataset.
- utils.py: This contains loss calculation and etc.

- save: contains oracle digit and color classifiers on C-MNIST data, and trained fades model.
- data: contains C-MNIST dataset with 95% color bias.

To download the dataset, please visit the following websites and follow the instructions:
- Adult dataset: https://github.com/Trusted-AI/AIF360
- CelebA dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
