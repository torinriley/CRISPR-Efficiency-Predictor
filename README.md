# CRISPR-Efficiency-Predictor

A machine learning model for predicting CRISPR-Cas9 editing efficiency using DNA sequence and epigenetic features.

## Overview
This repository provides an implementation of a deep learning-based model for predicting the efficiency of CRISPR-Cas9 gene editing. By incorporating both DNA sequence information and epigenetic features (e.g., chromatin accessibility and histone modifications), the model achieves high accuracy and minimal prediction error.

## Features
- Utilizes a convolutional neural network (CNN) for DNA sequence processing.
- Integrates epigenetic features (CTCF, DNase, H3K4me3, RRBS) to enhance prediction accuracy.
- Predicts CRISPR efficiency scores for target DNA sequences.

## Dataset
This project utilizes data from the **DeepCRISPR** dataset. For more information, visit the dataset's original repository and citation below:

- **Dataset**: [DeepCRISPR GitHub Repository](https://github.com/bm2-lab/DeepCRISPR)
- **Citation**: Guohui Chuai, Qi Liu et al. *DeepCRISPR: optimized CRISPR guide RNA design by deep learning*. 2018 (Manuscript submitted).

## Hugging Face Model Repository
You can also find this model on Hugging Face for easy access and deployment:
- [CRISPR Efficiency Model on Hugging Face](https://huggingface.co/torinriley/CRISPR-Efficiency)
